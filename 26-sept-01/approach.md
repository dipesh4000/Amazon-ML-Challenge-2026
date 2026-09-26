# Business Entity Resolution --- Proposed Architecture

## 1. Objective

The objective is to resolve entities across Source 1, Source 2, and
Source 3 using **only the supplied challenge data**.

No external databases, APIs, geocoding services, business registries, or
internet-based entity enrichment are used.

The system is designed for a very large dataset (10M+ records), so the
architecture must prioritize:

-   high candidate-generation recall,
-   controlled memory usage,
-   scalable indexing,
-   hard-negative generation,
-   precise final matching,
-   correct handling of singleton/no-match entities,
-   and validation of the **entire retrieval-to-decision pipeline**.

The central design principle is:

> **Do not attempt expensive semantic matching against the entire
> dataset. First construct a high-recall suspected-answer space, then
> apply increasingly expensive models to that smaller space.**

------------------------------------------------------------------------

# 2. Overall Architecture

``` text
                         TRAIN / TEST DATA
                                |
                                v
                    +-----------------------+
                    | Canonicalization      |
                    | name / address /      |
                    | country / numerics    |
                    +-----------+-----------+
                                |
                                v
                  +-------------------------+
                  | MULTI-BLOCK GENERATION  |
                  |                         |
                  | exact-name blocks       |
                  | name-prefix blocks      |
                  | name n-gram blocks      |
                  | address-token blocks    |
                  | numeric/address blocks  |
                  | country-aware blocks    |
                  | hybrid blocks            |
                  +-----------+-------------+
                              |
                              v
                   SUSPECTED ANSWER POOL
                              |
                       candidate union
                              |
                              v
                   Blocking Recall Check
                              |
                              v
                 +------------------------+
                 | Semantic Retrieval     |
                 | ArcFace / XLM-R        |
                 | trained embeddings     |
                 +-----------+------------+
                             |
                             v
                      Top-K Candidates
                             |
                             v
                 +------------------------+
                 | Pair Verification      |
                 |                        |
                 | fuzzy similarities     |
                 | address similarity     |
                 | token overlap          |
                 | country consistency    |
                 | embedding similarity  |
                 | retrieval provenance   |
                 +-----------+------------+
                             |
                             v
                         LightGBM
                             |
                             v
                 +------------------------+
                 | Decision Layer         |
                 | threshold + margin     |
                 | singleton handling     |
                 +-----------+------------+
                             |
                             v
                      FINAL MATCHES
```

------------------------------------------------------------------------

# 3. Stage 0 --- Data Canonicalization

All processing starts from the supplied records.

No external normalization or enrichment is permitted.

For each record, retain:

-   raw business name,
-   raw address,
-   raw country,
-   source identifier,
-   entity identifier.

Generate normalized representations for retrieval and feature
computation.

## Name normalization

Apply deterministic transformations such as:

-   Unicode normalization,
-   case folding,
-   punctuation normalization,
-   whitespace normalization,
-   common corporate suffix removal where appropriate,
-   `&` normalization,
-   tokenization.

The raw value must remain available.

Example:

``` text
"ABC Restaurant Pvt. Ltd."
        |
        v
"abc restaurant"
```

## Address normalization

Generate multiple representations rather than relying on one aggressive
normalized string:

-   normalized full address,
-   address tokens,
-   numeric tokens,
-   distinctive tokens,
-   character n-grams,
-   optional postal/ZIP-like token representation when present.

Example:

``` text
"12-B, M.G. Road, New Delhi"
        |
        +--> normalized address
        +--> tokens
        +--> numeric tokens
        +--> character n-grams
```

## Country

Country is treated as a useful signal and blocking component, but not as
an assumption about the complete universe of countries.

Country equality should remain available as a pairwise feature.

------------------------------------------------------------------------

# 4. Stage 1 --- Multi-Block Candidate Generation

This is the first major modeling stage.

The goal is **not to decide whether two records are the same**.

The goal is:

> Find a sufficiently broad set of plausible candidates while keeping
> the candidate space computationally manageable.

For each Source 1 record, multiple independent blocking strategies are
executed.

## 4.1 Exact normalized-name block

Key:

``` text
country + normalized_name
```

This captures highly reliable exact-name matches.

------------------------------------------------------------------------

## 4.2 Name-prefix blocks

Generate several prefixes from the normalized name.

Example:

``` text
"starbucks coffee"

sta
star
starb
starbucks
```

Prefix length should be tuned using validation data.

Very short prefixes should be avoided because they create huge buckets.

------------------------------------------------------------------------

## 4.3 Name character n-gram blocks

Generate character n-grams from normalized names.

Example:

``` text
"restaurant"

res
est
sta
tau
aur
ura
ran
ant
```

The exact n-gram size should be experimentally selected.

This allows records with minor spelling and formatting differences to
enter the same candidate pool.

------------------------------------------------------------------------

## 4.4 Address-token blocks

Generate inverted indexes from distinctive address tokens.

Examples:

``` text
country + road token
country + locality token
country + postal/numeric token
country + distinctive address token
```

Very common tokens should be ignored or down-weighted.

------------------------------------------------------------------------

## 4.5 Numeric/address blocks

Numbers can be highly informative in addresses.

Examples:

``` text
building number
postal code
street number
suite/unit number
```

These should be represented separately so that a shared number can
become useful evidence without becoming an unconditional match.

------------------------------------------------------------------------

## 4.6 Hybrid blocks

Combine signals to produce more selective buckets.

Examples:

``` text
country + name_prefix + address_token

country + name_ngram + numeric_token

country + normalized_name + address_signature
```

Hybrid blocks reduce enormous buckets while retaining candidates that
may be missed by exact matching.

------------------------------------------------------------------------

# 5. Candidate Union

Each blocking strategy produces a candidate set.

For a Source 1 record:

``` text
C_name
C_prefix
C_ngram
C_address
C_numeric
C_hybrid
```

The initial candidate pool is:

``` text
C_initial =
    C_name
    UNION C_prefix
    UNION C_ngram
    UNION C_address
    UNION C_numeric
    UNION C_hybrid
```

The union is important because no individual blocking rule is expected
to achieve perfect recall.

Each candidate should retain provenance describing **why it entered the
pool**.

Example:

``` text
candidate_entity_id
bucket_type
bucket_key
bucket_rank
```

A candidate may have:

``` text
exact_name = true
address_token = true
name_ngram = true
```

This provenance becomes useful downstream.

------------------------------------------------------------------------

# 6. Bucket Management at 10M+ Scale

The blocking system must not store large Python objects or duplicated
DataFrame rows for every bucket.

Prefer compact integer identifiers and inverted indexes:

``` text
bucket_key -> [record_id_1, record_id_2, record_id_3, ...]
```

Use:

-   integer record IDs,
-   sorted arrays,
-   compact serialized indexes,
-   chunked processing,
-   memory-mapped structures where appropriate,
-   Parquet/Arrow for persistent tabular storage.

Avoid loading all 10M+ records into a single large pandas object.

The objective is to make:

``` text
10M+ records
        |
        v
compact blocking indexes
        |
        v
candidate IDs
```

rather than copying complete records into every bucket.

------------------------------------------------------------------------

# 7. Bucket-Level Validation

Before expensive embeddings or pair classification are introduced,
evaluate the blocking system itself.

For each validation Source 1 record, compare the candidate pool against
ground truth.

Measure:

``` text
Blocking Recall
Candidate count per Source 1
Median candidate count
Mean candidate count
95th percentile candidate count
Maximum bucket/candidate size
Fraction with zero candidates
```

The critical metric is:

``` text
Blocking Recall =
true matches appearing in candidate pool
-----------------------------------------
all true matches
```

This establishes the upper bound for later stages.

If the blocking stage misses a true entity, no downstream classifier can
recover it.

------------------------------------------------------------------------

# 8. Identity-Aware Train/Validation Split

Random pair-level splitting is not sufficient for the final evaluation
strategy.

Where the supplied ground truth allows entity clusters to be
reconstructed, validation should preferably be performed at the
**entity/identity cluster level**.

Conceptually:

``` text
Entity Cluster A
    S1-A
    S2-A
    S3-A

Entity Cluster B
    S1-B
    S2-B

Entity Cluster C
    S1-C
    S3-C
```

Clusters should be assigned consistently to train or validation.

This reduces leakage where nearly identical representations of the same
underlying entity appear on both sides of the split.

The final holdout should remain untouched while model and threshold
decisions are being developed.

------------------------------------------------------------------------

# 9. Stage 2 --- Identity-Oriented Semantic Retrieval

After high-recall blocking, semantic retrieval is applied to the
candidate space.

The preferred experimental direction is an **ArcFace-style
metric-learning model** based on a transformer encoder such as XLM-R.

The purpose of ArcFace here is not final classification.

Its purpose is to learn an embedding space in which:

``` text
same entity
    -> embeddings close together

different entities
    -> embeddings separated
```

The training identity/class should be derived from the supplied training
ground truth rather than external business information.

------------------------------------------------------------------------

# 10. Record Representation for the Encoder

The model should receive the supplied fields together rather than
independently embedding each field and blindly averaging the results.

Conceptually:

``` text
[name]
ABC Restaurant

[address]
12 MG Road, New Delhi

[country]
India
```

or an equivalent separator-based representation.

The model can therefore learn interactions between:

-   business name,
-   address,
-   country,
-   formatting variations,
-   and other supplied textual evidence.

Raw fields remain available for downstream deterministic features.

------------------------------------------------------------------------

# 11. ArcFace Training

The metric-learning stage should use known identity relationships
derived only from the challenge training data.

A simplified training flow is:

``` text
record
   |
   v
XLM-R encoder
   |
   v
embedding
   |
   v
ArcFace objective
   |
   v
identity-discriminative embedding space
```

The model should be evaluated using identity-aware validation.

The primary retrieval measurements are:

``` text
Recall@5
Recall@10
Recall@20
Recall@50
Recall@100
```

The exact K should be selected according to the downstream candidate
budget.

------------------------------------------------------------------------

# 12. Candidate Retrieval After Metric Learning

ArcFace embeddings are used to retrieve semantically similar candidates.

FAISS or another ANN implementation can be used for this stage.

The important principle is:

``` text
blocking candidates
       +
semantic candidates
       |
       v
candidate union
```

Semantic retrieval should not automatically replace deterministic
blocking.

The two systems provide complementary evidence.

Blocking captures exact/structured relationships.

Metric learning captures noisy textual variations.

------------------------------------------------------------------------

# 13. Stage 3 --- Pair Verification

The candidate pool is now small enough for richer pairwise computation.

For each candidate pair, calculate features such as:

## Name features

-   normalized exact equality,
-   RapidFuzz ratios,
-   Levenshtein-style similarity,
-   Jaro-Winkler,
-   token overlap,
-   Jaccard similarity,
-   character n-gram similarity.

## Address features

-   normalized address similarity,
-   token Jaccard,
-   character similarity,
-   shared numeric tokens,
-   postal/numeric agreement,
-   distinctive-token overlap.

## Semantic features

-   ArcFace embedding cosine similarity,
-   other embedding similarity where available.

## Retrieval features

-   semantic rank,
-   bucket rank,
-   number of blocking channels that retrieved the candidate,
-   bucket type indicators.

## Country features

-   country equality,
-   missingness indicators where appropriate.

------------------------------------------------------------------------

# 14. Candidate Provenance as a Feature

The candidate's route into the suspected-answer pool is itself evidence.

Example:

``` text
Candidate A:
    exact_name
    address_token
    semantic_top20

Candidate B:
    name_ngram only
```

Candidate A should be distinguishable from Candidate B.

Therefore retain features such as:

``` text
hit_exact_name
hit_prefix
hit_name_ngram
hit_address
hit_numeric
hit_hybrid
hit_semantic
number_of_retrieval_channels
best_retrieval_rank
```

These features can be supplied to the downstream classifier.

------------------------------------------------------------------------

# 15. Stage 4 --- LightGBM Verification

LightGBM is used as the final pairwise verifier.

Input:

``` text
pair features
    +
retrieval provenance
    +
semantic similarity
```

Output:

``` text
P(candidate is the same entity)
```

Training negatives should preferably come from the actual retrieval
system rather than arbitrary random records.

This creates **hard negatives** that resemble real mistakes the model
will encounter during inference.

Example:

``` text
Positive:
ABC Restaurant
ABC Restaurant Pvt Ltd
same address

Hard Negative:
ABC Restaurant
ABC Restaurant Services
similar name, different address
```

------------------------------------------------------------------------

# 16. Hard Negative Mining

For each training Source 1 record:

1.  retrieve candidates through the real blocking system,
2.  add known positive candidates,
3.  identify high-ranked non-matching candidates,
4.  retain the strongest negatives,
5.  train the verifier on these realistic mistakes.

This is preferable to generating millions of obviously unrelated random
negatives.

The number of negatives per Source 1 record should be tuned based on
memory and validation performance.

------------------------------------------------------------------------

# 17. Stage 5 --- Final Decision Layer

The classifier probability should not automatically become the final
answer.

Because the evaluation metric is F_0.5, the final decision should
explicitly control precision.

The decision layer can use:

``` text
probability threshold
+
best-vs-second-best margin
+
candidate support
+
singleton handling
```

Example:

``` text
Candidate A = 0.94
Candidate B = 0.41
```

has strong separation.

Whereas:

``` text
Candidate A = 0.94
Candidate B = 0.92
```

indicates ambiguity and should be treated differently.

The threshold and margin must be selected using validation data rather
than intuition.

------------------------------------------------------------------------

# 18. Singleton / No-Match Handling

Singletons must be treated as first-class cases.

A Source 1 entity with no true corresponding entity should not be forced
into a match merely because the system found a plausible candidate.

The decision system therefore needs an explicit rejection path:

``` text
candidate scores low
       |
       v
NO MATCH
```

rather than:

``` text
choose highest candidate regardless of confidence
```

This is especially important because correct singleton predictions
receive full metric credit.

------------------------------------------------------------------------

# 19. Full Validation Pipeline

The validation set must simulate inference as closely as possible.

The complete validation flow is:

``` text
Validation S1
     |
     v
canonicalization
     |
     v
multi-block candidate generation
     |
     v
blocking recall
     |
     v
ArcFace / semantic retrieval
     |
     v
candidate union
     |
     v
pair feature generation
     |
     v
LightGBM
     |
     v
threshold + margin
     |
     v
predicted entity set
     |
     v
F_0.5
```

Do not evaluate only the LightGBM classifier on artificially sampled
pairs.

The real question is:

> Can the complete pipeline recover the correct entity from the original
> search space?

------------------------------------------------------------------------

# 20. Metrics to Track

Track metrics separately for every stage.

## Blocking

``` text
Blocking Recall
Candidates / S1
Zero-candidate rate
Candidate distribution
```

## Semantic retrieval

``` text
Recall@5
Recall@10
Recall@20
Recall@50
Recall@100
```

## Pair classifier

``` text
pair precision
pair recall
PR-AUC where useful
```

## Final entity resolution

``` text
Macro F_0.5
Micro precision
Micro recall
Singleton accuracy
False positive rate
False negative rate
```

The primary final metric remains the challenge's F_0.5.

------------------------------------------------------------------------

# 21. Error Analysis

After every major experiment, inspect:

### False negatives

``` text
Why did the true candidate disappear?

blocking failure?
semantic retrieval failure?
pair classifier failure?
threshold too high?
```

### False positives

``` text
Why did an incorrect candidate win?

same name?
same address?
common business?
country ambiguity?
insufficient margin?
```

### Singleton errors

``` text
Why did the system force a match?
```

This allows the correct stage to be improved instead of blindly
increasing model complexity.

------------------------------------------------------------------------

# 22. Scaling Strategy for 10M+ Records

The pipeline should be designed as a streaming system.

Do not construct giant in-memory pandas tables.

Recommended data flow:

``` text
Raw TSV
  |
  v
chunked reader
  |
  v
normalized records
  |
  +--> compact blocking indexes
  |
  +--> Parquet / Arrow
  |
  v
candidate IDs
  |
  v
embedding batches
  |
  v
ANN index
  |
  v
candidate pairs
  |
  v
chunked pair features
  |
  v
LightGBM
```

Embeddings should be generated in batches and persisted incrementally.

Candidate pairs should also be processed in chunks rather than retained
as one massive DataFrame.

------------------------------------------------------------------------

# 23. Memory-Constrained Embedding Configuration

For Colab-scale experimentation, conservative FAISS settings can be used
initially:

``` python
EMBEDDING_INDEX_BATCH_ROWS = 5_000
EMBEDDING_TRAIN_ROWS = 30_000
EMBEDDING_NLIST = 1_024
EMBEDDING_PQ_M = 24
EMBEDDING_NPROBE = 24
EMBEDDING_TOP_K = 20
EMBEDDING_CHECKPOINT_ROWS = 500_000
RUN_TEST_INFERENCE = False
```

These values are intended for **pipeline development and stability**,
not as final quality settings.

For large-scale execution, the system should use a higher-memory
environment and benchmark the trade-off between:

``` text
index size
retrieval latency
retrieval recall
RAM usage
```

before selecting final FAISS parameters.

------------------------------------------------------------------------

# 24. Recommended Experiment Sequence

Do not introduce every component simultaneously.

## Experiment 1 --- Blocking baseline

``` text
normalization
    +
multi-block retrieval
```

Measure blocking recall.

## Experiment 2 --- Existing embedding baseline

``` text
blocking
    +
current multilingual embedding
    +
ANN
```

Measure Recall@K.

## Experiment 3 --- ArcFace retrieval

``` text
blocking
    +
ArcFace-trained embeddings
    +
ANN
```

Compare Recall@K against Experiment 2.

## Experiment 4 --- Verification

``` text
best candidate generation
    +
pair features
    +
LightGBM
```

Measure F_0.5.

## Experiment 5 --- Decision refinement

Tune:

``` text
threshold
margin
singleton rejection
candidate support
```

using validation only.

## Experiment 6 --- Final holdout

Freeze:

-   normalization,
-   blocks,
-   embedding model,
-   ANN parameters,
-   verifier,
-   threshold,
-   margin,
-   singleton logic.

Then evaluate exactly once on the untouched holdout.

------------------------------------------------------------------------

# 25. Final Recommended Architecture

The final target system is:

``` text
                         10M+ RECORDS
                              |
                              v
                    CANONICALIZATION
                              |
                              v
                  MULTI-BLOCK GENERATION
                              |
          +-------------------+-------------------+
          |                   |                   |
       NAME BLOCKS       ADDRESS BLOCKS      HYBRID BLOCKS
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                       CANDIDATE UNION
                              |
                              v
                    BLOCKING RECALL TEST
                              |
                              v
                ARCFace / XLM-R EMBEDDINGS
                              |
                              v
                     ANN RETRIEVAL
                              |
                              v
                    TOP-K CANDIDATES
                              |
                              v
                 CANDIDATE PROVENANCE
                              |
                              v
                    PAIRWISE FEATURES
                              |
                              v
                         LIGHTGBM
                              |
                              v
                  PROBABILITY + MARGIN
                              |
                    +---------+---------+
                    |                   |
                 MATCH              NO MATCH
                    |
                    v
                 F_0.5
```

## Core principle

The system should become progressively more expensive:

``` text
CHEAP + BROAD
      |
      v
blocking
      |
      v
MODERATE
      |
      v
ANN / semantic retrieval
      |
      v
EXPENSIVE + PRECISE
      |
      v
pairwise verification
      |
      v
FINAL DECISION
```

This architecture keeps the strongest ideas from the current retrieval +
LightGBM pipeline while adding the new identity-aware blocking and
ArcFace retrieval stage.

Most importantly, **blocking is treated as a learned/evaluated component
of the solution rather than a simple preprocessing step**. Its recall
determines the maximum possible performance of every downstream model.
