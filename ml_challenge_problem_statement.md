# Amazon ML Challenge — Business Entity Resolution

## 1. Problem Statement

In large-scale commercial platforms, business identity data arrives from multiple independent sources. Each source contributes partial, noisy fragments of information about the same real-world entities.

These fragments share **no common identifiers**. The task is to determine which records refer to the same business.

### Objective

Build an **ML-based Entity Resolution (ER) solution** that, given business records from **3 independent data sources** with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

- **Source 1** is the deduplicated reference source.
- For every Source 1 entity, find **all matching records** from Source 2 and Source 3.
- A Source 1 entity may match **zero, one, or many** records.

> **Important:** This is a many-to-many matching problem from Source 1 → Source 2/Source 3, not simply a one-to-one classification problem.

---

## 2. File Format

All challenge files are **tab-separated values (`.tsv`)**.

Use an explicit tab separator:

```python
import pandas as pd

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

> **Warning:** Reading a `.tsv` without `sep="\t"` can silently produce a single column containing the entire line.

---

## 3. Data Description

Each source file contains the following columns:

| Column | Description |
|---|---|
| `entity_id` | Unique record identifier. Prefix indicates source: `S1-`, `S2-`, or `S3-`. |
| `business_name` | Business name; may contain abbreviations, legal suffixes, typos, or transliterations. |
| `business_address` | Business address; may be partial, reordered, incomplete, or landmark-based. |
| `country` | Country label. Training contains US and India; test additionally contains France. |

### Important: Country Is an Open Set

The test set contains **France**, even though France does not appear in training.

Therefore:

- Do **not** hard-code `{US, India}`.
- Do **not** filter out France.
- Do **not** one-hot encode country assuming only US and India exist.
- Every test entity must appear in the submission.

There is no separate `source` column.

The source is determined by:

1. The file in which the record appears.
2. The `entity_id` prefix (`S1-`, `S2-`, `S3-`).

---

## 4. Ground Truth

Training ground truth is stored in:

```text
train_ground_truth.tsv
```

It contains:

| Column | Description |
|---|---|
| `source1_entity_id` | Entity ID of a Source 1 record |
| `matched_entity_ids` | Comma-separated matching IDs from Source 2 and/or Source 3 |

An empty `matched_entity_ids` means the Source 1 entity has **no matches**.

---

## 5. Expected Noise Patterns

### Business Name Variations

Expect:

- Abbreviations:
  - `Corp` ↔ `Corporation`
  - `Pvt` ↔ `Private`
  - `Ltd` ↔ `Limited`
- Legal suffix inconsistencies
- DBA / trade names
- Punctuation differences
- `&` ↔ `and`
- Word-order changes
- Typos

### Address Variations

Expect:

- `Rd` ↔ `Road`
- `St` ↔ `Street`
- Transliteration differences
- Missing components
  - PIN code
  - state
  - other address components
- Landmark-based references
  - e.g. `Near SBI ATM`
- Municipal numbering differences
- Reordered address components

---

# 6. Dataset Structure

## Training

```text
dataset/
└── train/
    ├── train_source1.tsv
    ├── train_source2.tsv
    ├── train_source3.tsv
    └── train_ground_truth.tsv
```

## Test

```text
dataset/
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

There is **no ground truth for the test set**.

To measure performance locally, create a validation split from the training data and evaluate using **F₀.₅**.

---

# 7. Overall ML Pipeline

A useful conceptual pipeline is:

```text
Source 1
   │
   ├──────────────┐
   │              │
   ▼              ▼
Source 2        Source 3
   │              │
   └──────┬───────┘
          ▼
 Candidate Generation / Blocking
          │
          ▼
 Candidate Pairs
          │
          ▼
 Feature Engineering
          │
          ├── Name similarity
          ├── Address similarity
          ├── Country information
          └── Other matching features
          │
          ▼
 Matching Model
          │
          ▼
 Final Match Decision
          │
          ▼
 matching_results.tsv
```

### Key principle

**Blocking controls the recall ceiling.**

If a true match is not generated as a candidate, the final model cannot recover it.

---

# 8. Output Files

The final submission must contain:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

---

## 8.1 `matching_results.tsv`

This contains the final entity matches.

### Columns

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 entity ID |
| `matched_entity_ids` | Comma-separated matching Source 2/Source 3 IDs |

### Example

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

### Rules

- Every Source 1 test entity must have **exactly one row**.
- If there is no match, leave `matched_entity_ids` empty.
- No duplicate IDs within a list.
- Only Source 2 and Source 3 IDs are allowed.
- IDs must exist in the test set.

---

# 9. `candidate_pairs.tsv`

This contains the candidate set produced by the **final blocking/candidate-generation stage**.

It represents every Source 2/Source 3 record that the matching model actually considered for each Source 1 entity.

### Example

```text
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
S1-00002             S3-00004
S1-00003
```

### Important

If your pipeline has multiple blocking/filtering stages:

```text
Raw records
    ↓
Blocking 1
    ↓
Blocking 2
    ↓
Filtering
    ↓
FINAL CANDIDATES
    ↓
ML Matching Model
```

`candidate_pairs.tsv` must contain the **last candidate set**, i.e. the exact candidates fed to the matching model.

### Required relationship

Every final match must be present in the candidate set:

```text
matching_results ⊆ candidate_pairs
```

If a matched ID never appeared as a candidate, the pipeline contains a bug.

---

# 10. Submission Validation

A validator is provided:

```text
utils/validate_submission.py
```

Run:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Result

```text
PASS
```

with exit code `0` means the files satisfy the required structural rules.

An exit code of `1` means issues were found.

The validator checks formatting and consistency; it **does not calculate your score**.

---

# 11. Final Submission Package

The final submission is a ZIP archive:

```text
<team_name>_submission.zip
```

Recommended structure:

```text
<team_name>_submission.zip
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

### `code/business_entity_resolution/`

Must contain a self-contained, runnable version of the pipeline.

It should include:

- All source code under `src/`
- `README.md` with exact reproduction instructions
- `requirements.txt` or equivalent environment file
- Pinned dependency versions

Anyone should be able to reproduce:

```text
Training/Test Data
       ↓
    Blocking
       ↓
   Matching
       ↓
matching_results.tsv
candidate_pairs.tsv
```

using only the contents of this folder.

---

# 12. Important Constraints

### Output constraints

1. Follow the exact output format.
2. `matched_entity_ids` may contain only Source 2/Source 3 IDs.
3. Source 1 self-matches are invalid.
4. Every Source 1 test entity must appear.
5. Duplicate `source1_entity_id` rows are invalid.
6. Duplicate entity IDs inside a list are invalid.

### Model constraint

The final model must:

- Use a **MIT or Apache 2.0 licensed model**
- Have **up to 8 billion parameters**

---

# 13. Evaluation Metric — F₀.₅

The challenge uses:

> **F₀.₅ Score**

with **β = 0.5**.

This is a **precision-heavy** metric.

### Formula

$$
F_{0.5} =
\frac{1.25 \times Precision \times Recall}
{0.25 \times Precision + Recall}
$$

The score is calculated:

1. Per Source 1 entity.
2. Then macro-averaged across all Source 1 entities.

---

## Why F₀.₅ Matters

The metric gives more importance to **precision** than recall.

In entity resolution, a false merge can be particularly harmful because it incorrectly combines two distinct businesses.

Therefore:

```text
False Merge
    ↓
Two different businesses treated as one
    ↓
High penalty
```

The challenge explicitly states that F₀.₅ weights precision **2× over recall**.

---

# 14. Singleton Handling

A **singleton** is a Source 1 entity that has no true matching record.

Correct behavior:

```text
Ground Truth:
S1-00003 → []

Prediction:
S1-00003 → []
```

Score for that entity:

```text
1.0
```

But if you incorrectly predict a match:

```text
Ground Truth:
S1-00003 → []

Prediction:
S1-00003 → [S2-00123]
```

Score:

```text
0.0
```

### Key takeaway

Do not force every Source 1 record to have a match.

**Predicting "no match" correctly is valuable.**

---

# 15. F₀.₅ Example

Suppose:

### Prediction

```text
S1-00001 →
[
    S2-00047,
    S2-00193,
    S3-00812
]
```

### Ground Truth

```text
S1-00001 →
[
    S2-00047,
    S3-00812
]
```

Correct predictions:

```text
2
```

Predicted:

```text
3
```

Actual:

```text
2
```

Therefore:

```text
Precision = 2 / 3 ≈ 0.667

Recall = 2 / 2 = 1.0
```

And:

```text
F₀.₅ ≈ 0.714
```

---

# 16. Leaderboard

## Public Leaderboard

During the challenge:

- Uses a subset of the test set.
- Provides real-time performance feedback.

## Private Leaderboard

After the challenge:

- Uses the remaining test data.
- Determines the final evaluation.

## Final Rankings

Final rankings are based on the **private leaderboard**.

Predictions are submitted for the **full test set** in both cases; the scoring system applies the relevant split.

---

# 17. Methodology Documentation

The methodology document must describe:

### 1. Methodology

Explain the overall entity-resolution approach.

### 2. Candidate Generation / Blocking

Explain:

- How candidates are generated.
- Which blocking keys are used.
- Why the blocking strategy was selected.

### 3. Model Architecture

Explain:

- Matching model
- Feature engineering
- Training procedure
- Decision threshold / matching logic, if applicable

### 4. Other Relevant Information

Include anything important for reproducing or understanding the approach.

There is **no page limit**.

> Prioritize clarity and technical depth over brevity.

---

# 18. Academic Integrity & Fair Play

## 🚫 External Data Lookup Is Strictly Prohibited

The solution must use only the provided challenge data.

You may **not** use:

- Commercial entity-resolution APIs
- External entity-resolution services
- Government business-registration databases
- Geocoding APIs
- Internet-based business lookups
- External databases
- External data augmentation

### Examples of prohibited approaches

```text
Business Name
      ↓
Google Search / External API
      ↓
Business Identity
```

```text
Address
   ↓
Geocoding API
   ↓
Normalized Location
```

These are prohibited.

### Enforcement

Submitted code and methodology are reviewed.

Evidence of external data lookup can result in **immediate disqualification**.

---

# 19. Suggested Areas to Explore

The challenge specifically highlights:

### Candidate Generation / Blocking

Invest heavily in blocking because:

```text
Blocking Recall
      ↓
Upper bound on final Recall
```

If the correct entity is removed during blocking, the matching model cannot recover it.

### String Similarity

Potential features include:

- Jaccard similarity
- Levenshtein distance
- TF-IDF cosine similarity

Apply these to:

- Business names
- Addresses

### Country

Pay attention to country-specific address patterns.

Remember:

```text
Country ∈ Open Set
```

Do not assume:

```text
Country ∈ {US, India}
```

### Precision vs Recall

Because the evaluation metric is F₀.₅:

```text
Precision matters more than Recall
```

Avoid overly aggressive merging.

### Singletons

Do not neglect entities with no matches.

Correctly predicting:

```text
No Match
```

can receive full credit for that entity.

---

# 20. Recommended Mental Model

Think of the challenge as a **two-stage system**:

```text
┌──────────────────────────────────────┐
│ Stage 1: Candidate Generation       │
│                                      │
│ Source 1 → Possible S2/S3 matches   │
│                                      │
│ Goal: High Recall                   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Stage 2: Candidate Matching          │
│                                      │
│ Candidate pairs → Match / No Match  │
│                                      │
│ Goal: High Precision                │
└──────────────────┬───────────────────┘
                   │
                   ▼
        matching_results.tsv
```

This decomposition is important because the two stages have different objectives:

| Stage | Primary Goal |
|---|---|
| Blocking / Candidate Generation | **Recall** |
| Matching Model | **Precision** |

The final evaluation combines both through **F₀.₅**.

---

# 21. End-to-End Checklist

Before submission:

- [ ] Read all three source files correctly using `sep="\t"`.
- [ ] Understand Source 1 as the reference source.
- [ ] Generate candidates from Source 2 and Source 3.
- [ ] Ensure blocking does not remove true matches unnecessarily.
- [ ] Build name similarity features.
- [ ] Build address similarity features.
- [ ] Handle country as an open-set string field.
- [ ] Train and validate using the training ground truth.
- [ ] Evaluate using macro F₀.₅.
- [ ] Tune matching decisions with precision in mind.
- [ ] Handle singletons explicitly.
- [ ] Generate candidates first, then final matches.
- [ ] Ensure every final match exists in `candidate_pairs.tsv`.
- [ ] Generate one output row for every Source 1 test entity.
- [ ] Keep unmatched entities' ID lists empty.
- [ ] Remove duplicate IDs.
- [ ] Use only S2/S3 IDs in final matches.
- [ ] Run `validate_submission.py`.
- [ ] Package code, outputs, requirements, README, and methodology.
- [ ] Verify that the entire pipeline can be reproduced from the submission package.

---

## Core Takeaway

The challenge is fundamentally:

> **Given noisy business records from three independent sources, identify which Source 2 and Source 3 records represent the same real-world business as each Source 1 record.**

The strongest conceptual decomposition is:

```text
Noisy Business Records
          ↓
   Normalization /
   Feature Engineering
          ↓
 Candidate Generation
       (Blocking)
          ↓
 Candidate Pairs
          ↓
 ML Matching Model
          ↓
 Match / No Match
          ↓
 matching_results.tsv
```

with:

```text
Candidate generation → protect Recall
Matching model       → protect Precision
Evaluation            → F₀.₅
```

Source-derived from the official challenge problem statement.
