# ML Challenge 2026: Business Entity Resolution Solution Methodology

**Team Name:** Top-K Resolution  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present a high-scalability Machine Learning solution for large-scale **Business Entity Resolution** across three noisy data sources (`Source 1`, `Source 2`, `Source 3`). Our solution combines **Multi-Stage Chunked Candidate Blocking** (exact normalized name, informative token indexing, postal/house number blocks, and character 3-5 gram TF-IDF cosine retrieval) with a **LightGBM Pairwise Classifier** trained on hard-negative pairs, and an **Entity-Aware Decision Engine** optimizing for the macro-averaged $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA revealed several domain-specific noise patterns:
- **Name Variations**: Abbreviations (`Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`), punctuation differences, and domain name strings (`maurewilliamscolombier.com`).
- **Address Noise**: Partial addresses, missing state/city names, house/street number transpositions, and landmark references.
- **Sparse Fields**: Missing addresses (`NaN`) across records.

### 2.2 Solution Strategy

- **Approach Type**: Multi-Stage Indexed Blocking + Hard-Negative Mined LightGBM Classifier + Entity-Aware Thresholding.
- **Core Innovation**: Memory-efficient TF-IDF character n-gram cosine retrieval coupled with an entity-level decision engine optimizing singletons vs multi-matches for $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To avoid $1.7\text{M} \times 10\text{M}$ dense matrix multiplication:
1. **Stage 1**: Exact Normalized Name Hash Block.
2. **Stage 2**: Informative Token & 3-gram Inverted Index Lookup.
3. **Stage 3**: House Number + Postal Code Address Block.
4. **Stage 4**: Character 3-5 gram TF-IDF Approximate Top-K Cosine Retrieval.

- **Candidate Recall Ceiling Achieved**: **87.2%** on held-out validation set.
- **Search Space Reduction**: **>99.9%** reduction.

---

## 4. Matching Model

### Features Extracted:
- **Name Features**: Exact raw match, exact normalized match, token Jaccard similarity, character 3-gram Jaccard, normalized Levenshtein similarity, 3-char prefix match, relative length difference.
- **Address Features**: Address token Jaccard, house/building number exact match, postal code match indicator, missing address indicator.
- **Structural Features**: Country match indicator, `source_pair` flag ($S1 \rightarrow S2$ vs $S1 \rightarrow S3$).

### Model Architecture & Tuning:
- **Model Type**: LightGBM Binary Gradient Boosted Classifier.
- **Sampling Strategy**: True positive ground truth pairs + top non-matching candidate pairs (Hard Negatives).
- **Singleton Threshold Selection**: Grid search probability threshold $\tau \in [0.15, 0.85]$ evaluated on entity-level validation split.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score**: **0.9217**
- **Validation Macro Precision**: **0.9541**
- **Validation Macro Recall**: **0.8677**
- **Optimal Probability Threshold**: **0.20**

---

## 6. Conclusion

The combination of multi-stage candidate generation, hard-negative feature learning, and entity-level singleton thresholding provides a scalable and precision-focused entity resolution pipeline for commercial business data.
