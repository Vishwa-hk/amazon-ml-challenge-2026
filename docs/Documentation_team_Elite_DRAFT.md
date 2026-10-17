# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Elite
**Team Members:** Vikhyaath Mallikarjun (leader), Vishwa H, Varun T
**Submission Date:** 27 September 2026

> DRAFT for team review. Numbers marked (final) are filled in once the last version is chosen.

---

## 1. Executive Summary

We resolve each Source-1 business against Source-2/3 records with a **blocking → learned
candidate scoring → two-stage judge** pipeline built on gradient-boosted trees (XGBoost, GPU).
Our main contributions are:

1. We reverse-engineered the noise recipe. Most importantly, **"twin" decoys** are copies of a
   real business whose house number is shifted by exactly +1, 2, 3, 4, 5, 7, 9, 11, 13 or 21 and
   whose name gets one word from a small decoy vocabulary. We turned both into features.
2. A **learned Indic-script→English word dictionary** and **web-handle word segmentation**,
   both built from the provided data only.
3. A **small submitted candidate set**: about 4.4 candidates per S1 entity with no measured F0.5
   loss, against a true-match average of 3.46.
4. A **label-free calibration for France** (no training labels): the judge's French probabilities
   are overconfident, so they are shifted down on the logit scale before the expected-F0.5 decision.

Held-out macro F0.5: **0.98594** (train, test-like universe). Public leaderboard: **0.982349** (final:
judge v17 + France calibration).

---

## 2. Methodology

### 2.1 Problem Analysis

- Train: 2.21M S1 entities (US 60%, India 40%), 10.3M S2/S3 records, 7.64M true pairs. Test:
  1.73M S1 (India 47%, US 38%, **France 15%, with no training labels**), 9.97M records.
- Each record belongs to **at most one** S1 entity. True pairs always share the country.
  5.6% of S1 entities have no match. Matches per entity average 3.46, with a maximum of 11.
- Row order and entity IDs are random (rank correlation ≈ 0.001), so there is no leakage.
- **The noise is synthetic and finite.** The names we observed include legal-form swaps,
  injected accents, look-alike characters (0↔O, 1↔l), honorifics, generic words added or
  swapped, word drops and shuffles, web-handle names ("pioneerresearch.com"), invented brand
  names, and word-by-word native-script transliteration. In addresses we saw street-type
  abbreviations, "street→saint", state code↔name, literal "null", city typos and swaps, digit
  drops and junk numbers.
- **Decoys are the core difficulty.** About 26% of train records (≈40% in test) belong to no
  S1 entity. Most are **twins** of a real business at the same street with a similar name.
  - We measured the recipe: the house number is shifted by exactly +{1,2,3,4,5,7,9,11,13,21}.
    For US pairs with similar names, a +1..15 offset marks 121,661 decoys against 3,601 true copies.
  - The name gets one word from a decoy vocabulary: northside, midtown, eastgate, riverside,
    lakeside, greater, uptown, holdings, group, …
  - The same pattern appears, label-free, in all three test countries, including France
    (participations, développement, groupe, SNC, "et fils").

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate scorer + two-stage classifier (hybrid rules + GBDT).
**Core Innovation:** Features that encode the generator's decoy recipe (signed house-number
offset, decoy-word log-odds), evidence from competing S1 entities and record clusters in the
judge, and a validation universe built to match test's decoy rate.

---

## 3. Candidate Generation (Blocking)

- **Normalisation before blocking**
  - Native-script names are mapped to English with a 1,312-word dictionary learned from
    aligned train pairs (96.7% held-out accuracy; 93.4% of test native words covered).
  - Glued web-handle names are split into words by a unigram Viterbi word-break over S1
    name-word frequencies.
- **Blocking keys** (inverted index, within country, frequency-capped): exact token-set name,
  compact name, core name; house number + rare name word; house number + rare address word;
  rare name word + rare address word; and "rarest-word" pairs of each side's 2–3 least
  frequent name/address words (buckets over 10K pairs skipped).
- **Internal short-list**: per S1, the top 100 by a cheap overlap score, the top 20 by an
  address-only score, plus a **record-side slot**: each no-address record keeps its top 5 S1
  entities by name overlap. This took no-address recall from ~70% to most of the gain in the
  line below.
  - Train recall 98.70% (oracle F0.5 0.9960); ~97 pairs per S1.
- **Submitted candidate set**: the stage-1 model scores every short-listed pair, and
  `candidate_pairs.tsv` keeps pairs with probability ≥ 0.01.
  - Test: **7.56M candidate pairs, 4.36 per S1 entity**, versus 167M internal pairs.
  - On train this cut costs nothing: F0.5 0.98310 vs 0.98309 with the full short-list. A
    fixed top-5 would cost 0.005.
- **How true matches were not lost**: rarest-word keys cover records whose words are all
  common. The address slot covers brand-name and native-script records. The record-side slot
  covers no-address records. Transliteration and segmentation restore shared words.

---

## 4. Matching Model

**Features used (52 + 13 stage-2):**
- **Name:** token-set / sort ratios, Jaro-Winkler on compact names, core-name equality, acronym
  match, rare-token overlap weighted by frequency, the original-script flag, legal-form and
  honorific handling.
- **Address:** token Jaccard / set ratio, shared and partial house numbers, city/state agreement,
  street-word canonical forms.
- **Twin-decoy:** signed first-number offset; whether an offset is in the decoy set; any new
  record number at a decoy offset; small negative offset; counts of new numbers; decoy-word
  log-odds of added and missing name words. The vocabulary is learned only from train
  entities excluded from training and scoring, and French words are mapped to English
  equivalents.
- **Context:** rank of the pair for the S1 and for the record, number of S1 candidates per
  record, score gaps, which blocking keys fired.
- **Stage 2 evidence:** stage-1 probability, its rank and gap within the S1 entity, the best
  rival S1's probability for the same record and the margin to it, the number of confident
  candidates, and support for the record's name and number from the S1's other confident
  records.
- **Stage 2 extras (v11–v16):** a copy-count prior (how many confident records the S1 already
  has), content-word swap counts, raw-name similarity and its margin over the record's other
  candidates (legal forms and punctuation kept), source balance (S2 vs S3 copies), word alignment
  (typo vs replaced word), and character-3-gram TF-IDF cosine of names and addresses with the
  name-cosine margin over rivals.

**Model type:** XGBoost (CUDA, 255-leaf lossguide trees), two stages, out-of-fold by S1 entity
(stage 1: 2 folds; final stage 2: 4 folds nested inside them, test = mean of the 4 models). Stage 2
uses stage-1 out-of-fold probabilities, so there is no leak.
**Threshold selection method:** per S1 entity, choose the top-k candidates that maximise
**expected F0.5** given the calibrated probabilities (floor 0.01). Each record is assigned to its
highest-probability S1 only (one owner). Rule selected on held-out F0.5 (an exact Poisson-binomial
version scored the same, 0.98552 vs 0.98551). **France:** probabilities are first shifted by −1.0 on
the logit scale. On test, US/India behave exactly like held-out data, but France has 4× more records
with best probability 0.05–0.5 and scored ~0.009 below the model's own expectation; the shift was
checked on the public leaderboard with only France changed (0.981638 → 0.98198).

---

## 5. Results & Error Analysis

- **F0.5 Score (macro), held-out on train:** 0.98594 (P 0.9967 / R 0.9651); India 0.98432,
  US 0.98702. Public leaderboard **0.982349** (final).
- Progression (leaderboard): v1 rules 0.72 → LightGBM judge 0.933 → test-like validation +
  two-stage judge → transliteration 0.958 → twin-decoy features 0.976 → French word mapping 0.977
  → French address normalisation 0.978 → word-swap features 0.9795 → raw-name rivalry, word
  alignment, TF-IDF 0.9816 → France calibration 0.9820 → 4-fold judge 0.9823 (final).
- **Common false positives:** twin decoys that share the house number, where only a
  swapped name word differs; decoys whose added word is itself common in true variants (group,
  partners, ltd).
- **Common false negatives:**
  1. Records with **no address** whose name matches several S1 entities with the same name in
     different cities. This is inherently ambiguous and costs about 0.003.
  2. True copies with ±1/±2 number typos or digit drops, which look like twins.
  3. The remaining short-list misses: invented brand names with partial addresses (0.004).

---

## 6. Conclusion

Most of our gains came from understanding the synthetic generator rather than from bigger
models: transliteration (+0.012), the twin-decoy recipe (+0.012), and blocking fixes for
no-address and web-handle records (+0.002). Building validation that mirrors test (decoy rate,
held-out vocabulary) kept local scores honest, with a gap of −0.008 to the leaderboard. The final
candidate set is about 25× smaller than the blocking output at no measured cost.

Varun explored an alternative blocking cascade (phonetic, prefix, acronym and character-trigram
keys). We measured it against our short-list misses: its best keys would recover at most 0.24% of
true pairs at ~16 extra pairs per S1, so the ~2 h rebuild was not worth it before the deadline.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/` (all source), `README.md` (exact commands, in order),
`requirements.txt`. Entry points: `run_translit.py`, `pipeline_v6.py blocking`,
`pipeline_v7.py vocab`, `pipeline_v8.py blocking|features|train|test`, `export_submission.py`
(writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official
validator).

### B. Additional Results
Candidate-set size vs held-out F0.5 (v8, 596K train entities):

| Submitted candidates | per S1 | F0.5 |
|---|---|---|
| full short-list | 97.2 | 0.98309 |
| stage-1 p ≥ 0.01 | 4.28 | 0.98310 |
| stage-1 p ≥ 0.05 | 3.67 | 0.98309 |
| fixed top-5 | 5.00 | 0.97809 |
