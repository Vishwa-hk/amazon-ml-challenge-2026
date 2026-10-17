# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team Elite  
**Team Members:** Vikhyaath M, Vishwa H, Varun T  
**Submission Date:** 2 October 2026

---

## 1. Executive Summary

We resolve each Source-1 business against the Source-2/3 records with a **blocking → stage-1 candidate scorer → stage-2 judge** pipeline built on gradient-boosted trees (XGBoost on GPU), with an expected-F0.5 decision per Source-1 entity. Most of our gains came from reverse-engineering the synthetic noise recipe — above all the **"twin" decoys** (copies of a real business with the house number shifted by a fixed offset and one name word swapped) — and turning it into features. We also contributed a **transliteration dictionary for Indic-script names** learned from the training data, a **4.4-per-entity submitted candidate set** with no measured accuracy loss, and a **label-free calibration for France**, which has no training labels.

**Final results:** held-out macro F0.5 **0.98594** (training data, test-like validation) · public leaderboard **0.982349**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Data.** Train: 2.21M S1 entities (US 60%, India 40%), 10.3M S2/S3 records, 7.64M true pairs. Test: 1.73M S1 entities (India 47%, US 38%, **France 15% — with no training labels**), 9.97M records.
- **Structure.** Each S2/S3 record belongs to **at most one** S1 entity, and true pairs always share the country. 5.6% of S1 entities have no match; matches per entity average 3.46 (maximum 11). Row order and entity IDs are random (rank correlation ≈ 0.001), so they carry no signal.
- **The noise is synthetic and comes from a finite recipe.**
  - *Names:* legal-form swaps (LLC↔Inc, Pvt Ltd↔Limited, SARL↔SAS), injected accents, look-alike characters (0↔O, 1↔l, 5↔S), honorifics (Sri, Shri, M/s), generic words added or swapped (services, group, holdings), word drops and shuffles, web-handle names ("pioneerresearch.com"), invented brand names, and word-by-word transliteration into Indic scripts (Devanagari, Bengali, Tamil, Telugu, Gujarati, Kannada, …).
  - *Addresses:* street-type abbreviations (Rd/Road, St/Street, and a "street→saint" corruption), state code↔name (VA↔Virginia), literal "null", city typos and swaps, digit drops and junk numbers, missing addresses. In France, records often name the **department** (Gironde, Nord) where the S1 address names the **region** (Nouvelle-Aquitaine, Hauts-de-France), and street types are abbreviated (R., Av, Bd, Imp, Rte, Ch, N°).
- **Decoys are the core difficulty.** About 26% of train records — and about 40% of test records — belong to no S1 entity. Most are **twins** of a real business: same street, similar name, but a different business.
  - *House number:* shifted by exactly **+1, 2, 3, 4, 5, 7, 9, 11, 13 or 21**, applied to any number in the address, including the last part of compound Indian numbers (27/3 → 27/4). For US pairs with similar names, an offset of +1 to +15 marks 121,661 decoys against only 3,601 true copies.
  - *Name:* one word swapped or added from a small decoy vocabulary (US: northside, midtown, eastgate, riverside, lakeside, greater, uptown, holdings, group…; India: overseas, infratech, exports…; France: participations, développement, groupe, SNC, "et Fils").
  - The same signature is visible without labels in all three test countries.
- **Test has more decoys than train**, so we validate in a **test-like universe** (Section 4).

### 2.2 Solution Strategy

**Approach Type:** Hybrid — rule-based normalisation and inverted-index blocking, then a two-stage gradient-boosted classifier (stage-1 candidate scorer, stage-2 judge) and an expected-F0.5 decision with one owner per record.

**Core Innovation:**
1. Features that encode the generator's decoy recipe: signed house-number offsets against the decoy offset set, and decoy-word log-odds.
2. Evidence from **competing S1 entities and record clusters** in the judge: rival margins, copy counts, and support from the S1's other confident records.
3. A validation universe built to match the test decoy rate, so local scores track the leaderboard.

---

## 3. Candidate Generation (Blocking)

**Normalisation before blocking**
- Case and accent folding, look-alike digit repair inside words, legal forms and honorifics removed, street types canonicalised, Indian 6-digit PIN codes excluded from house numbers.
- **Transliteration:** native-script names are mapped word by word to English with a **1,312-word dictionary learned from aligned training pairs** (96.7% held-out accuracy; 93.4% of test native words covered).
- **Web-handle names** ("tristatekeystonehawk.com") are split into words by a unigram word-break over S1 name-word frequencies.
- **France:** a department→region map is learned from co-occurrence in the address data (a region is an S1 address part that co-occurs with many cities; a department is a record-only part whose records name those cities). French street types are unified.

**Blocking keys used** (inverted index, always within country, every key frequency-capped):
- Token-set name; compact name (no spaces); core name (generic words removed).
- House number + rare name word; house number + rare address word; rare name word + rare address word.
- Five **"rarest-word" keys** built from each side's 2–3 least frequent name and address words. Buckets above 10K pairs are skipped.

**Internal short-list** (per S1 entity, then scored by the model):
- The top 100 records by a cheap overlap score (2 × name Jaccard + 2 × address Jaccard + shared number).
- The top 20 by an address-only score.
- A **record-side slot**: every record without an address keeps its top 5 S1 entities by name overlap. Seen from the S1 side these records rank below address-bearing candidates and were being cut.

Train recall of the short-list is **98.70%** of true pairs (oracle F0.5 ceiling 0.9960). On test the internal short-list has 167.4M pairs (96.6 per S1).

**Candidate pairs generated (submitted `candidate_pairs.tsv`):** the stage-1 model scores every short-listed pair, and only pairs with stage-1 probability **≥ 0.01** are submitted: **7,625,615 pairs = 4.40 per S1 entity**, 22× fewer than the internal short-list (true matches average 3.46 per entity). Every submitted match is drawn from this set. On training data this cut costs nothing (held-out F0.5 0.98310 with the cut vs 0.98309 with the full short-list); a fixed top-5 would cost 0.005.

**How we ensured true matches were not lost:**
- Rarest-word keys catch records whose words are all common.
- The address-only slot catches brand-name and native-script records whose names share nothing.
- The record-side slot catches records without addresses.
- Transliteration and web-handle segmentation restore shared words before blocking.
- Each change was checked against the list of true training pairs still missing. Short-list recall went from 58% (rule baseline) to 96% (frequency-capped and rarest-word keys), 97.6% (transliteration) and 98.7% (record-side slot and segmentation).

---

## 4. Matching Model

**Features used.** Stage 1 uses 52 pair features. The final stage-2 judge uses **95**: the 52, plus 13 stage-1 evidence features and 30 rivalry/name features.
- **Name features:** fuzzy ratio, token-set / token-sort / partial ratios, Jaro-Winkler on compact names, raw-string ratio, token Jaccard, document frequency of the rarest shared token, length and token-count differences, exact-key flags (token-set, compact, core name), a flag for originally non-Latin names. Character-3-gram **TF-IDF cosine** of the names (fitted on each split's own names) and its margin over the record's other candidates. **Word alignment** that separates a typo (high-similarity word pair) from a replaced word. **Content-word swap** counts (a word replaced vs added vs dropped).
- **Address features:** token-set ratio, fuzzy ratio, token Jaccard, document frequency of the rarest shared address word, empty-address flags, shared / partially shared house numbers, number counts, TF-IDF cosine of the addresses.
- **Twin-decoy features (11):** signed offset between the first house numbers; whether the offset is in the decoy set; whether any new record number equals an S1 number plus a decoy offset; small negative offsets; number of new numbers; decoy-word log-odds of words added to and missing from the record name. The vocabulary is learned only from training entities that are excluded from training and scoring; French legal forms and business words are mapped to English equivalents before the lookup.
- **Context and competition features:**
  - The pair's rank and score gap for the S1 and for the record, and the number of S1 candidates per record.
  - Which blocking keys fired, and the source (S2 or S3).
  - From stage 1: probability, rank, gap and sum within the S1; best rival S1's probability for the same record and the margin to it; number of rivals and of confident candidates.
  - Support for the record's name and house number from the S1's other confident records.
  - A **copy-count prior** (how many confident records the S1 already has, against the training distribution of copies per entity).
  - Raw-name similarity margin over the record's rivals (legal forms and punctuation kept).
  - Source balance (an entity's copies tend to be balanced across S2 and S3).

**Model type:** XGBoost (CUDA, histogram method), lossguide trees with up to 255 leaves, subsample 0.8, column sample 0.8, early stopping on a 5% validation split of the training entities. Stage 1 uses learning rate 0.08 and L2 1.0; the stage-2 judge uses learning rate 0.03, L2 2.0 and minimum child weight 2 (about 2,400–2,600 trees per fold). Both models are tree ensembles far below the 8B-parameter limit.
- **Stage 1** scores every short-listed pair. It is trained with 2-fold out-of-fold splits by S1 entity, so the stage-1 probabilities used downstream on training data are always out-of-fold.
- **Stage 2 (judge)** is trained on all training pairs with stage-1 probability ≥ 0.005 (a margin below the 0.01 submission cut), using all kept training entities. It uses **4 folds nested inside stage 1's 2**, so its inputs stay out-of-fold. The test prediction is the mean of the 4 fold models.

**Validation — the test-like universe.** We randomly remove 19% of the training S1 entities. Their records stay in as unmatched records, which raises the decoy rate to the ~40% seen in test. The removed entities are never trained on or scored, and the decoy-word vocabulary is learned only from them. All held-out scores below are macro F0.5 in this universe, computed with a scorer that parses files exactly like the official validator.

**Threshold selection method:** per S1 entity, choose the top-k candidates that maximise **expected F0.5** given the model's probabilities, including the expected-F0.5 of predicting nothing (which scores 1.0 when the entity has no match). Each record is first given to its single most probable S1 entity (one owner per record). The rule variant was selected on held-out F0.5; an exact Poisson-binomial computation of expected F0.5 gave the same score (0.98552 vs 0.98551).
**France calibration:** France has no labels. On test, US and India predictions behave exactly like the held-out data (same matches per entity, empty rate and uncertainty band). France, however, has 4× more records whose best probability lies between 0.05 and 0.5, and its leaderboard-implied score was about 0.009 below the model's own expectation beyond the US/India gap. So French probabilities are shifted by **−1.0 on the logit scale** before the same decision rule; US and India are unchanged. The shift was set using one public-leaderboard probe in which only France changed (0.981638 → 0.98198).

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** held-out **0.98594** (precision 0.9967, recall 0.9651) — India 0.98432, US 0.98702. **Public leaderboard: 0.982349** (final submission).
- **Progression (public leaderboard):** rules 0.721 → LightGBM judge 0.933 → test-like validation + two-stage judge + transliteration 0.958 → France threshold 0.960 → twin-decoy features + record-side slot 0.976 → French word mapping 0.977 → French address normalisation + copy-count prior 0.978 → word-swap features 0.9795 → raw-name rivalry, word alignment and TF-IDF features 0.9816 → France calibration 0.9820 → 4-fold judge **0.9823**.
- **Where the remaining held-out loss sits** (≈0.014): true pairs missing from the short-list (≈0.004, the oracle ceiling), true candidates the judge rejects, and a small number of false matches (precision is 0.997).
- **Common false positives (wrong merges):**
  - Twin decoys that share the house number, where only a swapped name word differs.
  - Decoys whose added word is also common in true variants (group, partners, holdings).
  - France (unlabelled, so observed rather than measured): most uncertain pairs are generic association-style names in the same town that differ by one word (Amicale, École, Club, Comité).
- **Common false negatives (missed matches):**
  1. Records **without an address** whose name fits several same-named S1 entities in different cities. This is ambiguous from the record alone (about 0.003 of F0.5), so the model correctly prefers to abstain.
  2. True copies with ±1/±2 house-number typos or digit drops, which look like twins.
  3. Remaining short-list misses: invented brand names with partial or missing addresses (≈0.004).

---

## 6. Conclusion

Understanding the synthetic generator mattered more than model size: transliteration (+0.012), the twin-decoy recipe (+0.012), blocking fixes for no-address and web-handle records (+0.002) and France-specific handling (about +0.005 on the leaderboard across several steps) carried most of the progress. Validating in a universe that mirrors test's decoy rate kept local scores honest: the final local-to-leaderboard gap was −0.004, almost all of it from unlabelled France. The submitted candidate set is 22× smaller than the internal short-list at no measured cost.

---

## 7. Other Relevant Information

- **Compliance.** No external data, services, APIs or pretrained models are used. Every learned artefact (transliteration dictionary, decoy-word vocabulary, models) comes from the provided training split. Models: XGBoost (Apache-2.0); LightGBM (MIT) was used only in early versions.
- **Use of test inputs (label-free only).** The France department→region map is learned from address co-occurrence, which includes the unlabelled test records because France appears only in test. Character-n-gram TF-IDF weights are fitted on each split's own names and addresses. The France calibration shift was chosen with public-leaderboard feedback, as described in Section 4. All matching decisions are made by the code.
- **Other ideas we tested and rejected** (held-out):
  - Averaging several judge versions: no gain.
  - A third stacking stage: worse, from cross-fold leakage.
  - Larger trees (511 leaves): no gain.
  - Retrieving 150 instead of 100 candidates per entity: +0.08% true pairs for +40% pairs.
  - Adding each record's best guess when probability ≥ 0.5: −0.001. F0.5 punishes a wrong match about 4× more than a miss.
  - An alternative blocking cascade explored by Varun (phonetic, 3-character prefix, acronym and character-trigram keys). Measured against our short-list misses, its best keys would recover at most 0.24% of true pairs at ~16 extra pairs per entity, so it was not merged.
  - An 8-fold judge reached 0.98604 held-out (+0.0001) but was finished after the submission window.
- **Compute.** One Windows PC: RTX 5060 Ti 16 GB, 31 GB RAM, 12 cores. A full rebuild from raw data takes about 3.5 hours. Test inference with trained models takes about 1 hour. Memory, not GPU, is the binding constraint.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains `src/` (all source), `README.md` (exact commands in order, with timings) and `requirements.txt` (pinned; Python 3.14, CUDA build of XGBoost). Pipeline entry points, run from `src/`:

| Step | Command | Produces |
|---|---|---|
| 1 | `run_translit.py` | transliteration dictionary |
| 2–3 | `pipeline_v6.py blocking --split train`, `pipeline_v7.py vocab` | decoy-word vocabulary |
| 4 | `pipeline_v8.py blocking --split train / test` | short-lists |
| 5 | `pipeline_v8.py features --split train` | training pair features |
| 6 | `pipeline_v8.py train --tag gpu_slot` | stage-1 scorer, its out-of-fold probabilities and the evidence features derived from them |
| 7 | `judge_v17.py train --tag v17` | final stage-2 judge (4 folds) |
| 8–10 | `pipeline_v11.py blocking / features --split test --fr-map`, then `pipeline_v11.py test --predict-only` | France re-blocked on normalised addresses; test pair features; stage-1 test probabilities |
| 11 | `judge_v17.py test --tag v17 …` | stage-2 test probabilities |
| 12 | `export_v17.py --p2 … --rule '<France logit_shift −1.0>' --out-dir output` | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, then runs the official validator |

Every step refuses to overwrite existing outputs. Seeds are fixed: fold hash 7 (sub-fold 13), test-like universe 11, XGBoost 42.

### B. Additional Results

**Submitted candidate-set size vs held-out F0.5** (training data):

| Submitted candidates | per S1 | F0.5 |
|---|---|---|
| full internal short-list | 97.2 | 0.98309 |
| stage-1 probability ≥ 0.01 (used) | 4.28 | 0.98310 |
| stage-1 probability ≥ 0.05 | 3.67 | 0.98309 |
| fixed top-5 | 5.00 | 0.97809 |

**Development history** (held-out macro F0.5 in the test-like universe; public leaderboard where uploaded):

| Version | Change | Held-out | Leaderboard |
|---|---|---|---|
| v6 | transliteration + two-stage GPU judge | 0.97007 | 0.958473 |
| v8 | twin-decoy features, record-side slot, web-handle segmentation | 0.98313 | 0.97556 |
| v9 | stage 2 retrained on the submitted region, all entities | 0.98418 | — |
| v11 | copy-count prior; French address normalisation | 0.98425 | ≈0.978 |
| v12b | word-swap features | 0.98434 | 0.979541 |
| v13 | raw-name rivalry, source balance | 0.98468 | — |
| v15 | word alignment | 0.98509 | — |
| v16 | character-3-gram TF-IDF | 0.98551 | 0.981638 |
| v16 + France calibration | France logit shift −1.0 | 0.98551 | 0.98198 |
| **v17 (final)** | 4-fold judge + France calibration | **0.98594** | **0.982349** |
