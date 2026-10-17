# Business Entity Resolution — Team Elite (Amazon ML Challenge 2026)

Code that turns the provided dataset into `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. All matching decisions are made by this code: rules plus
gradient-boosted trees (XGBoost). No external data, services, APIs or pretrained
models are used. Everything learned (the transliteration dictionary, the decoy-word
vocabulary, the models) is learned from the provided **train** split only.

## Pipeline at a glance

```
raw TSVs ──► normalisation ──► blocking (inverted-index keys) ──► pair features ──► stage-1 scorer ──► candidate_pairs.tsv
             · Indic-script names → English words (learned dictionary)     (~97 pairs / S1, internal)    (stage-1 p ≥ 0.01: ~4.4 / S1)
             · web-handle names split into words                                                               │
                                                                                                    stage-2 judge (+ rival and
                                                                                                    cluster evidence) ──► matching_results.tsv
                                                                                                    (expected-F0.5 decision, one owner per record)
```

1. **Normalisation** (`normalize.py`, `normalize_v2.py`, `translit.py`, `segment.py`): case, accents,
   look-alike characters, legal forms, honorifics, street-type abbreviations; native-script
   (Devanagari, Bengali, Tamil, Telugu, Gujarati, Kannada, …) words mapped to English words
   with a dictionary learned from aligned train pairs; glued web-handle names
   (`tristatekeystonehawk.com`) split into words by a unigram word-break over S1 name words.
2. **Blocking** (`blocking_v2.py`, `blocking_v3.py`, `blocking_v8.py`): 11 inverted-index key
   families within country (token-set / core name, house number + rare name or address word,
   rarest-word pairs, …) with frequency caps; per S1, the top 100 by a cheap overlap score, the
   top 20 by address score, and, for records with no address, each record's top 5 S1 entities
   by name overlap. Train: 98.7% of true pairs kept.
3. **Pair features** (`features_v3.py`, `features_v4.py`, `features_v7.py`): 52 name, address,
   number and context features, including the **twin-decoy** features (signed house-number
   offset against the S1 address; decoy-word log-odds learned from train pairs).
4. **Stage 1** (XGBoost, GPU) scores every blocked pair. **`candidate_pairs.tsv` = pairs with
   stage-1 probability ≥ 0.01** (≈4.4 per S1 entity on test; on train this cut loses no F0.5).
5. **Stage 2** (XGBoost, GPU) re-scores the candidates with evidence from competing S1 entities
   and from the record cluster. Matches are chosen per S1 by maximising expected F0.5, with each
   record given to at most one S1 entity. Matches are always a subset of the candidates.

## Requirements

- Python 3.14 (64-bit), `pip install -r requirements.txt`
- NVIDIA GPU with CUDA for XGBoost (developed on an RTX 5060 Ti 16 GB); 31 GB RAM is enough
  (the blocking step peaks at ~20 GB). Run one step at a time.
- About **110 GB of free disk** for intermediate files (candidate lists and pair features, written to `runs/`).

## Setup

The code finds the project root **3 levels above `src/`**, i.e. the folder that contains `code/` (the root of the
submission zip). Place the challenge's provided `student_resource/` folder there, unchanged:

```
<root>/
├── code/business_entity_resolution/src/ …   (this folder)
├── student_resource/dataset/{train,test}/    (provided data: source1/2/3 TSVs, train ground truth)
├── student_resource/utils/validate_submission.py   (provided validator, run automatically by the last step)
├── data_cache/    (created automatically: parquet copies of the TSVs)
├── runs/          (created automatically: all intermediate files and models)
└── output/        (written by the last step)
```

## Reproduce (run from `src/`, with `PYTHONIOENCODING=utf-8`)

| # | Command | Output | Time |
|---|---|---|---|
| 1 | `python run_translit.py` | `runs/translit/dictionary.parquet` | 1 min |
| 2 | `python pipeline_v6.py blocking --split train` | `runs/blocking_v6/` (used by step 3 only) | 15 min |
| 3 | `python pipeline_v7.py vocab` | `runs/v7/decoy_vocab.parquet` | 1 min |
| 4 | `python pipeline_v8.py blocking --split train` then `python pipeline_v8.py blocking --split test` | `runs/blocking_v8/` | 13 + 12 min |
| 5 | `python pipeline_v8.py features --split train` | `runs/v8/train_features/` | 15 min |
| 6 | `python pipeline_v8.py train --tag gpu_slot` | `runs/v8/gpu_slot/` (stage-1 models, out-of-fold probabilities, evidence features, `decision.json`) | 13 min |
| 7 | `python judge_v17.py train --tag v17` | final stage-2 judge (`runs/v17/v17/`): v9's all-entity training on the submitted region + rival-count prior (v11) + word-swap (v12b) + raw-name rivalry and source balance (v13) + word alignment (v15) + char-3-gram TF-IDF cosine (v16), trained with 4 nested folds (v17) | 30 min |
| 8 | `python pipeline_v11.py blocking --split test` | `runs/blocking_v11/` (France re-blocked on normalised addresses; US/India = step 4's) | 2 min |
| 9 | `python pipeline_v11.py features --split test --fr-map` | `runs/v11fr/test_features/` | 12 min |
| 10 | `mkdir -p ../../../runs/v11fr/gpu_slot && cp ../../../runs/v8/gpu_slot/stage*_fold*.ubj ../../../runs/v8/gpu_slot/decision.json ../../../runs/v11fr/gpu_slot/`, then `python pipeline_v11.py test --fr-map --tag gpu_slot --predict-only` | stage-1 test predictions + evidence features in `runs/v11fr/gpu_slot/` | 12 min |
| 11 | `python judge_v17.py test --tag v17 --v8-dir ../../../runs/v11fr --blocking-dir ../../../runs/blocking_v11 --out-dir ../../../runs/v17_default` | cached stage-2 test predictions `runs/v17/v17/test_p2_v11fr.parquet` (+ files with the uncalibrated default rule, not submitted) | 18 min |
| 12 | `python export_v17.py --p2 ../../../runs/v17/v17/test_p2_v11fr.parquet --blocking-dir ../../../runs/blocking_v11 --rule '{"default": {"kind": "expected_f05", "t_scale": 1.0, "max_rank": 1000, "floor": 0.01}, "by_country": {"France": {"kind": "expected_f05", "t_scale": 1.0, "max_rank": 1000, "floor": 0.01, "logit_shift": -1.0}}}' --out-dir ../../../output` | **`output/matching_results.tsv`, `output/candidate_pairs.tsv`** (stage-1 p >= 0.01) + official validator | 2 min |

Total ≈ 3–3.5 hours. (Step 10's copy uses Git Bash / Unix syntax; on PowerShell use `New-Item -ItemType Directory -Force` and `Copy-Item`.)

**France calibration (step 12).** France has no training labels. On test, US/India predictions behave exactly
like the held-out data, but France has 4x more records whose best probability is 0.05–0.5, and its
leaderboard-implied score was ~0.009 below what the model's own probabilities predict (beyond the US/India
gap): French probabilities are overconfident. France's probabilities are therefore shifted by −1.0 on the
logit scale before the expected-F0.5 decision (US/India unchanged). The shift was checked label-free on the
public leaderboard (v16: 0.981638 → 0.98198 with only France changed).

`--fr-map` translates French legal forms and business words (SARL→LLC, SAS→Corp, Groupe→Group,
Participations→Holdings, …) before the decoy-word lookup, for French records only. France has no
training labels, and the vocabulary learned on US/India otherwise misreads French legal forms.

Every command refuses to overwrite existing outputs. Seeds are fixed (fold hash seed 7,
test-like universe seed 11, validation samples seeded, XGBoost seed 42). Two sources of small
run-to-run differences remain: GPU floating point, and ties in the blocking score when the
top-100 per S1 is cut (ties are broken by row order; ~3% of pairs near the cut can swap, with no
measurable effect on F0.5).

`--fr-map` / `normalize_fr.py` (France only): French legal forms and business words are mapped to
their English equivalents for the decoy-word lookup, and French addresses are normalised —
the department -> region map (Nord/Pas-de-Calais -> Hauts-de-France, Gironde -> Nouvelle-Aquitaine,
Loire-Atlantique -> Pays de la Loire) is learned from the data through the cities each department
co-occurs with, and French street types (R./Av/Bd/All/Imp/Rte/Ch, "N°") are unified.

## Validation design

- **Held-out score**: 2-fold cross-validation by S1 entity on train (the final judge splits each fold in two
  again — 4 nested folds — so upstream features stay out-of-fold), in a **test-like universe**:
  19% of train S1 entities are removed so that 40% of records are decoys, as in test. The removed
  entities are never trained on or scored; the decoy-word vocabulary is learned only from them.
- Local metric = the official macro F0.5 (`scorer.py`, same parsing as the validator).

| Version | Held-out F0.5 | Leaderboard |
|---|---|---|
| v1 rules | 0.7150 | 0.7209 |
| v6 translit + two-stage judge | 0.97007 | 0.960435 |
| v8 twin-decoy features + blocking v8 | 0.98313 | 0.97556 |
| v8 + French word mapping | 0.98313 (France unlabelled) | 0.977408 |
| v9 judge (all train entities, submitted region only) | 0.98418 | — |
| v11: judge + rival-count prior, French address normalisation | 0.98425 | ≈0.978 |
| v12b: + word-swap features | 0.98434 | 0.979541 |
| v13: + raw-name rivalry, source balance | 0.98468 | — |
| v15: + word alignment (typo vs replaced word) | 0.98509 | — |
| v16: + char-3-gram TF-IDF cosine | 0.98551 | 0.981638 |
| v16 + France calibration (logit shift −1.0) | 0.98551 (France unlabelled) | 0.98198 |
| **v17: 4 nested folds + France calibration (final)** | **0.98594** | **0.982349** |

## Files (`src/`)

`scorer.py` metric · `baseline.py` I/O helpers · `normalize*.py` cleaning · `translit.py` / `run_translit.py`
dictionary · `segment.py` web-handle names · `blocking_v2/v3/v8.py` candidate generation ·
`features_v3/v4/v7.py` pair features · `judge_v3/v4/v5.py` shared judge code (decision rules,
one-owner, stage-2 evidence, GPU training) · `pipeline_v6/v7/v8.py` end-to-end commands ·
`export_submission.py` small candidate set + final files · `judge_v9.py` (larger stage 2) ·
`judge_v11/v12/v13/v15/v16.py` successive stage-2 feature sets (each imports the previous) · **`judge_v17.py`
final judge** · **`export_v17.py` final files with per-country calibration** · `normalize_fr.py` French addresses.
