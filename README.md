# Amazon ML Challenge 2026 — Business Entity Resolution

**Team Elite · Final leaderboard: 0.982349 macro F0.5 · Ranked 885 / 80,000+ registered students**

A two-stage entity-resolution system built in a 72-hour hackathon (25–27 Sep 2026). The task:
match each business in a reference list to its noisy duplicate records across two other sources —
~10 million records spanning the **US, India and France**, with deliberate look-alike decoys and
~40% of records belonging to no reference entity.

No external data, APIs or pretrained models are used. Everything — the transliteration dictionary,
the decoy-word vocabulary, and the models — is learned from the provided training split only.

## Approach

```
raw TSVs → normalisation → blocking (inverted-index keys) → pair features → stage-1 scorer → stage-2 judge → matches
```

1. **Normalisation** — case/accents/look-alike characters, legal forms, honorifics, street
   abbreviations; **Indic-script names (Devanagari, Bengali, Tamil, Telugu, Gujarati, Kannada…)
   transliterated to English** via a dictionary learned from aligned train pairs; glued web-handle
   names (`tristatekeystonehawk.com`) split back into words.
2. **Blocking** — 11 inverted-index key families within country, frequency-capped; keeps **98.7%
   of true pairs** while cutting the candidate space to a few per reference entity.
3. **Pair features** — 52 name/address/number/context features, including signed house-number
   offsets and **decoy-word log-odds** that catch look-alike traps.
4. **Stage-1 XGBoost** scores every blocked pair; candidates = probability ≥ 0.01.
5. **Stage-2 XGBoost judge** re-scores candidates with evidence from competing entities and the
   record cluster, choosing matches per entity by **maximising expected F0.5** (one owner per record).
6. **France calibration** — France has no training labels and overconfident probabilities, so its
   scores are shifted −1.0 on the logit scale before the decision (verified label-free on the public
   board). US/India unchanged.

Full method, validation design, and the version-by-version score progression (v1 rules 0.72 →
v17 final 0.982) are in **[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md)**
and the write-up under [`docs/`](docs/).

## Results

| Version | Held-out F0.5 | Leaderboard |
|---|---|---|
| v1 rules | 0.7150 | 0.7209 |
| v6 translit + two-stage judge | 0.97007 | 0.960435 |
| v8 twin-decoy features | 0.98313 | 0.97556 |
| v16 + char-3-gram TF-IDF | 0.98551 | 0.981638 |
| **v17: 4 nested folds + France calibration (final)** | **0.98594** | **0.982349** |

## Repository layout

| Path | What |
|---|---|
| `code/business_entity_resolution/` | The final pipeline (normalisation, blocking, features, two-stage judge, export). Start with its README. |
| `scripts/` | EDA and error-analysis scripts used during the hackathon. |
| `teammates/varun_*/` | Teammate's independent blocking/pipeline experiments (included with permission). |
| `baseline_pipeline.py` | The problem-agnostic Day-1 scaffold (get a valid submission on the board fast). |
| `docs/` | The team's final documentation (write-up + PDF). |

> **Note:** the challenge dataset is **not** included — it is Amazon's and cannot be redistributed.
> To reproduce, place the provided `student_resource/` folder at the repo root as described in the
> pipeline README.

## Team

**Team Elite** (Amazon ML Challenge 2026):
- **Vishwa Harikumar** — the final submitted pipeline and modelling (normalisation & Indic
  transliteration, blocking, pair features, two-stage XGBoost judge, France calibration), EDA and
  error analysis, submission and coordination.
- **Varun** — independent blocking and pipeline experiments (in `teammates/`, evaluated but not
  merged into the final submission).
- **Vikhyaath Mallikarjun** — team lead.
