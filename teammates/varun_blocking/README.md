# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

This repository contains the complete, runnable machine learning pipeline for the Business Entity Resolution Challenge.

## Project Structure

```
amazon_ml_entity_resolution/
├── main.py                           # CLI entry point to run the pipeline
├── requirements.txt                  # Dependency list
├── Documentation_template.md         # Final methodology writeup
├── output/                           # Contains matching_results.tsv and candidate_pairs.tsv
├── src/
│   └── business_entity_resolution/
│       ├── config.py                 # Hyperparameters and path constants
│       ├── io.py                     # TSV & Parquet I/O routines
│       ├── preprocess.py             # Multi-representation text normalizer
│       ├── blocking.py               # Multi-stage candidate generator
│       ├── features.py               # Pairwise feature extractor
│       ├── model.py                  # Hard negative miner & LightGBM model
│       ├── decision.py               # Entity decision & threshold optimizer
│       ├── evaluate.py               # Macro F0.5 & candidate recall metrics
│       └── pipeline.py               # End-to-end workflow runner
└── tests/                            # Unit tests for preprocessing, features, & outputs
```

## Setup & Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run unit tests:
   ```bash
   python -m pytest tests/
   ```

## Execution Instructions

### Full Pipeline Run (Train, Validate, & Infer Full Test Set)
```bash
python main.py
```

### Validate Output Files with Official Submission Validator
```bash
python main.py --validate-only
```

### Development Sample Run (Quick test on 2,000 S1 entities)
```bash
python main.py --sample-train 2000 --sample-test 1000
```
