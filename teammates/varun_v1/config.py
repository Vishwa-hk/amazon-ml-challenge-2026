import os
from pathlib import Path

# Relative Project Paths
PROJECT_ROOT = Path(__file__).parent.resolve()

# Look for dataset in local project data/ folder or student_resource folder
DATASET_BASE = PROJECT_ROOT / "data"
if not DATASET_BASE.exists() or not (DATASET_BASE / "train").exists():
    DATASET_BASE = PROJECT_ROOT.parent / "student_resource" / "student_resource" / "dataset"
if not DATASET_BASE.exists():
    DATASET_BASE = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\dataset")

TRAIN_DIR = DATASET_BASE / "train"
TEST_DIR = DATASET_BASE / "test"

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Dataset Files
TRAIN_S1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1_PATH = TEST_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DIR / "test_source3.tsv"

# Legal Suffix Standardization Map
LEGAL_SUFFIX_MAP = {
    "inc": "corporation",
    "incorporated": "corporation",
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "pvt": "private",
    "llc": "llc",
    "llp": "llp",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
    "enterprises": "enterprise",
    "services": "service",
    "solutions": "solution",
    "technologies": "technology",
    "tech": "technology",
}

# Stopwords to filter for token blocking
STOPWORDS = {
    "company", "corporation", "limited", "private", "llc", "llp", "inc", "ltd",
    "pvt", "enterprise", "service", "solution", "technology", "group", "holdings",
    "international", "global", "india", "us", "usa", "france", "and", "the", "of",
    "in", "for", "at", "to", "a", "an", "&"
}

# Blocking Hyperparameters
BLOCKING_CONFIG = {
    "max_candidates_per_s1": 15,         # Top-K candidate cap per S1 entity (optimized for Reduction Ratio tie-breaker)
    "min_token_len": 3,                  # Min length for token indexing
    "tfidf_max_features": 100000,        # TF-IDF vocabulary size limit
    "char_ngram_range": (3, 5),          # Character n-gram range
}

# Training Hyperparameters
MODEL_CONFIG = {
    "random_state": 42,
    "num_hard_negatives_per_pos": 5,     # Hard negatives per positive match
    "num_random_negatives_per_pos": 2,   # Random negatives per positive match
    "val_entity_split": 0.20,            # 20% validation split by S1 entity
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
}
