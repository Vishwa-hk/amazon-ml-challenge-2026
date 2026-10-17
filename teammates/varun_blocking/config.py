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

# Legal Suffix Standardization Map (US, India, France, International)
LEGAL_SUFFIX_MAP = {
    "inc": "corporation",
    "incorporated": "corporation",
    "corp": "corporation",
    "co": "company",
    "cie": "company",
    "compagnie": "company",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "llp": "llp",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "eurl": "eurl",
    "sci": "sci",
    "snc": "snc",
    "gie": "gie",
    "scop": "coop",
    "coop": "coop",
    "ets": "etablissement",
    "etablissements": "etablissement",
    "soc": "societe",
    "societe": "societe",
    "assn": "association",
    "association": "association",
    "enterprises": "enterprise",
    "services": "service",
    "solutions": "solution",
    "technologies": "technology",
    "tech": "technology",
}

# Stopwords to filter for token blocking (English, French, generic)
STOPWORDS = {
    "company", "corporation", "limited", "private", "llc", "llp", "inc", "ltd",
    "pvt", "enterprise", "service", "solution", "technology", "group", "holdings",
    "international", "global", "india", "us", "usa", "france", "and", "the", "of",
    "in", "for", "at", "to", "a", "an", "&", "le", "la", "les", "de", "du", "des",
    "et", "au", "aux", "en", "sur", "pour", "par", "dans", "un", "une", "d", "l",
    "st", "ste", "saint", "sainte"
}

# Blocking Hyperparameters
BLOCKING_CONFIG = {
    "max_candidates_per_s1": 30,         # Maximum candidate ceiling budget
    "top_k": 30,
    "rare_token_freq": 200,              # Max frequency for rare token inverted index
    "rare_prefix_freq": 120,             # Max frequency for rare prefix inverted index
    "residual_min_cands": 4,             # Threshold candidate count to trigger residual fallback
    "char_ngram_range": (3, 5),          # Character n-gram range for residual shingle index
}

# Training Hyperparameters (Precision-Weighted LightGBM)
MODEL_CONFIG = {
    "random_state": 42,
    "num_hard_negatives_per_pos": 8,     # Empirically optimized hard negative ratio
    "num_random_negatives_per_pos": 1,   # Random negatives per positive match
    "val_entity_split": 0.20,            # 20% validation split by S1 entity
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.72,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
}
