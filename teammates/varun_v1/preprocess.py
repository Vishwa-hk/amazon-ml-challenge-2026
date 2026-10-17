import re
import pandas as pd
from typing import List, Tuple
from config import LEGAL_SUFFIX_MAP, STOPWORDS

PUNCT_RE = re.compile(r"[^\w\s]")
MULTISPACE_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")
POSTAL_RE = re.compile(r"\b\d{5,6}\b")

def clean_text(text: str) -> str:
    """Basic text cleanup: lowercases and strips non-alphanumeric chars."""
    if not text or pd.isna(text):
        return ""
    text = text.lower().strip()
    text = PUNCT_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()

def normalize_business_name(name: str) -> str:
    """Normalize legal suffixes and business acronyms without losing key tokens."""
    cleaned = clean_text(name)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    norm_tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(norm_tokens)

def extract_informative_tokens(cleaned_name: str) -> str:
    """Extract informative non-stopword tokens from cleaned business name."""
    if not cleaned_name:
        return ""
    tokens = cleaned_name.split()
    info_tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    return " ".join(info_tokens)

def extract_postal_code(address: str) -> str:
    """Extract 5 or 6 digit PIN / postal code from address string."""
    if not address or pd.isna(address):
        return ""
    match = POSTAL_RE.search(str(address))
    return match.group(0) if match else ""

def extract_numbers(text: str) -> str:
    """Extract all numeric tokens (house/street numbers, unit numbers)."""
    if not text or pd.isna(text):
        return ""
    numbers = DIGITS_RE.findall(str(text))
    return " ".join(numbers)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multi-representation preprocessing to a DataFrame containing
    ['entity_id', 'business_name', 'business_address', 'country'].
    Returns enhanced DataFrame with normalized features.
    """
    out_df = df.copy()
    
    # Fill NAs
    out_df['business_name'] = out_df['business_name'].fillna('').astype(str)
    out_df['business_address'] = out_df['business_address'].fillna('').astype(str)
    out_df['country'] = out_df['country'].fillna('').astype(str).str.upper()
    
    # 1. Cleaned & Normalized Names
    out_df['name_clean'] = out_df['business_name'].apply(clean_text)
    out_df['name_norm'] = out_df['business_name'].apply(normalize_business_name)
    out_df['name_info_tokens'] = out_df['name_norm'].apply(extract_informative_tokens)
    out_df['name_prefix_3'] = out_df['name_norm'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    
    # 2. Address & Postal Code Preprocessing
    out_df['address_clean'] = out_df['business_address'].apply(clean_text)
    out_df['postal_code'] = out_df['business_address'].apply(extract_postal_code)
    out_df['addr_numbers'] = out_df['address_clean'].apply(extract_numbers)
    
    return out_df
