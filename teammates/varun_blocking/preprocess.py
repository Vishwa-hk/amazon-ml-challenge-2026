import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from config import LEGAL_SUFFIX_MAP, STOPWORDS, CACHE_DIR

PUNCT_RE = re.compile(r"[^\w\s]")
MULTISPACE_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")
POSTAL_RE = re.compile(r"\b\d{5,6}\b")

SOUNDEX_MAP = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6'
}

def clean_text(text: str) -> str:
    """Basic text cleanup: Unicode NFKD normalization, lowercases, and strips non-alphanumeric chars."""
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
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
    """Extract 5 or 6 digit PIN / postal code from address string (covers US, India, France)."""
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

def extract_first_last_tokens(text: str) -> Tuple[str, str]:
    """Extract first and last token from a cleaned string."""
    if not text:
        return "", ""
    toks = text.split()
    if not toks:
        return "", ""
    return toks[0], toks[-1]

def extract_acronym(text: str) -> str:
    """Extract initials/acronym from multi-token business name."""
    if not text:
        return ""
    toks = [t for t in text.split() if t not in STOPWORDS and len(t) >= 2]
    if len(toks) >= 2:
        return "".join([t[0] for t in toks if t[0].isalnum()])
    return ""

def soundex_encode(word: str) -> str:
    """Fast, robust Soundex phonetic encoder."""
    if not word or not word.strip():
        return "0000"
    w = unicodedata.normalize('NFKD', str(word)).encode('ASCII', 'ignore').decode('utf-8').upper().strip()
    w = "".join([c for c in w if c.isalpha()])
    if not w:
        return "0000"
    
    first = w[0]
    codes = [first]
    for char in w[1:]:
        code = SOUNDEX_MAP.get(char, '0')
        if code != '0' and code != codes[-1]:
            codes.append(code)
            if len(codes) == 4:
                break
                
    while len(codes) < 4:
        codes.append('0')
    return "".join(codes)

def aggressive_normalize_name(name: str) -> str:
    """Aggressively normalize business name: strip legal terms, non-alphanumeric, space."""
    norm = normalize_business_name(name)
    info = extract_informative_tokens(norm)
    return PUNCT_RE.sub("", info).replace(" ", "")

def extract_phonetic_keys(name_info_tokens: str) -> Tuple[str, str, str]:
    """Extracts phonetic_first, phonetic_key1, phonetic_key2 for blocking."""
    if not name_info_tokens:
        return "", "", ""
    toks = [t for t in name_info_tokens.split() if len(t) >= 2]
    if not toks:
        return "", "", ""
    p_first = soundex_encode(toks[0])
    p_k1 = p_first
    p_k2 = soundex_encode(toks[1]) if len(toks) > 1 else p_k1
    return p_first, p_k1, p_k2

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-performance single-pass multi-representation preprocessor.
    Processes 100k rows in ~1 second via direct tuple mapping and Unicode normalization.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'entity_id', 'business_name', 'business_address', 'country',
            'name_clean', 'name_norm', 'name_info_tokens', 'name_agg_norm',
            'name_prefix_3', 'acronym', 'first_token', 'last_token',
            'phonetic_first', 'phonetic_key1', 'phonetic_key2',
            'address_clean', 'postal_code', 'addr_numbers', 'house_number'
        ])
        
    names = df['business_name'].fillna('').astype(str).values
    addrs = df['business_address'].fillna('').astype(str).values
    countries = df['country'].fillna('').astype(str).str.upper().values
    
    n_rows = len(df)
    
    name_cleans = [None] * n_rows
    name_norms = [None] * n_rows
    name_info_tokens = [None] * n_rows
    name_agg_norms = [None] * n_rows
    name_prefix_3s = [None] * n_rows
    acronyms = [None] * n_rows
    first_tokens = [None] * n_rows
    last_tokens = [None] * n_rows
    phonetic_firsts = [None] * n_rows
    phonetic_key1s = [None] * n_rows
    phonetic_key2s = [None] * n_rows
    address_cleans = [None] * n_rows
    postal_codes = [None] * n_rows
    addr_numbers = [None] * n_rows
    house_numbers = [None] * n_rows
    
    legal_map = LEGAL_SUFFIX_MAP
    stops = STOPWORDS
    soundex_map = SOUNDEX_MAP
    punct_sub = PUNCT_RE.sub
    multi_sub = MULTISPACE_RE.sub
    digits_find = DIGITS_RE.findall
    postal_search = POSTAL_RE.search
    unf = unicodedata.normalize
    
    for i in range(n_rows):
        # 1. Business Name Normalization
        name_str = names[i]
        if name_str:
            name_ascii = unf('NFKD', name_str).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
            name_c = multi_sub(' ', punct_sub(' ', name_ascii)).strip()
        else:
            name_c = ""
            
        name_cleans[i] = name_c
        
        if name_c:
            toks = name_c.split()
            norm_toks = [legal_map.get(t, t) for t in toks]
            n_norm = " ".join(norm_toks)
            
            info_toks = [t for t in norm_toks if t not in stops and len(t) >= 2]
            n_info = " ".join(info_toks)
            n_agg = "".join(info_toks)
            pref3 = n_norm[:3] if len(n_norm) >= 3 else n_norm
            
            acr = "".join([t[0] for t in info_toks if t[0].isalnum()]) if len(info_toks) >= 2 else ""
            f_tok = info_toks[0] if info_toks else ""
            l_tok = info_toks[-1] if info_toks else ""
            
            # Soundex computation on first token
            if f_tok:
                w_chars = [c for c in f_tok.upper() if c.isalpha()]
                if w_chars:
                    first_c = w_chars[0]
                    scodes = [first_c]
                    for char in w_chars[1:]:
                        code = soundex_map.get(char, '0')
                        if code != '0' and code != scodes[-1]:
                            scodes.append(code)
                            if len(scodes) == 4:
                                break
                    while len(scodes) < 4:
                        scodes.append('0')
                    p_first = "".join(scodes)
                else:
                    p_first = "0000"
            else:
                p_first = ""
                
            p_k1 = p_first
            p_k2 = p_first
            if len(info_toks) > 1:
                w_chars2 = [c for c in info_toks[1].upper() if c.isalpha()]
                if w_chars2:
                    first_c2 = w_chars2[0]
                    scodes2 = [first_c2]
                    for char in w_chars2[1:]:
                        code2 = soundex_map.get(char, '0')
                        if code2 != '0' and code2 != scodes2[-1]:
                            scodes2.append(code2)
                            if len(scodes2) == 4:
                                break
                    while len(scodes2) < 4:
                        scodes2.append('0')
                    p_k2 = "".join(scodes2)
        else:
            n_norm = ""
            n_info = ""
            n_agg = ""
            pref3 = ""
            acr = ""
            f_tok = ""
            l_tok = ""
            p_first = ""
            p_k1 = ""
            p_k2 = ""
            
        name_norms[i] = n_norm
        name_info_tokens[i] = n_info
        name_agg_norms[i] = n_agg
        name_prefix_3s[i] = pref3
        acronyms[i] = acr
        first_tokens[i] = f_tok
        last_tokens[i] = l_tok
        phonetic_firsts[i] = p_first
        phonetic_key1s[i] = p_k1
        phonetic_key2s[i] = p_k2
        
        # 2. Address Normalization
        addr_str = addrs[i]
        if addr_str:
            addr_ascii = unf('NFKD', addr_str).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
            addr_c = multi_sub(' ', punct_sub(' ', addr_ascii)).strip()
            pm = postal_search(addr_ascii)
            post = pm.group(0) if pm else ""
            nums = digits_find(addr_c)
            num_str = " ".join(nums)
            house = nums[0] if nums else ""
        else:
            addr_c = ""
            post = ""
            num_str = ""
            house = ""
            
        address_cleans[i] = addr_c
        postal_codes[i] = post
        addr_numbers[i] = num_str
        house_numbers[i] = house
        
    out_df = pd.DataFrame({
        'entity_id': df['entity_id'].values,
        'business_name': names,
        'business_address': addrs,
        'country': countries,
        'name_clean': name_cleans,
        'name_norm': name_norms,
        'name_info_tokens': name_info_tokens,
        'name_agg_norm': name_agg_norms,
        'name_prefix_3': name_prefix_3s,
        'acronym': acronyms,
        'first_token': first_tokens,
        'last_token': last_tokens,
        'phonetic_first': phonetic_firsts,
        'phonetic_key1': phonetic_key1s,
        'phonetic_key2': phonetic_key2s,
        'address_clean': address_cleans,
        'postal_code': postal_codes,
        'addr_numbers': addr_numbers,
        'house_number': house_numbers
    })
    return out_df
