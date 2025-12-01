# utils/names.py
import unicodedata
import re

def normalize_name(name: str) -> str:
    if not name:
        return ""
    # Normalize to NFKD and remove diacritics (e.g., ë -> e)
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Casefold for robust lowercase across Unicode
    s = s.casefold()
    # Normalize whitespace (collapse multiple spaces, trim ends)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def log_normalization(label: str, raw: str):
    # Show raw string, casefold-only, and accent-stripped variant
    cf = raw.casefold() if raw else ""
    nfkd = unicodedata.normalize("NFKD", raw) if raw else ""
    deacc = "".join(c for c in nfkd if not unicodedata.combining(c)) if raw else ""
    print(f"[{label}] raw={raw!r} casefold={cf!r} deaccent={deacc.casefold()!r}")
