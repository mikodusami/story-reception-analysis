"""Cleaning helpers shared by data-cleaning.ipynb and the tests.

Every function here works on plain values or pandas Series and has no side effects.
"""
import re

import numpy as np
import pandas as pd

# ── Review text ──────────────────────────────────────────────────────────────

REPLACEMENTS = {
    # mojibake (UTF-8 read as Latin-1); fix these before single characters
    'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€\x9d': '"', 'â€"': '-', 'â€¦': '...', 'Â': '',
    # typographic punctuation
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': ' - ', '…': '...', '\xa0': ' ',
}


def clean_review_text(s: pd.Series) -> pd.Series:
    """Strip HTML and URLs, fix mojibake, transliterate accents, drop other non-ASCII, collapse spaces."""
    s = s.fillna('').astype(str)
    s = s.str.replace(r'<[^>]+>', ' ', regex=True)                 # HTML tags
    s = s.str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)   # URLs
    for bad, good in REPLACEMENTS.items():
        s = s.str.replace(bad, good, regex=False)
    s = s.str.normalize('NFKD')                                    # é -> e + combining accent
    s = s.str.replace(r'[̀-ͯ]', '', regex=True)          # drop the combining accents
    s = s.str.replace(r'[^\x00-\x7F]+', ' ', regex=True)           # emoji, symbols, other scripts
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()


def normalize_title(s: pd.Series) -> pd.Series:
    """Lowercase, collapse whitespace, drop a leading 'the', 'a' or 'an'."""
    s = s.fillna('').str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    return s.str.replace(r'^(the|a|an)\s+', '', regex=True)


def parse_genres(s: pd.Series) -> pd.Series:
    """'Fantasy|Fiction|Epic Fantasy' -> ['Fantasy', 'Fiction', 'Epic Fantasy']; missing -> []."""
    return s.fillna('').map(lambda g: [x.strip() for x in g.split('|') if x.strip()])


def runtime_to_minutes(s: pd.Series) -> pd.Series:
    """'1h 25m' -> 85, '45m' -> 45, '2h' -> 120; anything else -> NaN."""
    hours = s.str.extract(r'(\d+)\s*h', expand=False).astype(float)
    mins = s.str.extract(r'(\d+)\s*m', expand=False).astype(float)
    total = hours.fillna(0) * 60 + mins.fillna(0)
    return total.where(hours.notna() | mins.notna())


# ── Critic ratings ───────────────────────────────────────────────────────────

QUALITATIVE = {'must watch': 5.0, 'highly recommended': 4.5, 'recommended': 4.0,
               'catch it on cable': 2.5, 'half price': 2.5}
LETTER_GRADES = {'A+': 5.0, 'A': 4.7, 'A-': 4.3, 'B+': 4.0, 'B': 3.7, 'B-': 3.3,
                 'C+': 3.0, 'C': 2.7, 'C-': 2.3, 'D+': 2.0, 'D': 1.7, 'D-': 1.3,
                 'F+': 1.0, 'F': 0.5, 'F-': 0.0}
WORD_NUMBERS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'half': 0.5}


def _scale(num, denom):
    return round(num / denom * 5, 2) if denom > 0 else np.nan


def _parse_critic_rating(raw):
    raw = str(raw).strip()
    if raw == '' or raw.lower() == 'nan':
        return np.nan
    low = raw.lower()

    if low in QUALITATIVE:                                            # 'Recommended'
        return QUALITATIVE[low]

    if m := re.match(r'^([\d.]+)\s*ripples?$', raw, re.I):            # '4 Ripples'
        return float(m.group(1))

    grade = raw.replace('=', '').strip()                              # 'C=' typo
    grade = re.sub(r'[\s-]*minus$', '-', grade, flags=re.I)           # 'A-minus', 'A minus'
    grade = re.sub(r'[\s-]*plus$', '+', grade, flags=re.I)            # 'B-plus'
    grade = re.sub(r'\s+', '', grade).upper()                         # 'C +'
    if grade in LETTER_GRADES:
        return LETTER_GRADES[grade]

    if (m := re.match(r'^(\w+)\s+stars?$', raw, re.I)) and m.group(1).lower() in WORD_NUMBERS:
        return float(WORD_NUMBERS[m.group(1).lower()])               # 'FIVE STARS'

    if m := re.match(r'^([\d.]+)\s+stars?$', raw, re.I):              # '3 stars', '.5 stars'
        return float(m.group(1))

    if m := re.match(r'^(\d+)\s+(\d+)/(\d+)\s+stars?$', raw, re.I):   # '3 1/2 stars'
        return round(int(m.group(1)) + int(m.group(2)) / int(m.group(3)), 2)

    if m := re.match(r'^(\d+)\s+(\d+)/(\d+)/(\d+)$', raw):            # '3 1/2/5'
        return _scale(int(m.group(1)) + int(m.group(2)) / int(m.group(3)), int(m.group(4)))

    if m := re.match(r'^([\d.]+)\s+(?:out\s+)?of\s+([\d.]+)$', raw, re.I):  # '2 out of 5', '4.5 of 5'
        return _scale(float(m.group(1)), float(m.group(2)))

    if m := re.match(r'^([\d.]+)\s*%$', raw):                         # '75%'
        return round(float(m.group(1)) / 20, 2)

    norm = re.sub(r'(\d+\.)\s+(\d+)', r'\1\2', raw)                   # '2. 5 / 5' -> '2.5 / 5'
    if m := re.match(r'^([\d.]+)\s*/\s*([\d.]+)$', norm):             # '3.5/5', '7/10', '18/20'
        return _scale(float(m.group(1)), float(m.group(2)))

    try:                                                              # plain number: guess the scale
        val = float(raw)
    except ValueError:
        return np.nan
    if val <= 5:
        return round(val, 2)
    if val <= 10:
        return round(val / 2, 2)
    if val <= 100:
        return round(val / 20, 2)
    return np.nan


def parse_critic_rating(raw):
    """Turn one Rotten Tomatoes critic rating ('3/5', 'B+', '3 1/2 stars', '7/10', ...) into 0-5.

    Returns NaN if the value is unreadable or off the scale (e.g. '7.5/5', '29/10').
    """
    try:
        val = _parse_critic_rating(raw)
    except (ValueError, ZeroDivisionError):
        return np.nan
    return val if 0 <= val <= 5 else np.nan
