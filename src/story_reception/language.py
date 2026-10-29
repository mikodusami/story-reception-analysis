"""Review language detection with Lingua, limited to the languages that appear in this data."""
from functools import lru_cache

import pandas as pd

LANGUAGE_NAMES = [
    'ENGLISH', 'SPANISH', 'PORTUGUESE', 'FRENCH', 'GERMAN', 'ITALIAN', 'DUTCH', 'POLISH',
    'RUSSIAN', 'BULGARIAN', 'UKRAINIAN', 'TURKISH', 'ARABIC', 'PERSIAN', 'INDONESIAN',
    'TAGALOG', 'VIETNAMESE', 'CHINESE', 'JAPANESE', 'KOREAN', 'GREEK', 'CROATIAN', 'CZECH',
    'ROMANIAN', 'HUNGARIAN', 'SWEDISH', 'HEBREW', 'HINDI', 'BENGALI', 'THAI',
]


@lru_cache(maxsize=1)
def _detector():
    # Built once: loading the language models takes a few seconds
    from lingua import Language, LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_languages(*(getattr(Language, n) for n in LANGUAGE_NAMES)).build()


def detect_language(s: pd.Series) -> pd.Series:
    """ISO 639-1 code per review ('en', 'es', ...), or 'unknown'.

    Runs on the first 400 characters of the raw text with HTML and URLs removed, before any
    accent or non-ASCII stripping, so non-Latin scripts are still visible to the detector.
    """
    sample = (s.fillna('').astype(str)
               .str.replace(r'<[^>]+>|https?://\S+|www\.\S+', ' ', regex=True)
               .str.slice(0, 400))
    found = _detector().detect_languages_in_parallel_of(sample.tolist())
    return pd.Series([lang.iso_code_639_1.name.lower() if lang else 'unknown' for lang in found],
                     index=s.index)
