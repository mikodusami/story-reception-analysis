"""Prediction helpers for the Streamlit app.

Plain functions over the artifacts written by app/train.py, kept separate from the UI so they can be unit-tested.
"""
import html
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Find the story_reception package when it isn't pip-installed (e.g. on Hugging Face Spaces)
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / 'src', _ROOT):
    if (_p / 'story_reception').exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from story_reception.clean import clean_review_text  # noqa: E402
from story_reception.features import pre_release_features  # noqa: E402

ART = Path(__file__).resolve().parent / 'artifacts'
POS, NEG = (42, 120, 214), (227, 73, 72)          # blue = pushes the rating up, red = down


@lru_cache(maxsize=1)
def load_artifacts(path: str = str(ART)) -> dict:
    p = Path(path)
    reviews = joblib.load(p / 'review_models.joblib')
    for m in reviews.values():
        m['names'] = m['vectorizer'].get_feature_names_out()
    return {
        'reviews': reviews,
        'films': joblib.load(p / 'film_models.joblib'),
        'books': joblib.load(p / 'book_model.joblib'),
        'films_sample': pd.read_csv(p / 'films_sample.csv'),
        'books_sample': pd.read_csv(p / 'books_sample.csv'),
        'books_points': pd.read_csv(p / 'books_points.csv'),
        'genre_divisiveness': json.loads((p / 'genre_divisiveness.json').read_text()),
        'metrics': json.loads((p / 'metrics.json').read_text()),
    }


# ── Reviews ─────────────────────────────────────────────────────────────────

def predict_review(text: str, platform: str, arts: dict) -> dict:
    """Predicted rating for a review, in stars and on 0-1, with each word's contribution to it."""
    m = arts['reviews'][platform]
    cleaned = clean_review_text(pd.Series([text or ''])).iloc[0]
    X = m['vectorizer'].transform([cleaned])
    normalized = float(np.clip(m['ridge'].predict(X)[0], 0, 1))
    lo, hi = m['scale']
    contributions = pd.Series(X.data * m['ridge'].coef_[X.indices], index=m['names'][X.indices]).sort_values()
    return {'normalized': normalized, 'stars': lo + normalized * (hi - lo), 'scale': (lo, hi),
            'contributions': contributions, 'words_used': int(X.nnz)}


def highlight_review(text: str, contributions: pd.Series) -> str:
    """HTML with single words shaded by their contribution (blue up, red down). Two-word phrases aren't shaded."""
    words = {w: c for w, c in contributions.items() if ' ' not in w}
    top = max((abs(c) for c in words.values()), default=0) or 1
    out, last = [], 0
    for match in re.finditer(r"[A-Za-z][A-Za-z']*", text):
        word = match.group().lower()
        out.append(html.escape(text[last:match.start()]))
        c = words.get(word)
        if c is not None and abs(c) >= 0.05 * top:
            r, g, b = POS if c > 0 else NEG
            alpha = 0.15 + 0.6 * abs(c) / top
            out.append(f'<span style="background: rgba({r},{g},{b},{alpha:.2f}); border-radius: 3px; '
                       f'padding: 0 2px" title="{c:+.3f}">{html.escape(match.group())}</span>')
        else:
            out.append(html.escape(match.group()))
        last = match.end()
    out.append(html.escape(text[last:]))
    return ''.join(out).replace('\n', '<br>')


# ── Films ───────────────────────────────────────────────────────────────────

def film_features(film: dict, arts: dict) -> pd.DataFrame:
    """One row of model features for a film described by genres, rating, runtime, year, month and language."""
    m = arts['films']
    month = int(film.get('release_month') or 0)
    year = film.get('release_year')
    row = {
        'genres': '|'.join(film.get('genres') or []),
        'content_rating': None if film.get('content_rating') in (None, 'Unrated') else film['content_rating'],
        'runtime_min': film.get('runtime_min'),
        'release_year': year,
        'release_date': f'{int(year)}-{month:02d}-15' if year and month else None,
        'original_language': film.get('original_language') or None,
    }
    X = pre_release_features(pd.DataFrame([row]), genre_vocab=m['genre_vocab'], language_vocab=m['language_vocab'])
    return X[m['columns']]


def predict_film(film: dict, arts: dict) -> dict:
    X = film_features(film, arts)
    m = arts['films']
    return {'gap': float(m['gap'].predict(X)[0]), 'split probability': float(m['split'].predict_proba(X)[0, 1])}


def describe_gap(gap: float) -> str:
    if abs(gap) < 5:
        return 'critics and audiences roughly agree'
    side = 'critics' if gap > 0 else 'audiences'
    return f'{side} rate it {abs(gap):.0f} points higher'


# ── Books ───────────────────────────────────────────────────────────────────

def book_genre_divisiveness(genres: str, arts: dict) -> pd.Series:
    """For a book's tags, each tag's average adjusted divisiveness (tags with 30+ books only), most divisive first."""
    table = arts['genre_divisiveness']
    tags = [g.strip() for g in str(genres or '').split(',') if g.strip() in table]
    return pd.Series({g: table[g] for g in tags}, dtype=float).sort_values(ascending=False)
