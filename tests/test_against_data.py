"""Check the package reproduces data/cleaned/ exactly. Skipped when the (git-ignored) data isn't present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from story_reception.clean import normalize_title, parse_critic_rating
from story_reception.features import star_stats

CLEAN_DIR = Path(__file__).resolve().parents[1] / 'data' / 'cleaned'
pytestmark = pytest.mark.skipif(not CLEAN_DIR.exists(), reason='data/cleaned not present')


@pytest.fixture(scope='module')
def critics():
    return pd.read_csv(CLEAN_DIR / 'cleaned_rotten_tomatoes_critic_reviews.csv',
                       usecols=['movie_title', 'rating', 'rating_parsed', 'title_join'])


def test_critic_ratings_match_cleaned_data(critics):
    reparsed = critics['rating'].map(parse_critic_rating).to_numpy(dtype=float)
    assert np.allclose(reparsed, critics['rating_parsed'].to_numpy(dtype=float), equal_nan=True)


def test_title_keys_match_cleaned_data(critics):
    assert (normalize_title(critics['movie_title']) == critics['title_join'].fillna('')).all()


DB_PATH = CLEAN_DIR.parent / 'reception.duckdb'


@pytest.mark.skipif(not DB_PATH.exists(), reason='data/reception.duckdb not built')
def test_warehouse_divisiveness_matches_features():
    duckdb = pytest.importorskip('duckdb')
    books = pd.read_csv(CLEAN_DIR / 'cleaned_goodreads_books.csv')
    py = star_stats(books).assign(book_id=books['book_id'])
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        sql = con.execute('SELECT book_id, star_sd, share_1_2_stars, share_5_stars FROM books').df()
    merged = py.merge(sql, on='book_id')
    assert len(merged) == len(books)
    for a, b in [('star sd', 'star_sd'), ('share 1-2 stars', 'share_1_2_stars'), ('share 5 stars', 'share_5_stars')]:
        assert np.allclose(merged[a], merged[b], equal_nan=True), a
