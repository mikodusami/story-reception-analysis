"""Check the package reproduces data/cleaned/ exactly. Skipped when the (git-ignored) data isn't present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from story_reception.clean import normalize_title, parse_critic_rating

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
