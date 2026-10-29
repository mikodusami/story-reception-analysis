"""Tests for the Streamlit app and its prediction helpers. They use the committed artifacts in app/artifacts/."""
import math
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / 'app'
pytestmark = pytest.mark.skipif(not (APP_DIR / 'artifacts' / 'metrics.json').exists(), reason='app artifacts not built')
pytest.importorskip('joblib')
sys.path.insert(0, str(APP_DIR))

from predict import (  # noqa: E402
    book_genre_divisiveness,
    film_features,
    highlight_review,
    load_artifacts,
    predict_film,
    predict_review,
)


@pytest.fixture(scope='module')
def arts():
    return load_artifacts()


@pytest.mark.parametrize('platform', ['Goodreads', 'RT audience', 'RT critics'])
def test_review_predictions_rank_and_stay_on_scale(arts, platform):
    rave = predict_review('An absolute masterpiece, brilliant and beautiful. Loved every page. Highly recommend.',
                          platform, arts)
    pan = predict_review('Boring, tedious and poorly written. The worst. A complete waste of time.', platform, arts)
    assert rave['stars'] > pan['stars']
    for r in (rave, pan):
        lo, hi = r['scale']
        assert lo <= r['stars'] <= hi and 0 <= r['normalized'] <= 1


def test_empty_review_gets_the_average(arts):
    r = predict_review('', 'Goodreads', arts)
    assert r['words_used'] == 0 and len(r['contributions']) == 0 and 0 <= r['normalized'] <= 1


def test_highlight_escapes_html(arts):
    text = '<script>alert(1)</script> masterpiece & boring'
    out = highlight_review(text, predict_review(text, 'Goodreads', arts)['contributions'])
    assert '<script>' not in out and '&lt;script&gt;' in out and '&amp;' in out
    assert 'rgba(42,120,214' in out and 'rgba(227,73,72' in out       # masterpiece up, boring down


def test_film_features_match_training_columns(arts):
    X = film_features({'genres': ['Horror', 'Not A Real Genre'], 'content_rating': 'Unrated', 'runtime_min': 80,
                       'release_year': 2024, 'release_month': 10, 'original_language': 'English'}, arts)
    assert list(X.columns) == arts['films']['columns'] and len(X) == 1
    assert X['genre: Horror'].iloc[0] == 1 and X['rated: Unrated'].iloc[0] == 1


def test_film_predictions_follow_the_analysis(arts):
    # Short unrated horror leans toward critics; a long PG-13 action-romance leans toward audiences
    critic_film = predict_film({'genres': ['Horror'], 'content_rating': 'Unrated', 'runtime_min': 80,
                                'release_year': 2023, 'release_month': 10, 'original_language': 'English'}, arts)
    audience_film = predict_film({'genres': ['Action', 'Romance'], 'content_rating': 'PG-13', 'runtime_min': 140,
                                  'release_year': 2023, 'release_month': 7, 'original_language': 'English'}, arts)
    assert critic_film['gap'] > audience_film['gap']
    for p in (critic_film, audience_film):
        assert math.isfinite(p['gap']) and 0 <= p['split probability'] <= 1


def test_missing_film_fields_still_predict(arts):
    p = predict_film({'genres': [], 'content_rating': None, 'runtime_min': None, 'release_year': None,
                      'release_month': None, 'original_language': None}, arts)
    assert math.isfinite(p['gap'])


def test_book_genre_divisiveness(arts):
    tags = book_genre_divisiveness('Romance, Dark Romance, Not A Tag, Mystery', arts)
    assert 'Not A Tag' not in tags.index and tags.is_monotonic_decreasing


def test_app_runs_end_to_end():
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    at = AppTest.from_file(str(APP_DIR / 'app.py'), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].label == 'Predicted rating'
    at.text_area[0].set_value('Boring and poorly written. A waste of time.').run()
    assert not at.exception and float(at.metric[0].value.split()[0]) < 3.5
