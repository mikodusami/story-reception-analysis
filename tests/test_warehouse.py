"""Build the DuckDB warehouse from tiny hand-made CSVs and check the models and quality checks."""
import math

import pandas as pd
import pytest

duckdb = pytest.importorskip('duckdb')

from story_reception.warehouse import build  # noqa: E402

MOVIES = pd.DataFrame({
    'movie_url': ['m/a', 'm/b', 'm/c'], 'title': ['A', 'B', 'C'], 'release_year': [2020, 2021, 2022],
    'runtime_min': [90, 120, 100], 'content_rating': ['PG', 'R', 'PG-13'], 'original_language': ['English'] * 3,
    'tomatometer': [90, 40, 70], 'tomatometer_count': [100, 50, 80], 'critic_avg_rating': [8.0, 5.0, 6.5],
    'top_critic_score': [88, 35, None], 'audience_score': [60, 70, None], 'audience_avg_rating': [3.5, 4.0, None],
    'audience_review_count': [500, 300, 0], 'genres': ['Horror|Drama', 'Comedy', None],
})
CRITICS = pd.DataFrame({
    'movie_url': ['m/a', 'm/b'], 'critic_name': ['X', 'Y'], 'publication': ['P', 'Q'], 'is_positive': [1, 0],
    'rating_parsed': [4.5, None], 'rating_normalized': [0.9, None], 'language': ['en', 'en'],
    'word_count': [10, 12], 'review_text': ['great', 'dull'],
})
AUDIENCE = pd.DataFrame({
    'movie_url': ['m/a'], 'reviewer_id': ['h1'], 'score': [3.5], 'rating_normalized': [2 / 3],
    'verified': [True], 'language': ['en'], 'word_count': [8], 'review_text': ['fine'],
})
BOOKS = pd.DataFrame({
    'book_id': [1, 2], 'title': ['Even', 'Split'], 'author': ['a', 'b'], 'publication_year': [2000, 2010],
    'avg_rating': [3.0, 3.0], 'ratings_count': [5, 4],
    'rating_dist_1': [1, 2], 'rating_dist_2': [1, 0], 'rating_dist_3': [1, 0], 'rating_dist_4': [1, 0],
    'rating_dist_5': [1, 2], 'genres': ['Fantasy|Fiction', 'Romance'],
})
BOOK_REVIEWS = pd.DataFrame({
    'book_id': [1, 2], 'reviewer_id': [10, 11], 'rating': [5, 1], 'rating_normalized': [1.0, 0.0],
    'language': ['en', 'en'], 'word_count': [20, 30], 'review_text': ['loved it', 'dnf'],
})


def write_clean_dir(path, movies=MOVIES, critics=CRITICS, audience=AUDIENCE, books=BOOKS, book_reviews=BOOK_REVIEWS):
    path.mkdir()
    movies.to_csv(path / 'cleaned_rotten_tomatoes_movies.csv', index=False)
    critics.to_csv(path / 'cleaned_rotten_tomatoes_critic_reviews.csv', index=False)
    audience.to_csv(path / 'cleaned_rotten_tomatoes_audience_reviews.csv', index=False)
    books.to_csv(path / 'cleaned_goodreads_books.csv', index=False)
    book_reviews.to_csv(path / 'cleaned_goodreads_reviews.csv', index=False)
    return path


def test_valid_data_builds_and_passes(tmp_path):
    db = tmp_path / 'w.duckdb'
    assert build(db, write_clean_dir(tmp_path / 'clean'), quiet=True) == {}

    with duckdb.connect(str(db)) as con:
        gap = dict(con.execute('SELECT movie_url, gap FROM film_gap ORDER BY movie_url').fetchall())
        split = dict(con.execute('SELECT movie_url, split FROM film_gap').fetchall())
        sd = dict(con.execute('SELECT title, star_sd FROM books').fetchall())
        genres = con.execute("SELECT genre FROM film_genres WHERE movie_url = 'm/a' ORDER BY genre").fetchall()

    assert gap == {'m/a': 30, 'm/b': -30}            # m/c has no audience score, so it's left out
    assert split == {'m/a': 'critics 20+ higher', 'm/b': 'audiences 20+ higher'}
    assert sd['Even'] == pytest.approx(math.sqrt(2))   # one rating at each of 1-5 stars
    assert sd['Split'] == pytest.approx(2.0)          # half 1-star, half 5-star: as divisive as possible
    assert genres == [('Drama',), ('Horror',)]


def test_checks_catch_bad_data(tmp_path):
    bad_movies = MOVIES.assign(tomatometer=[120, 40, 70])                     # off the 0-100 scale
    bad_critics = pd.concat([CRITICS, CRITICS.iloc[[0]].assign(movie_url='m/missing')])   # review of no film
    failures = build(tmp_path / 'w.duckdb', write_clean_dir(tmp_path / 'clean', movies=bad_movies,
                                                             critics=bad_critics), quiet=True)
    assert {'scores_in_range', 'orphan_reviews'} <= set(failures)
