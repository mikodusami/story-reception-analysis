import math

import pandas as pd
import pytest

from story_reception.features import (
    LEAKY_COLUMNS,
    book_features,
    book_post_features,
    content_rating_group,
    feature_family,
    post_release_features,
    pre_release_features,
    series_status,
    star_stats,
)

FILMS = pd.DataFrame({
    'genres': ['Horror|Drama', 'Comedy', 'Drama', None],
    'content_rating': ['R', 'TV-MA', None, 'NR'],
    'runtime_min': [95, None, 130, 80],
    'release_year': [2020, 2021, None, 1999],
    'release_date': ['2020-07-04', '2021-12-20', None, '1999-03-01'],
    'original_language': ['English', 'British English', 'French (France)', None],
    'tomatometer_count': [100, 0, 50, None],
    'audience_review_count': [1000, 10, None, 0],
    'tomatometer': [90, 40, 70, 10], 'audience_score': [60, 70, 50, 20],
})


def test_content_rating_group():
    ratings = pd.Series(['G', 'PG-13', 'TV-14', 'TV-MA', None, 'NR', 'R'])
    assert content_rating_group(ratings).tolist() == ['G', 'PG-13', 'TV', 'TV', 'Unrated', 'Unrated', 'R']


def test_pre_release_features():
    X = pre_release_features(FILMS, min_genre_films=2, top_languages=3)
    assert list(X.filter(like='genre: ').columns) == ['genre: Drama']     # only Drama appears twice
    assert X['genre: Drama'].tolist() == [1, 0, 1, 0]
    assert X['genre count'].tolist() == [2, 1, 1, 0]
    assert X['rated: R'].tolist() == [1, 0, 0, 0]
    assert X['rated: Unrated'].tolist() == [0, 0, 1, 1]
    assert X['rated: TV'].tolist() == [0, 1, 0, 0]
    assert X['language: English'].tolist() == [1, 1, 0, 0]               # British English counts as English
    assert X['language: French (France)'].tolist() == [0, 0, 1, 0]
    assert 'language: Unknown' not in X
    assert X['release month'].tolist()[:2] == [7, 12]
    assert X['summer release (May–Aug)'].tolist()[0] == 1
    assert X['holiday release (Nov–Dec)'].tolist()[1] == 1
    assert math.isnan(X['runtime (min)'].iloc[1])                         # missing stays missing


def test_no_feature_comes_from_the_target():
    X = pre_release_features(FILMS, min_genre_films=1)
    assert not set(X.columns) & set(LEAKY_COLUMNS)
    changed = FILMS.assign(tomatometer=[0, 0, 0, 0], audience_score=[100, 100, 100, 100])
    pd.testing.assert_frame_equal(X, pre_release_features(changed, min_genre_films=1))


def test_post_release_features():
    P = post_release_features(FILMS)
    assert P['log audience count'].iloc[0] == math.log1p(1000)
    assert P['log critic count'].iloc[1] == 0
    assert math.isnan(P['log critic count'].iloc[3])


def test_feature_family():
    assert [feature_family(c) for c in ['genre: Horror', 'genre count', 'rated: PG', 'runtime (min)',
                                        'release month', 'summer release (May–Aug)', 'language: Hindi',
                                        'log audience count', 'synopsis svd 3']] == \
        ['genre', 'genre', 'rated', 'runtime', 'release timing', 'release timing', 'language',
         'popularity', 'synopsis']


BOOKS = pd.DataFrame({
    'rating_dist_1': [1, 50, 0, 0], 'rating_dist_2': [1, 0, 0, 0], 'rating_dist_3': [1, 0, 0, 0],
    'rating_dist_4': [1, 0, 10, 0], 'rating_dist_5': [1, 50, 0, 0],
    'genres': ['Fantasy|Fiction', 'Romance', 'Fantasy', None],
    'num_pages': [300, 0, None, 120], 'publication_year': [2001, 2015, None, 1990],
    'series_name': ['Saga', '', 'Saga', 'Cycle'], 'series_position': ['1-4', '', '2.5', ''],
    'ratings_count': [5, 100, 10, 0],
    'awards': ['Hugo Award (WINNER)|Nebula Award (NOMINEE)|Locus Award (NOMINEE)', '', None, 'X (WINNER)'],
})


def test_star_stats():
    st = star_stats(BOOKS)
    assert st['star count'].tolist() == [5, 100, 10, 0]
    assert st['star mean'].iloc[0] == pytest.approx(3.0)
    assert st['star sd'].iloc[0] == pytest.approx(math.sqrt(2))   # one rating at each star
    assert st['star sd'].iloc[1] == pytest.approx(2.0)            # half 1-star, half 5-star: the maximum
    assert st['star sd'].iloc[2] == pytest.approx(0.0)            # everyone gave 4 stars
    assert st['share 1-2 stars'].iloc[1] == pytest.approx(0.5)
    assert st.iloc[3].isna()[['star mean', 'star sd']].all()      # no ratings


def test_book_features():
    X = book_features(BOOKS, min_genre_books=2)
    assert list(X.filter(like='genre: ').columns) == ['genre: Fantasy']
    assert X['log pages'].iloc[0] == pytest.approx(math.log(300))
    assert math.isnan(X['log pages'].iloc[1])                     # 0 pages is treated as missing
    assert X['in a series'].tolist() == [1, 0, 1, 1]
    assert X['box set or omnibus'].tolist() == [1, 0, 0, 0]          # '1-4' is a box set, not book 1
    assert math.isnan(X['series position'].iloc[0]) and X['series position'].iloc[2] == 2.5
    assert math.isnan(X['series position'].iloc[3])                  # in a series, position unknown
    assert X['first in series'].tolist() == [0, 0, 0, 0]


def test_series_status():
    books = pd.DataFrame({'series_name': ['', 'S', 'S', 'S', 'S', 'S', None],
                          'series_position': ['', '1', '3', '1-3', '16-17, 21', ' Webcomic', None]})
    assert series_status(books).tolist() == ['standalone', 'first in a series', 'later in a series',
                                             'box set or omnibus', 'box set or omnibus',
                                             'series, position unknown', 'standalone']


def test_book_post_features():
    P = book_post_features(BOOKS)
    assert P['award wins'].tolist() == [1, 0, 0, 1]
    assert P['award nominations'].tolist() == [2, 0, 0, 0]
    assert P['log ratings count'].iloc[1] == pytest.approx(math.log1p(100))


def test_fixed_vocab_reproduces_training_columns():
    from story_reception.features import vocab_from_columns
    full = pre_release_features(FILMS, min_genre_films=1, top_languages=3)
    one = FILMS.iloc[[1]]                                     # a single new film
    fixed = pre_release_features(one, genre_vocab=vocab_from_columns(full.columns, 'genre: '),
                                 language_vocab=vocab_from_columns(full.columns, 'language: ')[1:])
    assert list(fixed.columns) == list(full.columns)
    pd.testing.assert_frame_equal(fixed, full.iloc[[1]], check_dtype=False)

    books_full = book_features(BOOKS, min_genre_books=1)
    books_one = book_features(BOOKS.iloc[[2]], genre_vocab=vocab_from_columns(books_full.columns, 'genre: '))
    assert list(books_one.columns) == list(books_full.columns)
