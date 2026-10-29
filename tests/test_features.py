import math

import pandas as pd

from story_reception.features import (
    LEAKY_COLUMNS,
    content_rating_group,
    feature_family,
    post_release_features,
    pre_release_features,
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
