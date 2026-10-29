import math

import pandas as pd
import pytest

from story_reception.clean import (
    clean_review_text,
    normalize_title,
    parse_critic_rating,
    parse_genres,
    runtime_to_minutes,
)


@pytest.mark.parametrize('raw, expected', [
    # fractions, with and without stray spaces
    ('3/5', 3.0), ('3.5 / 5', 3.5), ('3/ 5', 3.0), ('2. 5 / 5', 2.5),
    ('7/10', 3.5), ('75/100', 3.75), ('18/20', 4.5), ('.5/4', 0.62),
    # percentages
    ('75%', 3.75), ('70%', 3.5),
    # letter grades, including spelled-out and spaced forms
    ('A+', 5.0), ('A-', 4.3), ('B+', 4.0), ('C', 2.7), ('F', 0.5),
    ('A-minus', 4.3), ('B-plus', 4.0), ('C-plus', 3.0), ('C +', 3.0), ('A minus', 4.3), ('C=', 2.7),
    # stars in words and numbers
    ('FIVE STARS', 5.0), ('THREE STARS', 3.0), ('half star', 0.5),
    ('3 stars', 3.0), ('1 star', 1.0), ('.5 stars', 0.5),
    ('3 1/2 stars', 3.5), ('3 1/4 stars', 3.25),
    # mixed fraction with an explicit scale
    ('3 1/2/5', 3.5),
    # "X out of Y" / "X of Y"
    ('2 out of 5', 2.0), ('4.5 of 5', 4.5), ('3.5 out of 5', 3.5),
    # plain numbers: scale guessed from size
    ('4', 4.0), ('7.5', 3.75), ('55', 2.75), ('9.5 ', 4.75),
    # qualitative labels and custom scales
    ('Must Watch', 5.0), ('Highly Recommended', 4.5), ('Recommended', 4.0),
    ('Catch It On Cable', 2.5), ('4 Ripples', 4.0),
])
def test_parse_critic_rating_formats(raw, expected):
    assert parse_critic_rating(raw) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize('raw', [
    '7.5/5', '8.5/5', '29/10',          # above the scale
    '0/0', '3/0',                        # zero denominator
    '', '   ', None, float('nan'), 'nan',
    'not a rating', '250',              # unreadable, or a number too big to guess a scale for
])
def test_parse_critic_rating_returns_nan(raw):
    assert math.isnan(parse_critic_rating(raw))


def test_parse_critic_rating_always_within_scale():
    values = ['3/5', 'A+', '100', '5', '0', 'F-', '100%', '5 out of 5']
    for v in values:
        assert 0 <= parse_critic_rating(v) <= 5, v


def test_clean_review_text():
    raw = pd.Series([
        'A cliché-ridden, naïve plot — Brontë it isn’t \U0001F644 https://t.co/x',
        '<p>Great <b>film</b></p>   see www.example.com',
        'donâ€™t   miss it',
        None,
    ])
    assert clean_review_text(raw).tolist() == [
        "A cliche-ridden, naive plot - Bronte it isn't",
        'Great film see',
        "don't miss it",
        '',
    ]


def test_clean_review_text_output_is_ascii():
    out = clean_review_text(pd.Series(['日本語のレビュー mixed with English', 'Ça va 👍']))
    assert all(s.isascii() for s in out)
    assert out.tolist() == ['mixed with English', 'Ca va']


def test_normalize_title():
    titles = pd.Series(['The Matrix', '  A   Quiet Place ', 'An American Tail', 'Theory of Everything', None])
    assert normalize_title(titles).tolist() == ['matrix', 'quiet place', 'american tail', 'theory of everything', '']


def test_parse_genres():
    genres = pd.Series(['Fantasy|Fiction| Epic Fantasy', 'Drama', '', None, '||'])
    assert parse_genres(genres).tolist() == [['Fantasy', 'Fiction', 'Epic Fantasy'], ['Drama'], [], [], []]


def test_runtime_to_minutes():
    out = runtime_to_minutes(pd.Series(['1h 25m', '45m', '2h', '', 'unknown']))
    assert out.iloc[:3].tolist() == [85, 45, 120]
    assert out.iloc[3:].isna().all()
