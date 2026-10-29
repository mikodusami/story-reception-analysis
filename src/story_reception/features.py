"""Features for the critic–audience gap models (RQ2) and the book divisiveness models (RQ3).

Features are split by when they become known:

- pre-release: genre, content rating, runtime, release timing, original language, synopsis.
- post-release: popularity (how many critic and audience ratings a film collected).

Critic and audience *scores* are never features: the target is built from them.
Missing values are left as NaN so model pipelines can impute inside each CV fold.
"""
import numpy as np
import pandas as pd

TARGET_COLUMNS = ('tomatometer', 'audience_score')
# Columns that are, or closely track, one side of the gap. Using them would leak the answer.
LEAKY_COLUMNS = TARGET_COLUMNS + ('critic_avg_rating', 'top_critic_score', 'audience_avg_rating',
                                  'critic_sentiment', 'audience_sentiment', 'audience_all_score',
                                  'audience_all_avg_rating', 'audience_all_sentiment',
                                  'critics_consensus', 'audience_consensus')

CONTENT_RATINGS = ('G', 'PG', 'PG-13', 'R', 'TV', 'Unrated')


def genre_lists(films: pd.DataFrame) -> pd.Series:
    """Genres as lists, from the pipe-separated 'genres' column or an existing 'genres_list' column."""
    if 'genres_list' in films and films['genres_list'].map(lambda v: isinstance(v, list)).all():
        return films['genres_list']
    return films['genres'].fillna('').map(lambda g: [x.strip() for x in g.split('|') if x.strip()])


def genre_flags(genres: pd.Series, min_titles: int, vocab=None) -> pd.DataFrame:
    """One 0/1 column per genre on at least min_titles titles, most common first.

    Pass vocab (a list of genre names saved at training time) to get exactly those columns, e.g. for one new title.
    """
    if vocab is None:
        counts = pd.Series([g for gl in genres for g in gl], dtype=object).value_counts()
        keep = counts[counts >= min_titles].index
    else:
        keep = pd.Index(vocab)
    dummies = genres.map('|'.join).str.get_dummies(sep='|')
    return (dummies.reindex(columns=keep, fill_value=0).astype(int)
            .rename(columns=lambda g: f'genre: {g}').set_axis(genres.index))


def content_rating_group(s: pd.Series) -> pd.Series:
    """MPAA rating, with all TV ratings pooled as 'TV' and missing as 'Unrated'."""
    s = s.fillna('Unrated').astype(str).str.strip()
    return s.where(~s.str.startswith('TV'), 'TV').where(s.isin(CONTENT_RATINGS) | s.str.startswith('TV'), 'Unrated')


def pre_release_features(films: pd.DataFrame, min_genre_films: int = 30, top_languages: int = 8,
                         genre_vocab=None, language_vocab=None) -> pd.DataFrame:
    """One row per film: genre flags, content rating, runtime, release year/month, language flags.

    genre_vocab and language_vocab fix the genre and non-English language columns (as saved at training time);
    without them the columns are chosen from the films passed in.
    """
    genres = genre_lists(films)
    out = genre_flags(genres, min_genre_films, genre_vocab)
    out['genre count'] = genres.str.len()

    rating = content_rating_group(films['content_rating'])
    for r in CONTENT_RATINGS:
        out[f'rated: {r}'] = (rating == r).astype(int)

    out['runtime (min)'] = pd.to_numeric(films['runtime_min'], errors='coerce')
    out['release year'] = pd.to_numeric(films['release_year'], errors='coerce')
    month = pd.to_datetime(films['release_date'], errors='coerce').dt.month
    out['release month'] = month
    out['summer release (May–Aug)'] = month.between(5, 8).astype(int).where(month.notna())
    out['holiday release (Nov–Dec)'] = month.between(11, 12).astype(int).where(month.notna())

    language = films['original_language'].fillna('Unknown').astype(str)
    out['language: English'] = language.str.contains('English').astype(int)
    if language_vocab is None:
        others = language[~language.str.contains('English')].value_counts()
        language_vocab = [lang for lang in others.index[:top_languages] if lang != 'Unknown']
    for lang in language_vocab:
        out[f'language: {lang}'] = (language == lang).astype(int)
    return out


def post_release_features(films: pd.DataFrame) -> pd.DataFrame:
    """Popularity after release: log counts of critic and audience ratings."""
    return pd.DataFrame({
        'log critic count': np.log1p(pd.to_numeric(films['tomatometer_count'], errors='coerce')),
        'log audience count': np.log1p(pd.to_numeric(films['audience_review_count'], errors='coerce')),
    }, index=films.index)


def feature_family(column: str) -> str:
    """Group a feature column into a family, for importance summaries ('genre: Horror' -> 'genre')."""
    if ':' in column:
        return column.split(':', 1)[0]
    families = {'genre count': 'genre', 'runtime (min)': 'runtime', 'release year': 'release timing',
                'release month': 'release timing', 'log critic count': 'popularity',
                'log audience count': 'popularity', 'log pages': 'length', 'publication year': 'publication year',
                'in a series': 'series', 'series position': 'series', 'first in series': 'series',
                'box set or omnibus': 'series',
                'log ratings count': 'popularity', 'award wins': 'awards', 'award nominations': 'awards',
                'star mean': 'average rating'}
    if column.startswith(('summer release', 'holiday release')):
        return 'release timing'
    return families.get(column, 'synopsis' if column.startswith(('synopsis', 'svd', 'emb')) else column)


# ── Books (RQ3: divisiveness) ────────────────────────────────────────────────

STARS = np.arange(1, 6)


def star_stats(books: pd.DataFrame) -> pd.DataFrame:
    """From the 1-5 star counts: number of ratings, mean, standard deviation, and polarization shares."""
    counts = books[[f'rating_dist_{i}' for i in STARS]].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    n = counts.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = counts @ STARS / n
        sd = np.sqrt((counts * (STARS[None, :] - mean[:, None]) ** 2).sum(axis=1) / n)
        low, five = counts[:, :2].sum(axis=1) / n, counts[:, 4] / n
    return pd.DataFrame({'star count': n, 'star mean': mean, 'star sd': sd,
                         'share 1-2 stars': low, 'share 5 stars': five}, index=books.index)


def first_number(s: pd.Series) -> pd.Series:
    """'1-4' -> 1, '2.5' -> 2.5, '' -> NaN."""
    return pd.to_numeric(s.fillna('').astype(str).str.extract(r'(\d+(?:\.\d+)?)', expand=False), errors='coerce')


def book_features(books: pd.DataFrame, min_genre_books: int = 30, genre_vocab=None) -> pd.DataFrame:
    """Known at publication: reader genre tags, length, publication year, series membership."""
    genres = genre_lists(books)
    out = genre_flags(genres, min_genre_books, genre_vocab)
    out['genre count'] = genres.str.len()
    # Caps keep a few bad or unusual values from dominating linear models: comic issue numbers as series
    # positions (404), box sets of 6,000+ pages, and edition years like 1812 for a 2000s omnibus (the only
    # pre-1900 year among modelled books; it can't be told apart from a real 1812 edition, so it's capped).
    pages = pd.to_numeric(books['num_pages'], errors='coerce')
    out['log pages'] = np.log(pages.where(pages > 0).clip(20, 2000))
    out['publication year'] = pd.to_numeric(books['publication_year'], errors='coerce').clip(lower=1900)
    in_series, box_set, position = _series_parts(books)
    out['in a series'] = in_series.astype(int)
    out['series position'] = position.clip(upper=30)
    out['first in series'] = (position == 1).astype(int)
    out['box set or omnibus'] = box_set.astype(int)
    return out


def _series_parts(books: pd.DataFrame):
    in_series = books['series_name'].fillna('').astype(str).str.strip() != ''
    text = books['series_position'].fillna('').astype(str).str.strip()
    box_set = in_series & text.str.contains(r'\d\s*[-–&,]\s*\d', regex=True)   # '1-4', '16-17, 21', '1 & 7'
    # NaN if standalone, a box set, or no number given
    position = first_number(text).where(in_series & ~box_set)
    return in_series, box_set, position


def series_status(books: pd.DataFrame) -> pd.Series:
    """'standalone', 'first in a series', 'later in a series', 'box set or omnibus' or 'series, position unknown'."""
    in_series, box_set, position = _series_parts(books)
    status = np.select([~in_series, box_set, position == 1, position > 1],
                       ['standalone', 'box set or omnibus', 'first in a series', 'later in a series'],
                       'series, position unknown')
    return pd.Series(status, index=books.index)


def book_post_features(books: pd.DataFrame) -> pd.DataFrame:
    """Known only after publication: popularity and awards."""
    awards = books['awards'].fillna('').astype(str)
    return pd.DataFrame({
        'log ratings count': np.log1p(pd.to_numeric(books['ratings_count'], errors='coerce')),
        'award wins': awards.str.count(r'\(WINNER\)'),
        'award nominations': awards.str.count(r'\(NOMINEE\)'),
    }, index=books.index)


def vocab_from_columns(columns, prefix: str) -> list:
    """['genre: Drama', 'runtime (min)', 'genre: Horror'] with 'genre: ' -> ['Drama', 'Horror']."""
    return [c[len(prefix):] for c in columns if c.startswith(prefix)]
