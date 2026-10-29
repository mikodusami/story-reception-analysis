"""Film features for the critic–audience gap models (RQ2).

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


def content_rating_group(s: pd.Series) -> pd.Series:
    """MPAA rating, with all TV ratings pooled as 'TV' and missing as 'Unrated'."""
    s = s.fillna('Unrated').astype(str).str.strip()
    return s.where(~s.str.startswith('TV'), 'TV').where(s.isin(CONTENT_RATINGS) | s.str.startswith('TV'), 'Unrated')


def pre_release_features(films: pd.DataFrame, min_genre_films: int = 30, top_languages: int = 8) -> pd.DataFrame:
    """One row per film: genre flags, content rating, runtime, release year/month, language flags."""
    out = pd.DataFrame(index=films.index)

    genres = genre_lists(films)
    counts = pd.Series([g for gl in genres for g in gl]).value_counts()
    for genre in counts[counts >= min_genre_films].index:
        out[f'genre: {genre}'] = genres.map(lambda gl, g=genre: int(g in gl))
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
    others = language[~language.str.contains('English')].value_counts()
    for lang in others.index[:top_languages]:
        if lang != 'Unknown':
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
                'log audience count': 'popularity'}
    if column.startswith(('summer release', 'holiday release')):
        return 'release timing'
    return families.get(column, 'synopsis' if column.startswith(('synopsis', 'svd', 'emb')) else column)
