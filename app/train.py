"""Train the models the Streamlit app serves, score them the way the notebooks do, and save them to app/artifacts/.

    python app/train.py          (or: make train-app)

Reads data/cleaned/ (not in the repo). Writes small artifacts that are committed so the app runs without the data:
review models, film-gap models, the book-divisiveness model, metrics.json, and samples of ~300 well-known films and
books. Sample predictions are out-of-fold: no title is shown a prediction from a model that trained on it.
"""
import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from story_reception.features import book_features, genre_lists, pre_release_features, star_stats, vocab_from_columns

SEED = 42
REPO = Path(__file__).resolve().parents[1]
CLEAN = REPO / 'data' / 'cleaned'
ART = Path(__file__).resolve().parent / 'artifacts'
SAMPLE_SIZE = 300

REVIEW_SOURCES = {   # platform: (file, title key, rating scale in stars: lowest, highest)
    'Goodreads': ('cleaned_goodreads_reviews.csv', 'book_id', (1.0, 5.0)),
    'RT audience': ('cleaned_rotten_tomatoes_audience_reviews.csv', 'movie_url', (0.5, 5.0)),
    'RT critics': ('cleaned_rotten_tomatoes_critic_reviews.csv', 'movie_url', (0.0, 5.0)),
}


def log(msg, start):
    print(f'{msg} ({time.time() - start:.0f}s)', flush=True)


def make_vectorizer():
    # Same settings as main.ipynb
    return TfidfVectorizer(max_features=5000, stop_words='english', min_df=5, max_df=0.95,
                           ngram_range=(1, 2), token_pattern=r'(?u)\b[a-zA-Z]{3,}\b', dtype=np.float32)


def train_review_models(metrics):
    models = {}
    for platform, (fname, key, scale) in REVIEW_SOURCES.items():
        start = time.time()
        df = pd.read_csv(CLEAN / fname, usecols=[key, 'review_text', 'rating_normalized', 'language'])
        df = df[df['language'].eq('en') & df['rating_normalized'].notna()].reset_index(drop=True)
        y, groups = df['rating_normalized'].to_numpy(), df[key].to_numpy()

        # Held-out score on the same 20% of titles as main.ipynb (Ridge alpha 3 was chosen there by grouped CV)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, y, groups))
        vec = make_vectorizer()
        ridge = Ridge(alpha=3).fit(vec.fit_transform(df['review_text'].iloc[tr]), y[tr])
        pred = np.clip(ridge.predict(vec.transform(df['review_text'].iloc[te])), 0, 1)
        lo, hi = scale
        metrics['reviews'][platform] = {'held-out R2': round(r2_score(y[te], pred), 3),
                                        'MAE (stars)': round(mean_absolute_error(y[te], pred) * (hi - lo), 2),
                                        'reviews': int(len(df)), 'titles': int(len(set(groups)))}

        vec = make_vectorizer()                         # final model: all reviews
        ridge = Ridge(alpha=3).fit(vec.fit_transform(df['review_text']), y)
        vec.stop_words_ = None                          # large and only needed for fitting
        models[platform] = {'vectorizer': vec, 'ridge': ridge, 'scale': scale}
        log(f"  reviews / {platform}: held-out R² {metrics['reviews'][platform]['held-out R2']}", start)
    joblib.dump(models, ART / 'review_models.joblib', compress=3)


def top_decile_precision(y, prob, rng):
    k = max(1, int(round(0.1 * len(prob))))
    return y[np.lexsort((rng.random(len(prob)), -prob))[:k]].mean()


def train_gap_models(metrics):
    start = time.time()
    movies = pd.read_csv(CLEAN / 'cleaned_rotten_tomatoes_movies.csv')
    films = movies.dropna(subset=['tomatometer', 'audience_score']).reset_index(drop=True)
    gap = (films['tomatometer'] - films['audience_score']).to_numpy()
    split = (np.abs(gap) >= 20).astype(int)
    X = pre_release_features(films)

    # Same models as gap-analysis.ipynb: random forest for the gap, gradient boosting for the 20+ point split
    reg = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33,
                                              n_jobs=-1, random_state=SEED))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                         n_iter_no_change=20, random_state=SEED)

    r2s, maes, aucs, precs = [], [], [], []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X):
        p = clone(reg).fit(X.iloc[tr], gap[tr]).predict(X.iloc[te])
        r2s.append(r2_score(gap[te], p))
        maes.append(mean_absolute_error(gap[te], p))
    rng = np.random.default_rng(SEED)
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X, split):
        prob = clone(clf).fit(X.iloc[tr], split[tr]).predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(split[te], prob))
        precs.append(top_decile_precision(split[te], prob, rng))
    metrics['films'] = {'films': int(len(films)), 'gap R2': round(np.mean(r2s), 3), 'gap R2 sd': round(np.std(r2s), 3),
                        'gap MAE (points)': round(np.mean(maes), 1), 'split AUC': round(np.mean(aucs), 3),
                        'split top-10% precision': round(np.mean(precs), 3), 'split base rate': round(split.mean(), 3)}
    log(f"  films: gap R² {metrics['films']['gap R2']}, split AUC {metrics['films']['split AUC']}", start)

    # Out-of-fold predictions for the sample, then final models on all films
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    films['predicted gap'] = cross_val_predict(reg, X, gap, cv=folds)
    films['split probability'] = cross_val_predict(clf, X, split, cv=folds, method='predict_proba')[:, 1]
    joblib.dump({'gap': reg.fit(X, gap), 'split': clf.fit(X, split), 'columns': list(X.columns),
                 'genre_vocab': vocab_from_columns(X.columns, 'genre: '),
                 'language_vocab': [v for v in vocab_from_columns(X.columns, 'language: ') if v != 'English']},
                ART / 'film_models.joblib', compress=3)

    sample = films.nlargest(SAMPLE_SIZE, 'audience_review_count')
    sample = sample.assign(gap=sample['tomatometer'] - sample['audience_score'],
                           genres=genre_lists(sample).map(', '.join))
    sample[['title', 'release_year', 'release_date', 'content_rating', 'runtime_min', 'original_language', 'genres',
            'tomatometer', 'tomatometer_count', 'audience_score', 'audience_review_count', 'gap', 'predicted gap',
            'split probability']].round(3).to_csv(ART / 'films_sample.csv', index=False)


def train_book_model(metrics):
    start = time.time()
    raw = pd.read_csv(CLEAN / 'cleaned_goodreads_books.csv')
    stats = star_stats(raw)
    keep = (stats['star count'] >= 100) & raw['author'].notna() & raw['genres'].fillna('').ne('')
    books, stats = raw[keep].reset_index(drop=True), stats[keep].reset_index(drop=True)
    sd, avg, authors = stats['star sd'].to_numpy(), stats['star mean'].to_numpy(), books['author'].to_numpy()
    X = book_features(books)
    X['star mean'] = avg

    curve = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(alpha=1e-3)).fit(avg[:, None], sd)
    avg_only = make_pipeline(ColumnTransformer([('avg', SplineTransformer(n_knots=6, degree=3), ['star mean'])]),
                             Ridge(alpha=1))
    # Settings as in divisive-books.ipynb; rounds and leaves fixed at a mid-grid choice for speed
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=SEED)
    r2_model, r2_avg = [], []
    for seed in (SEED, SEED + 1):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, sd, authors):
            r2_model.append(r2_score(sd[te], clone(model).fit(X.iloc[tr], sd[tr]).predict(X.iloc[te])))
            r2_avg.append(r2_score(sd[te], clone(avg_only).fit(X.iloc[tr], sd[tr]).predict(X.iloc[te])))
    beyond = 1 - (1 - np.mean(r2_model)) / (1 - np.mean(r2_avg))
    metrics['books'] = {'books': int(len(books)), 'authors': int(len(set(authors))),
                        'average-only R2': round(np.mean(r2_avg), 3), 'model R2': round(np.mean(r2_model), 3),
                        'model R2 sd': round(np.std(r2_model), 3), 'share beyond the average': round(beyond, 3)}
    log(f"  books: R² {metrics['books']['model R2']} ({beyond:.0%} beyond the average)", start)

    oof = cross_val_predict(model, X, sd, cv=GroupKFold(n_splits=5, shuffle=True, random_state=SEED), groups=authors)
    expected = curve.predict(avg[:, None])
    books = books.assign(**{'star sd': sd, 'average': avg, 'typical sd': expected, 'adjusted': sd - expected,
                            'predicted sd': oof})
    books['divisiveness percentile'] = books['adjusted'].rank(pct=True)
    joblib.dump({'model': model.fit(X, sd), 'curve': curve, 'columns': list(X.columns),
                 'genre_vocab': vocab_from_columns(X.columns, 'genre: ')}, ART / 'book_model.joblib', compress=3)

    # Average adjusted divisiveness per genre tag (30+ books), for explaining a book's tags in the app
    genre_cols = [c for c in X.columns if c.startswith('genre: ')]
    tag_means = {c[len('genre: '):]: round(float(books.loc[X[c] == 1, 'adjusted'].mean()), 4) for c in genre_cols}
    (ART / 'genre_divisiveness.json').write_text(json.dumps(tag_means, indent=1))

    sample = books.nlargest(SAMPLE_SIZE, 'ratings_count').assign(genres=lambda d: genre_lists(d).map(', '.join))
    sample[['title', 'author', 'publication_year', 'ratings_count', 'average', 'star sd', 'typical sd', 'adjusted',
            'divisiveness percentile', 'predicted sd', 'genres',
            *[f'rating_dist_{i}' for i in range(1, 6)]]].round(4).to_csv(ART / 'books_sample.csv', index=False)
    # Everyone's (average, sd) for the app's scatter, rounded and without titles
    pd.DataFrame({'average': avg.round(3), 'star sd': sd.round(3)}).to_csv(ART / 'books_points.csv', index=False)


def main():
    ART.mkdir(exist_ok=True)
    metrics = {'trained': date.today().isoformat(), 'reviews': {}}
    start = time.time()
    train_review_models(metrics)
    train_gap_models(metrics)
    train_book_model(metrics)
    (ART / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    log('Saved artifacts to app/artifacts/', start)
    for f in sorted(ART.iterdir()):
        print(f'  {f.name:28s} {f.stat().st_size / 1e6:6.2f} MB')


if __name__ == '__main__':
    main()
