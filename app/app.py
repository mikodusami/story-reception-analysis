"""Story Reception: an interactive front end for the three research questions.

    streamlit run app/app.py          (or: make app)
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import (  # noqa: E402
    book_genre_divisiveness,
    describe_gap,
    highlight_review,
    load_artifacts,
    predict_film,
    predict_review,
)

BLUE, RED, GRAY, INK2 = '#2a78d6', '#e34948', '#b8b7b2', '#52514e'
REPO_URL = 'https://github.com/mikodusami/story-reception-analysis'

st.set_page_config(page_title='Story Reception', page_icon='📚', layout='wide')
arts = load_artifacts()
metrics = arts['metrics']

st.title('Story Reception')
st.markdown(
    'What makes critics, audiences and readers react the way they do? Three models trained on about 180,000 '
    'Goodreads and Rotten Tomatoes reviews, each scored on books and films it never saw. '
    f'[Code and full analysis]({REPO_URL}).')

tab_review, tab_film, tab_book = st.tabs(['Review → rating', 'Critics vs. audiences', 'Divisive books'])

# ── 1. Review → rating ──────────────────────────────────────────────────────
EXAMPLES = {
    'Write your own': '',
    'A rave': 'An absolute masterpiece. I could not put it down from start to finish, and the ending '
              'left me in tears. Highly recommend.',
    'A pan': 'Tedious, bland and poorly written. I wanted to love it but it was a waste of time. DNF at page 120.',
    'A mixed review': 'The world building is incredible, but the pacing drags in the middle and the romance felt '
                      'forced. Still, the last hundred pages are excellent.',
    'Negation (the model struggles)': 'Not a great movie at all. Not funny, not clever, not worth it.',
    'Sarcasm (the model struggles)': 'Fantastic film if you enjoy watching paint dry for three hours. '
                                     'A true masterpiece of boredom.',
}

with tab_review:
    st.subheader('How many stars will this review give?')
    st.caption('TF-IDF of single words and two-word phrases, with Ridge regression. One model per platform.')
    left, right = st.columns([3, 2], gap='large')
    with left:
        platform = st.radio('Platform', list(arts['reviews']), horizontal=True)
        example = st.selectbox('Start from an example', list(EXAMPLES), index=1)
        text = st.text_area('Review text', value=EXAMPLES[example], height=150,
                            placeholder='Paste or write a review...')
    m = metrics['reviews'][platform]
    with right:
        if text.strip():
            result = predict_review(text, platform, arts)
            lo, hi = result['scale']
            st.metric('Predicted rating', f"{result['stars']:.1f} of {hi:g} stars")
            st.progress(result['normalized'])
            st.caption(f"On {platform} reviews of titles it never saw, this model explains {m['held-out R2']:.0%} "
                       f"of the variation in ratings and is typically off by {m['MAE (stars)']:.2f} stars. "
                       f"Trained on {m['reviews']:,} reviews.")
        else:
            st.info('Write or paste a review to see a prediction.')

    if text.strip():
        st.markdown('**Which words moved the prediction** (blue up, red down; hover for the value)')
        shaded = highlight_review(text, result['contributions'])
        st.markdown(f"<div style='line-height: 1.9; font-size: 1.05rem; margin-bottom: 1.2rem'>{shaded}</div>",
                    unsafe_allow_html=True)
        contrib = result['contributions']
        if len(contrib):
            top = pd.concat([contrib.nsmallest(6), contrib.nlargest(6)]).drop_duplicates()
            top = top[top.abs() > 1e-4]
            chart_df = pd.DataFrame({'term': top.index, 'contribution': top.values})
            bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3, height=14).encode(
                x=alt.X('contribution:Q', title='Contribution to the predicted rating (0–1 scale)'),
                y=alt.Y('term:N', sort='-x', title=None),
                color=alt.condition('datum.contribution > 0', alt.value(BLUE), alt.value(RED)),
                tooltip=['term', alt.Tooltip('contribution:Q', format='+.3f')])
            st.altair_chart(bars.properties(height=28 * len(chart_df) + 20), width='stretch')
        st.caption('A bag-of-words model adds up word signals and ignores word order, so negation ("not great") and '
                   'sarcasm fool it. Those were the main causes of its worst errors in the analysis.')

# ── 2. Critics vs. audiences ────────────────────────────────────────────────
films = arts['films_sample']
fm = metrics['films']
film_models = arts['films']
LANGUAGES = ['English', *film_models['language_vocab'], 'Other']
RATINGS = ['G', 'PG', 'PG-13', 'R', 'Unrated']

with tab_film:
    st.subheader('Will critics and audiences disagree?')
    st.caption('Gap = Tomatometer − audience score, in points. Positive means critics liked the film more. '
               'The models only use what is known before release: genre, content rating, runtime, release timing '
               'and language. No scores.')
    mode = st.radio('Choose', ['Look up a film', 'Describe a new film'], horizontal=True, label_visibility='collapsed')

    if mode == 'Look up a film':
        films = films.assign(label=films['title'] + ' (' + films['release_year'].fillna(0).astype(int).astype(str)
                             .replace('0', '?') + ')')
        options = sorted(films['label'])
        default = 'The Notebook (2004)'                 # audiences far above critics, and the model saw it coming
        choice = st.selectbox(f'{len(films)} widely rated films (type to search)', options,
                              index=options.index(default) if default in options else 0)
        if choice:
            f = films[films['label'] == choice].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Tomatometer', f"{f['tomatometer']:.0f}%", help=f"{int(f['tomatometer_count']):,} critic reviews")
            c2.metric('Audience score', f"{f['audience_score']:.0f}%",
                      help=f"{int(f['audience_review_count']):,} audience ratings")
            c3.metric('Actual gap', f"{f['gap']:+.0f}", help=describe_gap(f['gap']))
            c4.metric('Model predicted', f"{f['predicted gap']:+.0f}",
                      help='Predicted by a model that never saw this film (out-of-fold).')
            st.markdown(f"**Actual:** {describe_gap(f['gap'])}. **Gap model:** {describe_gap(f['predicted gap'])}. "
                        f"**Split model** (trained separately): **{f['split probability']:.0%}** chance of a 20+ point "
                        f"split, against {fm['split base rate']:.0%} for a typical film.")
            st.caption(f"{f['genres']} · {f['content_rating'] if pd.notna(f['content_rating']) else 'Unrated'} · "
                       f"{int(f['runtime_min']) if pd.notna(f['runtime_min']) else '?'} min · {f['original_language']}")
    else:
        with st.form('film'):
            c1, c2, c3 = st.columns(3)
            genres = c1.multiselect('Genres', film_models['genre_vocab'], default=['Horror'])
            rating = c1.selectbox('Content rating', RATINGS, index=4)
            runtime = c2.slider('Runtime (minutes)', 60, 200, 85)
            year = c2.slider('Release year', 1990, 2026, 2024)
            month = c3.selectbox('Release month', list(range(1, 13)), index=9,
                                 format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B'))
            language = c3.selectbox('Original language', LANGUAGES)
            submitted = st.form_submit_button('Predict')
        film = {'genres': genres, 'content_rating': rating, 'runtime_min': runtime, 'release_year': year,
                'release_month': month, 'original_language': None if language == 'Other' else language}
        pred = predict_film(film, arts)
        c1, c2 = st.columns(2)
        c1.metric('Predicted gap', f"{pred['gap']:+.0f} points", help=describe_gap(pred['gap']))
        c2.metric('Chance of a 20+ point split', f"{pred['split probability']:.0%}",
                  delta=f"{(pred['split probability'] - fm['split base rate']) * 100:+.0f} points vs. a typical film",
                  delta_color='off')
        st.markdown(f"**{describe_gap(pred['gap']).capitalize()}.**")

    st.caption(f"Held-out accuracy ({fm['films']:,} films, 10-fold CV): the gap model explains {fm['gap R2']:.0%} "
               f"of the gap and is typically off by {fm['gap MAE (points)']:.0f} points; the split model has ROC-AUC "
               f"{fm['split AUC']:.2f}, and {fm['split top-10% precision']:.0%} of its most confident 10% of split "
               f"calls are right (vs. a {fm['split base rate']:.0%} base rate). Useful for kinds of films, "
               "not exact numbers.")

# ── 3. Divisive books ───────────────────────────────────────────────────────
books = arts['books_sample']
bm = metrics['books']

with tab_book:
    st.subheader('How divided are its readers?')
    st.caption("Divisiveness = the spread (standard deviation) of a book's 1–5 star ratings, compared with what is "
               'typical for its average rating. Highly rated books have less room to spread, so the comparison '
               'matters.')
    labels = books['title'].str.slice(0, 70) + ' — ' + books['author']
    default = labels[books['title'].str.startswith('Twilight')].tolist()
    choice = st.selectbox(f'{len(books)} of the most-rated books', sorted(labels),
                          index=sorted(labels).index(default[0]) if default else 0)
    b = books[labels == choice].iloc[0]

    pct = b['divisiveness percentile']
    verdict = (f'more divisive than {pct:.0%} of books' if pct >= 0.5
               else f'more consensus than {1 - pct:.0%} of books')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Average rating', f"{b['average']:.2f}", help=f"{int(b['ratings_count']):,} ratings")
    c2.metric('Spread (star sd)', f"{b['star sd']:.2f}")
    c3.metric('Typical spread at that average', f"{b['typical sd']:.2f}")
    c4.metric('Divisiveness', f"{b['adjusted']:+.2f}", help='Spread minus the typical spread for its average.')
    st.markdown(f"**{b['title']}** is {verdict} with a similar average rating. Without seeing its ratings, the model "
                f"predicted a spread of {b['predicted sd']:.2f} from its average, genres, length and year.")

    left, right = st.columns([3, 2], gap='large')
    with left:
        points = arts['books_points']
        grid = pd.DataFrame({'average': [3 + i * 0.02 for i in range(101)]})
        grid['star sd'] = arts['books']['curve'].predict(grid[['average']].to_numpy())
        x = alt.X('average:Q', title='Average rating', scale=alt.Scale(domain=[3, 5]))
        y = alt.Y('star sd:Q', title='Spread of ratings (star sd)', scale=alt.Scale(domain=[0, 1.9]))
        base = alt.Chart(points).mark_circle(size=10, opacity=0.25, color=GRAY).encode(x=x, y=y)
        typical = alt.Chart(grid).mark_line(color=INK2, strokeWidth=2).encode(x=x, y=y)
        here = alt.Chart(pd.DataFrame([b])).mark_point(size=220, filled=True, color=BLUE, stroke='white',
                                                        strokeWidth=2, opacity=1).encode(
            x=x, y=y, tooltip=['title', alt.Tooltip('star sd:Q', format='.2f')])
        chart = (base + typical + here).properties(height=320, title='Every book (gray), the typical spread '
                                                   '(line) and this one (blue)')
        st.altair_chart(chart, width='stretch')
    with right:
        dist = pd.DataFrame({'stars': [f'{i}★' for i in range(1, 6)],
                             'share': [b[f'rating_dist_{i}'] for i in range(1, 6)]})
        dist['share'] = dist['share'] / dist['share'].sum()
        bars = alt.Chart(dist).mark_bar(cornerRadiusEnd=3, color=BLUE).encode(
            x=alt.X('stars:N', title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('share:Q', title='Share of ratings', axis=alt.Axis(format='%')),
            tooltip=['stars', alt.Tooltip('share:Q', format='.1%')])
        st.altair_chart(bars.properties(height=320, title='Its ratings'), width='stretch')

    tags = book_genre_divisiveness(b['genres'], arts)
    if len(tags):
        chips = ' '.join(
            f"<span style='display:inline-block;margin:2px;padding:2px 8px;border-radius:10px;"
            f"background:{'rgba(42,120,214,0.15)' if v > 0 else 'rgba(227,73,72,0.15)'}' title='{v:+.3f}'>"
            f"{g} {'↑' if v > 0 else '↓'}</span>" for g, v in tags.items())
        st.markdown('**Its genre tags** (↑ tags whose books tend to divide readers, ↓ tags whose readers agree)')
        st.markdown(chips, unsafe_allow_html=True)

    with st.expander('Most divisive and most consensus books in this sample'):
        cols = ['title', 'author', 'average', 'star sd', 'adjusted']
        a, c = st.columns(2)
        a.markdown('**Most divisive for their average**')
        a.dataframe(books.nlargest(10, 'adjusted')[cols], hide_index=True)
        c.markdown('**Most consensus for their average**')
        c.dataframe(books.nsmallest(10, 'adjusted')[cols], hide_index=True)

    st.caption(f"Held-out accuracy ({bm['books']:,} books, 10 folds grouped by author): the average rating alone "
               f"explains {bm['average-only R2']:.0%} of the spread; the model explains {bm['model R2']:.0%}, or "
               f"{bm['share beyond the average']:.0%} of what the average leaves unexplained.")

st.divider()
st.caption(f"Data scraped from Goodreads and Rotten Tomatoes in spring 2026; this app ships only small samples and "
           f"trained models (trained {metrics['trained']}). Associations, not causes. [Source]({REPO_URL})")
