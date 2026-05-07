# What Makes a Story Stick?

### Decoding Critical and Audience Reception in Film and Literature

**Team 4** - QAC Project 3  
Michael-Andre Odusami · Saveer Pande · Daniel Abaye · Anthony Laryukhin

---

## Overview

This project investigates what separates universally beloved stories from critically panned ones, using data scraped from three major review platforms: **Goodreads**, **Rotten Tomatoes**, and **Letterboxd**. We combine web scraping, data cleaning, TF-IDF text analysis, and multi-label regression to answer two core research questions:

1. **Review Language & Reception** - Do the specific words used in reviews predict ratings? Are there lexical signatures that distinguish critic approval from audience approval?
2. **Genre & Trope Performance** - Which genres consistently receive high vs. low ratings, and does this differ between film and literature? Where do critics and audiences systematically disagree?

---

## Repository Structure

```
├── data-collection.ipynb   # Web scraping infrastructure and pipeline (Goodreads, Rotten Tomatoes, Letterboxd)
├── data-cleaning.ipynb     # Data cleaning, normalization, and feature engineering
├── main.ipynb              # Analysis, visualizations, and research conclusions
├── data/
│   ├── goodreads-books.csv
│   ├── goodreads-reviews.csv
│   ├── rotten-tomatoes-movies.csv
│   ├── rotten-tomatoes-critic-reviews.csv
│   ├── rotten-tomatoes-audience-reviews.csv
│   ├── combined_letterboxd_tables.csv
│   └── cleaned/            # Output of data-cleaning.ipynb
│       ├── cleaned_goodreads_books.csv
│       ├── cleaned_goodreads_reviews.csv
│       ├── cleaned_rotten_tomatoes_movies.csv
│       ├── cleaned_rotten_tomatoes_critic_reviews.csv
│       ├── cleaned_rotten_tomatoes_audience_reviews.csv
│       └── cleaned_letterboxd_movies.csv
```

---

## Datasets

| Dataset                     | Size (after cleaning) | Key Fields                                          |
| --------------------------- | --------------------- | --------------------------------------------------- |
| Goodreads Books             | 5,788 books           | avg_rating, genres, description, author             |
| Goodreads Reviews           | 140,576 reviews       | rating (1–5), review_text                           |
| Rotten Tomatoes Movies      | 3,001 films           | tomatometer, audience_score, genres, consensus text |
| RT Critic Reviews           | 25,137 reviews        | rating (parsed to 0–5), sentiment, review_text      |
| RT Audience Reviews         | 15,266 reviews        | score, review_text                                  |
| Letterboxd (Top/Bottom 500) | 1,000 films           | rating (0–5), genres, tier                          |

---

## Notebook Descriptions

### `data-collection.ipynb`

Handles all data acquisition. Key components:

- **Shared scraping infrastructure** - `SyncHttpClient` and `AsyncHttpClient` with exponential backoff, rotating user agents, rate-limit handling (429/403), and Cloudflare bypass. Built on an abstract plugin architecture (`BaseDiscoverer`, `BaseDetailScraper`, `BasePipeline`) so new sources can be added by subclassing.
- **Goodreads pipeline** - Three-phase: (1a) sync crawl of genre-tagged Listopia index pages, (1b) async scraping of individual list pages into a shared queue, (2) async detail scraping of book pages via the embedded `__NEXT_DATA__` Apollo GraphQL cache. Outputs `goodreads-books.csv` and `goodreads-reviews.csv`.
- **Rotten Tomatoes pipeline** - Uses **Playwright** for JS-rendered browse pages (Phase 1, runs in a background thread), then `aiohttp` for detail + review pages (Phase 2, async). Outputs three CSVs: movies, critic reviews, and audience reviews. _Note: Playwright does not run in Google Colab - run this notebook locally._
- **Letterboxd scraper** - Scrapes the official Letterboxd Top 500 and the Anti-Letterboxd Bottom 500 lists using `requests` + `BeautifulSoup`, fetching film metadata and engagement stats via a separate CSI endpoint.
- **Resume support** - All pipelines cache discovered listings to JSON and use `CsvWriter`'s append mode, so a crashed run can be resumed without re-scraping.

### `data-cleaning.ipynb`

Prepares the raw CSVs for analysis:

- Drops duplicates and selects only the columns needed for analysis
- Handles missing values: drops rows with no usable key (book ID, avg rating); fills categoricals with `'Unknown'`; fills counts with 0
- Cleans review text: strips HTML tags, URLs, and encoding artifacts (`â€™`, `\xa0`, etc.); collapses whitespace; drops reviews under 5 words
- Normalizes all ratings to a common 0–1 scale for cross-platform comparison
- Parses RT critic ratings from 15+ mixed formats (fractions, letter grades, star counts, percentages, word numbers) into a uniform 0–5 float
- Adds join-friendly normalized title columns (lowercase, article-stripped) for linking reviews to their parent titles
- Parses genre strings (pipe-separated for Goodreads/RT, comma-separated for Letterboxd) into Python lists
- Saves all six cleaned datasets to `data/cleaned/`

### `main.ipynb`

Loads the cleaned datasets and runs the full analysis:

**Research Question 1 - Review Language & Reception**

- Filters to English-only reviews using a common-word-ratio heuristic
- Fits TF-IDF (5,000 features, unigrams + bigrams) + Linear Regression per platform; extracts the 20 strongest positive and negative word predictors
- Splits reviews into High (≥75th percentile) and Low (≤25th percentile) rating tiers and compares word-frequency profiles
- Builds a cross-platform heatmap comparing how the same words predict ratings differently for Goodreads, RT Audience, and RT Critics
- Reports R² per platform: Goodreads (0.52), RT Critics (0.60), RT Audience (0.75)

**Research Question 2 - Genre & Trope Performance**

- Encodes multi-label genre lists with `MultiLabelBinarizer`, then fits Linear Regression to predict normalized ratings
- Three parallel models: Letterboxd score, Goodreads avg_rating, and RT (critics vs. audience side-by-side)
- Produces a cross-medium genre impact heatmap showing which genres win and lose across all four rating voices

---

## Setup & Usage

### Prerequisites

```bash
pip install aiohttp requests beautifulsoup4 lxml nest_asyncio playwright pandas numpy scikit-learn matplotlib seaborn curl_cffi
playwright install chromium
```

For Colab, also install system deps for Playwright:

```bash
apt-get install -y libatk1.0-0 libatk-bridge2.0-0 libcups2 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 libnspr4 libnss3
```

> **Note:** The Rotten Tomatoes scraper requires Playwright and must be run in a local environment, not Google Colab.

### Running the Notebooks (in order)

1. **`data-collection.ipynb`** - Run the cell at the bottom to launch a pipeline. Uncomment the source(s) you want:

   ```python
   run_pipelines(["goodreads"])          # Goodreads only
   run_pipelines(["rottentomatoes"])     # RT only (local env required)
   run_pipelines(["goodreads", "rottentomatoes"])  # Both
   ```

   Raw CSVs will be written to the `data/` directory (configurable via `DATA_DIR`).

2. **`data-cleaning.ipynb`** - Run all cells. Reads from `data/`, writes cleaned CSVs to `data/cleaned/`.

3. **`main.ipynb`** - Run all cells. Loads from `data/cleaned/` and produces all analysis charts.

If you already have the cleaned CSVs, you can skip straight to `main.ipynb`.

---

## Key Findings

**Review language predicts ratings across all platforms.** The strongest Goodreads predictor is `dnf` (did not finish) at −0.77, followed by skepticism words like _supposedly_ and _apparently_. Critics use diagnostic language (_surface level_, _devoid_, _soulless_) while audiences use simpler emotional terms (_boring_, _loved_, _waste_).

**Genre effects are real but platform-dependent.** Drama and Animation consistently score well across film platforms; Horror carries a large negative coefficient for film but not for literature, where horror readers are a self-selected enthusiast audience. Documentary and Art House genres are rewarded far more by RT critics than by general audiences.

**Critics and audiences speak differently.** The RT critic/audience split is most visible in the cross-platform genre heatmap: critics heavily reward Documentary and Variety content; audiences are relatively more tolerant of Horror and Action.

---

## Credits

| Contributor           | Work                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Michael-Andre Odusami | Scraping infrastructure, Goodreads scraper, Rotten Tomatoes scraper, data cleaning, text analytics, lexical analysis |
| Daniel Abaye          | Letterboxd scraper, data cleaning, data analysis & visualization, genre analysis                                     |
| Saveer Pande          | QAC formatting                                                                                                       |
| Anthony Laryukhin     | Additional contributions                                                                                             |
