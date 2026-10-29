"""Rotten Tomatoes pipeline: Playwright scrolls the browse pages on a background thread and feeds an async
queue of movie, critic-review and audience-review scrapes. Writes three rotten-tomatoes-*.csv files.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

import bs4

# Playwright is only needed for the JavaScript-rendered browse pages.
try:
    from playwright.sync_api import Page, sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # the module still imports; RTBrowseCollector will fail if used
    PLAYWRIGHT_AVAILABLE = False

from .base import (
    AsyncHttpClient,
    BaseDetailScraper,
    BaseDiscoverer,
    BasePipeline,
    BaseScraperConfig,
    CsvWriter,
    setup_logger,
)


@dataclass
class MovieListing:
    """Lightweight record from a browse page tile."""
    title: str                      # movie title from the tile
    url: str                        # full URL (e.g. https://www.rottentomatoes.com/m/the_matrix)
    critic_score_browse: str = ""   # tomatometer shown on the tile (may differ from detail page)
    audience_score_browse: str = "" # audience score shown on the tile


@dataclass
class MovieDetail:
    """Full metadata scraped from an individual movie page.

    Combines data from four sources on the page:
    1. JSON-LD (name, rating, genres, directors, actors)
    2. media-scorecard-json (detailed critic/audience/top-critic scores)
    3. Movie Info section (runtime, box office, distributor)
    4. Consensus divs (critic and audience consensus text)
    """
    title: str = ""
    url: str = ""
    critic_score_browse: str = ""     # from browse tile (for comparison)
    audience_score_browse: str = ""   # from browse tile (for comparison)
    name: str = ""                    # official movie name from JSON-LD
    content_rating: str = ""          # e.g. "PG-13", "R"
    release_date: str = ""
    genres: str = ""                  # pipe-separated: "Action|Adventure|Sci-Fi"
    directors: str = ""               # pipe-separated director names
    actors: str = ""                  # pipe-separated actor names
    tomatometer: str = ""             # critic score (0-100)
    tomatometer_count: str = ""       # number of critic reviews
    critic_avg_rating: str = ""       # average critic rating (e.g. "7.5/10")
    critic_sentiment: str = ""        # "positive", "negative", "mixed"
    critic_certified: str = ""        # whether it's "Certified Fresh"
    critic_liked: str = ""            # count of positive critic reviews
    critic_not_liked: str = ""        # count of negative critic reviews
    top_critic_score: str = ""        # tomatometer from top critics only
    top_critic_count: str = ""
    top_critic_liked: str = ""
    top_critic_not_liked: str = ""
    top_critic_sentiment: str = ""
    top_critic_certified: str = ""
    audience_score: str = ""          # verified audience score (0-100)
    audience_avg_rating: str = ""     # average audience rating (e.g. "4.2/5")
    audience_review_count: str = ""
    audience_sentiment: str = ""
    audience_certified: str = ""
    audience_liked: str = ""
    audience_not_liked: str = ""
    audience_rating_band: str = ""    # distribution across rating bands
    audience_score_type: str = ""     # "verified" or "all"
    audience_all_score: str = ""      # score including unverified reviews
    audience_all_avg_rating: str = ""
    audience_all_review_count: str = ""
    audience_all_liked: str = ""
    audience_all_not_liked: str = ""
    audience_all_sentiment: str = ""
    synopsis: str = ""
    runtime: str = ""                 # e.g. "2h 16m"
    box_office_usa: str = ""
    original_language: str = ""
    release_date_theaters: str = ""
    release_date_streaming: str = ""
    distributor: str = ""
    production_co: str = ""
    critics_consensus: str = ""       # one-line critic summary
    audience_consensus: str = ""      # one-line audience summary


@dataclass
class CriticReview:
    """A single critic review from a movie's /reviews page."""
    movie_title: str = ""
    movie_url: str = ""
    critic_name: str = ""
    publication: str = ""    # e.g. "The New York Times"
    sentiment: str = ""      # "positive" or "negative"
    rating: str = ""         # original rating text (e.g. "B+", "3.5/4")
    review_text: str = ""


@dataclass
class AudienceReview:
    """A single audience review from a movie's /reviews?type=user page."""
    movie_title: str = ""
    movie_url: str = ""
    user_name: str = ""
    score: str = ""          # star rating (0.5 to 5.0)
    review_text: str = ""
    verified: str = ""       # "true" if the reviewer has a verified ticket purchase


def _build_category_urls() -> tuple[str, ...]:
    """Generate a comprehensive set of RT browse page URLs.

    We combine multiple filter axes to get diverse movies:
      Tier 0: Unfiltered sort-only pages (broadest mix, catches worst-rated films)
      Tier 1: Rating filter × sort order (rating diversity)
      Tier 2: Genre × sort order (fills genre gaps)
      Tier 3: Genre × rating combos (catches niche intersections)

    This produces ~700+ unique URLs. We process them in tier order, and the
    adaptive skipping logic in the browse collector will bail on low-yield pages.
    """
    base = "https://www.rottentomatoes.com/browse"
    paths = ["movies_at_home", "movies_in_theaters"]  # two main movie sections

    rating_filters = [
        "audience:verified_hot", "audience:upright", "audience:spilled",  # audience ratings
        "critics:certified_fresh", "critics:fresh", "critics:rotten",      # critic ratings
    ]
    genre_filters = [
        "action", "adventure", "animation", "comedy", "crime",
        "documentary", "drama", "fantasy", "history", "horror",
        "kids_and_family", "music", "mystery_and_thriller", "romance",
        "sci_fi", "sports", "war", "western", "biography", "anime",
    ]
    sort_orders = [
        "popular", "newest",
        "critic_highest", "critic_lowest",
        "audience_highest", "audience_lowest",
    ]
    urls: list[str] = []

    # tier 0: no filters, just sort order — surfaces the broadest mix
    for path in paths:
        for sort in sort_orders:
            urls.append(f"{base}/{path}/sort:{sort}")

    # tier 1: rating filter × sort — ensures we get spilled AND fresh movies
    for path in paths:
        for rating in rating_filters:
            for sort in sort_orders:
                urls.append(f"{base}/{path}/{rating}~sort:{sort}")

    # tier 2: genre × sort — fills genre diversity gaps
    for path in paths:
        for genre in genre_filters:
            for sort in sort_orders:
                urls.append(f"{base}/{path}/genres:{genre}~sort:{sort}")

    # tier 3: genre × rating combos (popular sort only to keep count manageable)
    for path in paths:
        for genre in genre_filters:
            for rating in rating_filters:
                urls.append(f"{base}/{path}/{rating}~genres:{genre}~sort:popular")

    return tuple(urls)


@dataclass(frozen=True)
class RottenTomatoesConfig(BaseScraperConfig):
    """RT-specific settings extending the shared base config."""
    category_urls: tuple[str, ...] = _build_category_urls()  # all browse page URLs

    target_movies: int = 10_000              # stop browsing once we have this many unique movies
    max_scroll_per_category: int = 8       # max "Load More" clicks per browse page
    scroll_pause: float = 0.3              # settle time after each "Load More" click

    max_critic_reviews: int = 50           # critic reviews to scrape per movie
    max_audience_reviews: int = 50         # audience reviews to scrape per movie

    detail_concurrency: int = 10           # Phase 2 concurrency (10+ triggers RT blocking)
    min_new_ratio: float = 0.05            # skip browse page if < 5% of movies are new

    movies_csv: str = "rotten-tomatoes-movies.csv"
    critic_reviews_csv: str = "rotten-tomatoes-critic-reviews.csv"
    audience_reviews_csv: str = "rotten-tomatoes-audience-reviews.csv"
    listings_cache: str = "movie-listings-cache.json"

    # RT uses a single user agent (Playwright browser context matches it)
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    )


class BrowsePageParser:
    """Parses movie tiles from a RT browse/discovery page."""

    @staticmethod
    def parse(html: str) -> list[MovieListing]:
        soup = bs4.BeautifulSoup(html, "lxml")
        movies: list[MovieListing] = []

        # each movie tile is a <div data-qa="discovery-media-list-item">
        tiles = soup.select('[data-qa="discovery-media-list-item"]')
        for tile in tiles:
            # try the caption link first (most common)
            link = tile.select_one('[data-qa="discovery-media-list-item-caption"]')
            href = link.get("href", "") if link else ""

            # fallback: some tiles have the link on a parent <a>
            if not href:
                parent_link = tile.find_parent("a")
                if parent_link:
                    href = parent_link.get("href", "")

            # fallback: look for any <a> with an /m/ href inside the tile
            if not href:
                any_movie_link = tile.select_one('a[href^="/m/"]')
                if any_movie_link:
                    href = any_movie_link.get("href", "")

            # extract the movie title
            title_el = (link or tile).select_one('[data-qa="discovery-media-list-item-title"]')
            if not title_el:
                title_el = tile.select_one('[data-qa="discovery-media-list-item-title"]')
            title = title_el.get_text(strip=True) if title_el else ""

            # only keep movie links (href starts with /m/)
            if not href or not href.startswith("/m/"):
                continue

            # extract scores from the tile
            critic_el = (link or tile).select_one('[slot="criticsScore"]')
            audience_el = (link or tile).select_one('[slot="audienceScore"]')

            movies.append(MovieListing(
                title=title,
                url="https://www.rottentomatoes.com" + href,
                critic_score_browse=critic_el.get_text(strip=True) if critic_el else "",
                audience_score_browse=audience_el.get_text(strip=True) if audience_el else "",
            ))

        return movies


class MoviePageParser:
    """Extracts structured metadata from an individual RT movie page.

    Combines four data sources into a single flat dict:
    1. JSON-LD: basic movie info (name, rating, genres, cast)
    2. media-scorecard-json: detailed scores (tomatometer, audience, top critics)
    3. Movie info section: runtime, box office, distributor, etc.
    4. Consensus divs: critic and audience consensus text
    """

    @staticmethod
    def parse(html: str) -> dict:
        soup = bs4.BeautifulSoup(html, "lxml")
        details: dict = {}
        MoviePageParser._parse_json_ld(soup, details)           # source 1
        MoviePageParser._parse_scorecard_json(soup, details)    # source 2
        MoviePageParser._parse_movie_info_section(soup, details)  # source 3
        MoviePageParser._parse_consensus(soup, details)         # source 4
        return details

    @staticmethod
    def _parse_json_ld(soup: bs4.BeautifulSoup, details: dict) -> None:
        """Extract basic info from the JSON-LD structured data block."""
        ld_tag = soup.find("script", type="application/ld+json")
        if not ld_tag or not ld_tag.string:
            return
        try:
            ld = json.loads(ld_tag.string)
        except (json.JSONDecodeError, TypeError):
            return

        details["name"] = ld.get("name", "")
        details["content_rating"] = ld.get("contentRating", "")
        details["release_date"] = ld.get("dateCreated", "")

        # genres can be a list or a single string
        genres = ld.get("genre", [])
        details["genres"] = "|".join(genres) if isinstance(genres, list) else str(genres)

        # directors can be a list of dicts or a single dict
        directors = ld.get("director", [])
        if isinstance(directors, list):
            details["directors"] = "|".join(d.get("name", "") for d in directors)
        elif isinstance(directors, dict):
            details["directors"] = directors.get("name", "")

        # actors is always a list of dicts
        actors = ld.get("actor", [])
        if isinstance(actors, list):
            details["actors"] = "|".join(a.get("name", "") for a in actors)

        # aggregate rating (fallback — scorecard JSON is more detailed)
        agg = ld.get("aggregateRating", {})
        if not details.get("tomatometer"):
            details["tomatometer"] = str(agg.get("ratingValue", ""))
        if not details.get("tomatometer_count"):
            details["tomatometer_count"] = str(agg.get("ratingCount", ""))

    @staticmethod
    def _parse_scorecard_json(soup: bs4.BeautifulSoup, details: dict) -> None:
        """Extract detailed scoring data from the media-scorecard-json script tag.

        This is the richest data source — it has critic, audience, and top-critic
        scores along with liked/not-liked counts and sentiment.
        """
        tag = soup.find("script", id="media-scorecard-json")
        if not tag or not tag.string:
            return
        try:
            data = json.loads(tag.string.strip())
        except (json.JSONDecodeError, TypeError):
            return

        # --- critic scores ---
        critics = data.get("criticsScore", {})
        details["tomatometer"] = str(critics.get("score", ""))
        details["tomatometer_count"] = str(critics.get("ratingCount", critics.get("reviewCount", "")))
        details["critic_avg_rating"] = str(critics.get("averageRating", ""))
        details["critic_sentiment"] = critics.get("sentiment", "")
        details["critic_certified"] = str(critics.get("certified", ""))
        details["critic_liked"] = str(critics.get("likedCount", ""))
        details["critic_not_liked"] = str(critics.get("notLikedCount", ""))

        # --- top critic scores (from overlay section) ---
        overlay = data.get("overlay", {})
        top = overlay.get("criticsTop", {})
        details["top_critic_score"] = str(top.get("score", ""))
        details["top_critic_count"] = str(top.get("ratingCount", top.get("reviewCount", "")))
        details["top_critic_liked"] = str(top.get("likedCount", ""))
        details["top_critic_not_liked"] = str(top.get("notLikedCount", ""))
        details["top_critic_sentiment"] = top.get("sentiment", "")
        details["top_critic_certified"] = str(top.get("certified", ""))

        # --- audience scores ---
        audience = data.get("audienceScore", {})
        details["audience_score"] = str(audience.get("score", ""))
        details["audience_avg_rating"] = str(audience.get("averageRating", ""))
        details["audience_review_count"] = str(audience.get("reviewCount", ""))
        details["audience_sentiment"] = audience.get("sentiment", "")
        details["audience_certified"] = str(audience.get("certified", ""))
        details["audience_liked"] = str(audience.get("likedCount", ""))
        details["audience_not_liked"] = str(audience.get("notLikedCount", ""))
        details["audience_rating_band"] = audience.get("bandedRatingCount", "")
        details["audience_score_type"] = audience.get("scoreType", "")

        # --- audience "all" (including unverified reviews) ---
        aud_all = overlay.get("audienceAll", {})
        details["audience_all_score"] = str(aud_all.get("score", ""))
        details["audience_all_avg_rating"] = str(aud_all.get("averageRating", ""))
        details["audience_all_review_count"] = str(aud_all.get("reviewCount", ""))
        details["audience_all_liked"] = str(aud_all.get("likedCount", ""))
        details["audience_all_not_liked"] = str(aud_all.get("notLikedCount", ""))
        details["audience_all_sentiment"] = aud_all.get("sentiment", "")

        if not details.get("synopsis"):
            details["synopsis"] = data.get("description", "")

    @staticmethod
    def _parse_movie_info_section(soup: bs4.BeautifulSoup, details: dict) -> None:
        """Extract runtime, box office, distributor, etc. from the info section."""
        section = soup.select_one('[data-qa="section:media-info"]')
        if not section:
            return

        # synopsis (fallback if not found in scorecard JSON)
        synopsis_el = section.select_one('[data-qa="synopsis-value"]')
        if synopsis_el and not details.get("synopsis"):
            details["synopsis"] = synopsis_el.get_text(strip=True)

        # map human-readable labels to our field names
        label_map = {
            "runtime": "runtime",
            "box office (gross usa)": "box_office_usa",
            "original language": "original_language",
            "release date (theaters)": "release_date_theaters",
            "release date (streaming)": "release_date_streaming",
            "distributor": "distributor",
            "production co": "production_co",
        }
        for item in section.select('[data-qa="item"]'):
            label_el = item.select_one('[data-qa="item-label"]')
            if not label_el:
                continue
            field_name = label_map.get(label_el.get_text(strip=True).lower())
            if not field_name:
                continue  # label doesn't match any field we care about
            value_els = item.select('[data-qa="item-value"]')
            details[field_name] = ", ".join(v.get_text(strip=True) for v in value_els)

    @staticmethod
    def _parse_consensus(soup: bs4.BeautifulSoup, details: dict) -> None:
        """Extract the one-line critic and audience consensus text."""
        for div_id, key in [
            ("#critics-consensus", "critics_consensus"),
            ("#audience-consensus", "audience_consensus"),
        ]:
            div = soup.select_one(div_id)
            if div:
                p = div.find("p")  # consensus text is in a <p> inside the div
                if p:
                    details[key] = p.get_text(strip=True)


class ReviewPageParser:
    """Parses critic and audience reviews from RT review pages."""

    @staticmethod
    def parse_critic_reviews(html: str, max_reviews: int = 10) -> list[CriticReview]:
        """Extract critic reviews from /reviews page. Reviews are in <review-card> elements."""
        soup = bs4.BeautifulSoup(html, "lxml")
        reviews: list[CriticReview] = []
        for card in soup.select("review-card"):
            if len(reviews) >= max_reviews:
                break
            name_el = card.select_one('[slot="name"]')       # critic's name
            pub_el = card.select_one('[slot="publication"]')  # publication name
            review_el = card.select_one('[slot="review"]')    # review text
            sentiment_el = card.select_one("score-icon-critics")  # fresh/rotten icon
            rating_text = ""
            rating_el = card.select_one('[slot="rating"]')
            if rating_el:
                for span in rating_el.select("span"):
                    t = span.get_text(strip=True)
                    if t:
                        rating_text = t  # e.g. "B+", "3.5/4", "8/10"
                        break
            reviews.append(CriticReview(
                critic_name=name_el.get_text(strip=True) if name_el else "",
                publication=pub_el.get_text(strip=True) if pub_el else "",
                sentiment=sentiment_el.get("sentiment", "") if sentiment_el else "",
                rating=rating_text,
                review_text=review_el.get_text(strip=True) if review_el else "",
            ))
        return reviews

    @staticmethod
    def parse_audience_reviews(html: str, max_reviews: int = 10) -> list[AudienceReview]:
        """Extract audience reviews from /reviews?type=user page."""
        soup = bs4.BeautifulSoup(html, "lxml")
        reviews: list[AudienceReview] = []
        for card in soup.select("review-card"):
            if len(reviews) >= max_reviews:
                break
            name_el = card.select_one('[slot="name"]')
            rating_el = card.select_one("rating-stars-group")  # star rating component
            content_el = card.select_one('[slot="content"]')   # review text
            reviews.append(AudienceReview(
                user_name=name_el.get_text(strip=True) if name_el else "",
                score=rating_el.get("score", "") if rating_el else "",  # e.g. "4.5"
                review_text=content_el.get_text(strip=True) if content_el else "",
                verified=card.get("verified-review", "false"),  # "true"/"false"
            ))
        return reviews


rt_logger = setup_logger("rt-scraper", "rt-scraper.log")


class RTBrowseCollector(BaseDiscoverer):
    """Scrolls through RT browse pages using Playwright to collect movie URLs."""

    def __init__(self, config: RottenTomatoesConfig):
        self._config = config
        self._parser = BrowsePageParser()

    def discover(self) -> list[MovieListing]:
        """Standalone discovery (not interweaved) — returns a flat list."""
        seen: dict[str, MovieListing] = {}
        self._browse_all(seen)
        return list(seen.values())

    def collect_into_queue(
        self,
        queue: asyncio.Queue,                    # where to push discovered MovieListings
        loop: asyncio.AbstractEventLoop,         # the async event loop (for thread-safe puts)
    ) -> dict[str, MovieListing]:
        """Crawl browse pages and feed new listings into the async queue.

        Runs on a background thread. Uses loop.call_soon_threadsafe() to
        safely push items from this thread into the asyncio queue.
        Returns the full seen dict for caching.
        """
        seen: dict[str, MovieListing] = {}  # keyed by lowercase title for dedup
        total_urls = len(self._config.category_urls)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, timeout=30000)  # headless Chrome
            context = browser.new_context(user_agent=self._config.user_agent)

            rt_logger.info("%d category URLs to crawl (target: %d movies)",
                           total_urls, self._config.target_movies)

            for idx, cat_url in enumerate(self._config.category_urls, 1):
                if len(seen) >= self._config.target_movies:
                    rt_logger.info("Reached target of %d unique movies", self._config.target_movies)
                    break

                label = cat_url.split("/browse/")[-1]  # short label for logging
                rt_logger.info("[browse %d/%d] %s", idx, total_urls, label)

                try:
                    new_listings = self._scroll_and_collect(context.new_page(), cat_url, seen)
                    # push each new listing into the async queue (thread-safe)
                    for listing in new_listings:
                        loop.call_soon_threadsafe(queue.put_nowait, listing)
                    if new_listings:
                        rt_logger.info("  +%d new (total: %d)", len(new_listings), len(seen))
                except Exception as e:
                    rt_logger.warning("Browse error on %s: %s", label, e)

                # save incremental cache every 50 URLs in case of crash
                if idx % 50 == 0:
                    self._save_incremental_cache(seen)

            browser.close()

        # signal "no more movies" to Phase 2
        loop.call_soon_threadsafe(queue.put_nowait, None)
        self._save_incremental_cache(seen)  # final cache save
        rt_logger.info("Browse complete: %d unique movies collected", len(seen))
        return seen

    def _browse_all(self, seen: dict[str, MovieListing]) -> None:
        """Browse all category pages without a queue (for standalone mode)."""
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent=self._config.user_agent)
            for cat_url in self._config.category_urls:
                if len(seen) >= self._config.target_movies:
                    break
                try:
                    self._scroll_and_collect(context.new_page(), cat_url, seen)
                except Exception as e:
                    rt_logger.warning("Browse error: %s", e)
            browser.close()

    def _save_incremental_cache(self, seen: dict[str, MovieListing]) -> None:
        """Save current listings to a partial cache file during collection."""
        cache_path = os.path.join(self._config.output_dir, self._config.listings_cache)
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            data = {key: asdict(val) for key, val in seen.items()}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            rt_logger.warning("Failed to save incremental cache: %s", e)

    def _dismiss_overlays(self, page: Page) -> None:
        """Dismiss cookie consent banners and app modals that block clicks.

        RT uses OneTrust for cookie consent and custom overlay-base modals.
        If we don't dismiss these, the "Load More" button click fails.
        """
        # try OneTrust cookie banner buttons
        for selector in (
            "#onetrust-accept-btn-handler",
            "#onetrust-reject-all-handler",
            "#onetrust-close-btn-container button",
        ):
            try:
                btn = page.query_selector(selector)
                if btn and btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(500)
                    return
            except Exception:
                pass

        # try custom RT overlay modals
        for selector in (
            'overlay-base[data-rtappmanager="overlayBase:close"]',
            "overlay-base",
        ):
            try:
                overlay = page.query_selector(selector)
                if overlay and overlay.is_visible():
                    close_btn = page.query_selector(f'{selector} [aria-label="Close"]')
                    if close_btn and close_btn.is_visible():
                        close_btn.click()
                        page.wait_for_timeout(500)
                        return
            except Exception:
                pass

        # last resort: press Escape to dismiss any overlay
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        except Exception:
            pass

    def _scroll_and_collect(
        self, page: Page, url: str, seen: dict[str, MovieListing],
    ) -> list[MovieListing]:
        """Scroll a single browse page, clicking "Load More" to reveal movies.

        Returns a list of NEW (previously unseen) MovieListings.
        Also adds them to the `seen` dict for cross-page deduplication.
        """
        page.goto(url, wait_until="domcontentloaded")  # navigate to the page
        page.wait_for_timeout(3000)  # wait for initial JS rendering
        self._dismiss_overlays(page)  # clear any blocking overlays

        new_listings: list[MovieListing] = []
        for scroll_num in range(self._config.max_scroll_per_category):
            if len(seen) >= self._config.target_movies:
                break  # reached global target

            html = page.content()  # get the current DOM state
            listings = self._parser.parse(html)  # extract movie tiles

            new_this_scroll = 0
            for m in listings:
                key = m.title.lower().strip()  # normalize for dedup
                if key and key not in seen:
                    seen[key] = m
                    new_listings.append(m)
                    new_this_scroll += 1

            # adaptive skipping: if the first scroll yields < 5% new, bail
            # (this page is mostly duplicates of movies we already have)
            if scroll_num == 0 and len(listings) > 10:
                ratio = new_this_scroll / len(listings) if listings else 0
                if ratio < self._config.min_new_ratio:
                    rt_logger.debug(
                        "Skipping low-yield page (%d/%d new): %s",
                        new_this_scroll, len(listings), url,
                    )
                    break

            # find and click the "Load More" button
            btn = page.query_selector('[data-qa="dlp-load-more-button"]')
            if not btn or not btn.is_visible():
                break  # no more content to load

            # count tiles before click so we can wait for new ones to appear
            tiles_before = len(page.query_selector_all('[data-qa="discovery-media-list-item"]'))

            try:
                self._dismiss_overlays(page)  # clear overlays before clicking
                btn.scroll_into_view_if_needed()
                btn.click(timeout=10000)
            except Exception:
                # fallback: use JavaScript to click (works even if the button is obscured)
                try:
                    page.evaluate(
                        'document.querySelector(\'[data-qa="dlp-load-more-button"]\')?.click()'
                    )
                except Exception:
                    break

            # adaptive wait: wait for new tiles to appear instead of a fixed sleep
            try:
                page.wait_for_function(
                    f'document.querySelectorAll(\'[data-qa="discovery-media-list-item"]\').length > {tiles_before}',
                    timeout=5000,
                )
            except Exception:
                pass  # timeout is fine — maybe no more content

            page.wait_for_timeout(int(self._config.scroll_pause * 1000))  # short settle time

        page.close()  # close the tab to free resources
        return new_listings


class RTDetailScraper(BaseDetailScraper):
    """Scrapes individual movie pages + review pages using asyncio + aiohttp."""

    def __init__(self, config: RottenTomatoesConfig, client: AsyncHttpClient):
        self._config = config
        self._client = client
        self._movie_parser = MoviePageParser()
        self._review_parser = ReviewPageParser()

    async def scrape_from_queue(
        self,
        queue: asyncio.Queue,     # MovieListing objects arrive here from the browse collector
        *writers: CsvWriter,      # [0]=movies, [1]=critic reviews, [2]=audience reviews
    ) -> tuple[int, ...]:
        """Consume movie listings, scrape each, write to three CSV files."""
        movie_writer, critic_writer, audience_writer = writers[0], writers[1], writers[2]
        flush_every = self._config.flush_every

        already_scraped = movie_writer.get_existing_values("url")
        if already_scraped:
            rt_logger.info("Crash-resume: found %d already-scraped movies, will skip them", len(already_scraped))

        movie_buf: list[MovieDetail] = []
        critic_buf: list[CriticReview] = []
        audience_buf: list[AudienceReview] = []
        movies_done = 0
        movies_failed = 0
        total_critics = 0
        total_audience = 0

        start_time = time.monotonic()
        pending: set[asyncio.Task] = set()  # in-flight scrape tasks
        browse_done = False  # set to True when we receive the None sentinel

        while True:
            # --- fill pending tasks up to the concurrency limit ---
            while len(pending) < self._config.detail_concurrency and not browse_done:
                try:
                    listing = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if listing is None:
                    browse_done = True  # browse collector is done
                    rt_logger.info("Browse phase complete, draining %d remaining", len(pending))
                    break

                if not listing.url or not listing.url.startswith("https://www.rottentomatoes.com/m/"):
                    continue  # skip non-movie URLs

                if listing.url in already_scraped:
                    rt_logger.debug("Skipping already-scraped: %s", listing.title)
                    continue

                task = asyncio.create_task(self._scrape_one(listing))
                pending.add(task)

            if not pending and browse_done:
                break  # all work done
            if not pending:
                await asyncio.sleep(0.1)
                continue

            # --- wait for at least one task to finish ---
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    detail, critics, audience, title = task.result()
                    movie_buf.append(detail)
                    critic_buf.extend(critics)
                    audience_buf.extend(audience)
                    movies_done += 1
                    total_critics += len(critics)
                    total_audience += len(audience)
                    rt_logger.info("[%d done] ✓ %s", movies_done, title)
                except Exception as e:
                    movies_failed += 1
                    rt_logger.error("[%d failed] ✗ %s: %s", movies_failed, type(e).__name__, e)

            # --- flush to CSV periodically ---
            if len(movie_buf) >= flush_every:
                movie_writer.append(movie_buf)
                critic_writer.append(critic_buf)
                audience_writer.append(audience_buf)
                movie_buf.clear()
                critic_buf.clear()
                audience_buf.clear()
                elapsed = time.monotonic() - start_time
                rate = movies_done / elapsed if elapsed > 0 else 0
                rt_logger.info("Flushed to CSV | %d scraped, %d failed (%.1f/sec)",
                               movies_done, movies_failed, rate)

        # flush remaining
        if movie_buf:
            movie_writer.append(movie_buf)
            critic_writer.append(critic_buf)
            audience_writer.append(audience_buf)

        return movies_done, total_critics, total_audience

    async def _scrape_one(
        self, listing: MovieListing
    ) -> tuple[MovieDetail, list[CriticReview], list[AudienceReview], str]:
        """Scrape one movie: metadata page + critic reviews + audience reviews."""
        url = listing.url

        # 1. movie metadata (JSON-LD + scorecard + info section + consensus)
        html = await self._client.get(url)
        parsed = self._movie_parser.parse(html)
        detail = MovieDetail(
            title=listing.title, url=url,
            critic_score_browse=listing.critic_score_browse,
            audience_score_browse=listing.audience_score_browse,
            **parsed,
        )

        # 2. critic reviews (separate page: /m/{slug}/reviews)
        critic_reviews: list[CriticReview] = []
        try:
            reviews_html = await self._client.get(url.rstrip("/") + "/reviews")
            critic_reviews = self._review_parser.parse_critic_reviews(
                reviews_html, self._config.max_critic_reviews
            )
            for r in critic_reviews:
                r.movie_title = listing.title  # attach movie context to each review
                r.movie_url = url
        except Exception:
            pass  # review page failed — not critical, we still have the metadata

        # 3. audience reviews (separate page: /m/{slug}/reviews?type=user)
        audience_reviews: list[AudienceReview] = []
        try:
            audience_html = await self._client.get(url.rstrip("/") + "/reviews?type=user")
            audience_reviews = self._review_parser.parse_audience_reviews(
                audience_html, self._config.max_audience_reviews
            )
            for r in audience_reviews:
                r.movie_title = listing.title
                r.movie_url = url
        except Exception:
            pass

        return detail, critic_reviews, audience_reviews, listing.title


class RottenTomatoesPipeline(BasePipeline):
    """Top-level orchestrator for the Rotten Tomatoes scraping pipeline."""

    def __init__(self, config: RottenTomatoesConfig | None = None):
        self._config = config or RottenTomatoesConfig()

    def run(self) -> None:
        cache_path = os.path.join(self._config.output_dir, self._config.listings_cache)
        cached = self._load_cache(cache_path)

        if cached:
            rt_logger.info("Loaded %d movie URLs from cache: %s", len(cached), cache_path)
            asyncio.run(self._run_from_cache(cached))
        else:
            rt_logger.info("No cache found — running interweaved browse + scrape")
            asyncio.run(self._run_interweaved())

    async def _run_interweaved(self) -> None:
        """Run browse (thread) + detail scraping (async) simultaneously."""
        queue: asyncio.Queue[MovieListing | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # open all three CSV writers
        out = self._config.output_dir
        movie_writer = CsvWriter(os.path.join(out, self._config.movies_csv), MovieDetail)
        critic_writer = CsvWriter(os.path.join(out, self._config.critic_reviews_csv), CriticReview)
        audience_writer = CsvWriter(os.path.join(out, self._config.audience_reviews_csv), AudienceReview)
        movie_writer.open()
        critic_writer.open()
        audience_writer.open()

        # create the async HTTP client for Phase 2
        detail_client = AsyncHttpClient(
            self._config.user_agents, self._config.request_delay,
            self._config.max_retries, self._config.retry_backoff,
            self._config.default_rate_limit_wait,
        )
        await detail_client.start(self._config.detail_concurrency)
        detail_scraper = RTDetailScraper(self._config, detail_client)

        # run the Playwright browse collector in a background thread
        collector = RTBrowseCollector(self._config)
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        browse_future = loop.run_in_executor(
            executor, collector.collect_into_queue, queue, loop
        )

        try:
            # consume from queue and scrape details concurrently
            movies_n, critics_n, audience_n = await detail_scraper.scrape_from_queue(
                queue, movie_writer, critic_writer, audience_writer
            )
        finally:
            movie_writer.close()
            critic_writer.close()
            audience_writer.close()
            await detail_client.close()

        await browse_future  # wait for browse thread to finish
        executor.shutdown(wait=False)
        rt_logger.info("Done! %d movies, %d critic reviews, %d audience reviews",
                       movies_n, critics_n, audience_n)

    async def _run_from_cache(self, listings: dict[str, MovieListing]) -> None:
        """Run Phase 2 only, using cached listings."""
        valid = {k: v for k, v in listings.items()
                 if v.url and v.url.startswith("https://www.rottentomatoes.com/m/")}
        if len(valid) < len(listings):
            rt_logger.warning("Filtered %d invalid URLs from cache (%d → %d)",
                              len(listings) - len(valid), len(listings), len(valid))

        queue: asyncio.Queue[MovieListing | None] = asyncio.Queue()
        for listing in valid.values():
            queue.put_nowait(listing)
        queue.put_nowait(None)  # sentinel

        out = self._config.output_dir
        movie_writer = CsvWriter(os.path.join(out, self._config.movies_csv), MovieDetail)
        critic_writer = CsvWriter(os.path.join(out, self._config.critic_reviews_csv), CriticReview)
        audience_writer = CsvWriter(os.path.join(out, self._config.audience_reviews_csv), AudienceReview)
        movie_writer.open()
        critic_writer.open()
        audience_writer.open()

        detail_client = AsyncHttpClient(
            self._config.user_agents, self._config.request_delay,
            self._config.max_retries, self._config.retry_backoff,
            self._config.default_rate_limit_wait,
        )
        await detail_client.start(self._config.detail_concurrency)

        try:
            detail_scraper = RTDetailScraper(self._config, detail_client)
            movies_n, critics_n, audience_n = await detail_scraper.scrape_from_queue(
                queue, movie_writer, critic_writer, audience_writer
            )
        finally:
            movie_writer.close()
            critic_writer.close()
            audience_writer.close()
            await detail_client.close()

        rt_logger.info("Done! %d movies, %d critic reviews, %d audience reviews",
                       movies_n, critics_n, audience_n)

    @staticmethod
    def _load_cache(path: str) -> dict[str, MovieListing] | None:
        """Load cached movie listings from JSON. Returns None if no cache."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {key: MovieListing(**val) for key, val in data.items()}
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            rt_logger.warning("Cache file corrupt, ignoring: %s", e)
            return None
