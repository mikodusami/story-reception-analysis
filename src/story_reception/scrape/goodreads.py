"""Goodreads pipeline: Listopia list discovery (sync), list pages (async), then book pages and reviews
from the embedded __NEXT_DATA__ Apollo cache (async). Writes goodreads-books.csv and goodreads-reviews.csv.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, fields

import bs4

from .base import (
    AsyncHttpClient,
    BaseDetailScraper,
    BaseDiscoverer,
    BasePipeline,
    BaseScraperConfig,
    CsvWriter,
    SyncHttpClient,
    load_cache,
    logger,
    save_cache,
    setup_logger,
)


@dataclass
class BookListing:
    """Lightweight record from a Goodreads list page.

    Contains just enough info to queue the book for detailed scraping.
    The list-level rating fields let us compare list ratings vs actual book page ratings.
    """
    title: str                   # book title as shown on the list
    url: str                     # full URL to the book page (e.g. https://www.goodreads.com/book/show/12345)
    book_id: str = ""            # numeric Goodreads book ID extracted from the URL
    avg_rating_list: str = ""    # average rating shown on the list (may differ from book page)
    num_ratings_list: str = ""   # number of ratings shown on the list
    list_rank: str = ""          # position on the list (e.g. "1", "2", "3")
    source_list: str = ""        # which list this book came from (for traceability)


@dataclass
class BookDetail:
    """Full metadata scraped from an individual Goodreads book page.

    All fields are strings because some may be empty/missing, and we want
    consistent CSV output without mixed types.
    """
    title: str = ""
    title_complete: str = ""     # includes subtitle if any
    url: str = ""
    book_id: str = ""
    legacy_id: str = ""          # older Goodreads ID format
    description: str = ""        # HTML-stripped book description
    image_url: str = ""          # cover image URL
    author: str = ""
    author_url: str = ""
    avg_rating: str = ""         # from the book page (more accurate than list page)
    ratings_count: str = ""
    reviews_count: str = ""
    rating_dist_1: str = ""      # count of 1-star ratings
    rating_dist_2: str = ""      # count of 2-star ratings
    rating_dist_3: str = ""      # count of 3-star ratings
    rating_dist_4: str = ""      # count of 4-star ratings
    rating_dist_5: str = ""      # count of 5-star ratings
    num_pages: str = ""
    book_format: str = ""        # e.g. "Hardcover", "Paperback", "Kindle Edition"
    publisher: str = ""
    publication_date: str = ""   # YYYY-MM-DD format
    isbn: str = ""
    isbn13: str = ""
    asin: str = ""               # Amazon Standard Identification Number
    language: str = ""
    genres: str = ""             # pipe-separated, e.g. "Fiction|Fantasy|Adventure"
    series_name: str = ""
    series_position: str = ""    # e.g. "1", "2.5"
    content_rating: str = ""
    awards: str = ""             # pipe-separated award names
    source_list: str = ""        # carried over from BookListing
    list_rank: str = ""          # carried over from BookListing


@dataclass
class Review:
    """A single user review scraped from a Goodreads book page.

    Reviews are embedded in the same __NEXT_DATA__ JSON as the book metadata,
    so we get them for free — no extra HTTP request needed.
    """
    book_title: str = ""         # which book this review belongs to
    book_url: str = ""
    reviewer_name: str = ""
    reviewer_id: str = ""
    rating: str = ""             # 1-5 star rating
    review_text: str = ""        # HTML-stripped review body
    created_at: str = ""         # YYYY-MM-DD when review was posted
    updated_at: str = ""         # YYYY-MM-DD when review was last edited
    like_count: str = ""         # how many people found this review helpful
    comment_count: str = ""
    shelves: str = ""            # pipe-separated shelf names (e.g. "read|fiction|favorites")


# these genre tags drive Phase 1a — we'll crawl /list/tag/{genre} for each one
GENRE_TAGS = [
    "fiction", "non-fiction", "fantasy", "science-fiction", "mystery",
    "romance", "thriller", "horror", "historical-fiction", "biography",
    "history", "classics", "young-adult", "children", "poetry",
    "philosophy", "psychology", "science", "self-help", "business",
    "art", "music", "travel", "crime", "adventure",
    "humor", "memoir", "graphic-novels", "comics", "drama",
]


@dataclass(frozen=True)
class GoodreadsConfig(BaseScraperConfig):
    """Goodreads-specific settings that extend the shared base config."""
    genre_tags: tuple[str, ...] = tuple(GENRE_TAGS)  # genres to discover lists from
    list_index_pages_per_genre: int = 3   # how many pages of /list/tag/{genre} to crawl (~20 lists per page)
    max_books_per_list: int = 1000          # cap books scraped from any single list
    max_list_pages: int = 5              # how deep to paginate into each list

    target_books: int = 10_000              # stop discovering once we have this many unique books
    max_reviews_per_book: int = 30        # reviews embedded in __NEXT_DATA__ (usually 30 max)

    # separate concurrency for list scraping vs book detail scraping
    list_concurrency: int = 4             # Phase 1b: lighter, fewer concurrent requests
    detail_concurrency: int = 6           # Phase 2: heavier, but 10+ triggers Cloudflare

    books_csv: str = "goodreads-books.csv"
    reviews_csv: str = "goodreads-reviews.csv"
    listings_cache: str = "goodreads-listings-cache.json"  # save discovered books for re-runs


def build_list_index_urls(
    genres: list[str] | None = None,
    max_pages_per_genre: int = 3,
) -> list[str]:
    """Build URLs for genre-tagged list index pages.

    Example output: https://www.goodreads.com/list/tag/fiction?page=1
    Each page shows ~20 curated lists. We grab a few pages per genre
    to get a diverse set of lists spanning different topics and quality levels.
    """
    base = "https://www.goodreads.com/list/tag"
    tags = genres or GENRE_TAGS
    urls: list[str] = []
    for tag in tags:
        for page in range(1, max_pages_per_genre + 1):
            urls.append(f"{base}/{tag}?page={page}")
    return urls


class ListIndexPageParser:
    """Parses a genre-tagged list index page (/list/tag/{genre}).

    These pages contain links like <a class="listTitle" href="/list/show/1.Best_Books_Ever">
    We extract the href to get Listopia list URLs for Phase 1b.
    """

    LIST_URL_RE = re.compile(r"^/list/show/(\d+)")  # matches /list/show/1234

    @staticmethod
    def parse(html: str) -> list[str]:
        """Extract all unique Listopia list URLs from the page."""
        soup = bs4.BeautifulSoup(html, "lxml")  # lxml is 2-5× faster than html.parser
        urls: list[str] = []
        seen: set[str] = set()  # track hrefs to avoid duplicates within the same page
        for a in soup.select("a.listTitle"):  # Goodreads wraps list names in <a class="listTitle">
            href = a.get("href", "")
            m = ListIndexPageParser.LIST_URL_RE.match(href)
            if not m:
                continue  # skip non-list links
            if href in seen:
                continue  # already found this list on this page
            seen.add(href)
            urls.append("https://www.goodreads.com" + href)  # convert relative → absolute URL
        logger.debug("ListIndexPageParser: extracted %d unique list URLs", len(urls))
        return urls


class ListPageParser:
    """Parses book listings from a single Listopia list page (/list/show/{id}).

    Each book row is a <tr itemtype="http://schema.org/Book"> containing the book
    title, URL, list rank, and a minirating string like "4.35 avg rating — 10,058,107 ratings".
    """

    BOOK_URL_RE = re.compile(r"^/book/show/(\d+)")  # matches /book/show/12345
    # parses "4.35 avg rating — 10,058,107 ratings" into (4.35, 10058107)
    RATING_RE = re.compile(r"([\d.]+)\s*avg rating\s*[—\-]\s*([\d,]+)\s*ratings?")

    @staticmethod
    def parse(html: str, source_list: str = "") -> list[BookListing]:
        """Extract book listings from the list page HTML."""
        soup = bs4.BeautifulSoup(html, "lxml")
        listings: list[BookListing] = []

        # each book is in a schema.org-typed table row
        rows = soup.select('tr[itemtype="http://schema.org/Book"]')
        for row in rows:
            title_link = row.select_one("a.bookTitle")  # <a class="bookTitle" href="/book/show/...">
            if not title_link:
                continue
            href = title_link.get("href", "")
            m = ListPageParser.BOOK_URL_RE.match(href)
            if not m:
                continue
            book_id = m.group(1)  # extract the numeric book ID from the URL

            # title text is inside a nested <span itemprop="name">
            name_el = title_link.select_one('span[itemprop="name"]')
            title = name_el.get_text(strip=True) if name_el else title_link.get_text(strip=True)

            # list rank from <td class="number"> (e.g. "1", "2")
            rank_el = row.select_one("td.number")
            rank = rank_el.get_text(strip=True) if rank_el else ""

            # extract avg rating + count from <span class="minirating">
            avg_rating = ""
            num_ratings = ""
            mini = row.select_one("span.minirating")
            if mini:
                rm = ListPageParser.RATING_RE.search(mini.get_text(" ", strip=True))
                if rm:
                    avg_rating = rm.group(1)                    # e.g. "4.35"
                    num_ratings = rm.group(2).replace(",", "")  # e.g. "10058107" (remove commas)

            listings.append(BookListing(
                title=title,
                url="https://www.goodreads.com" + href,
                book_id=book_id,
                avg_rating_list=avg_rating,
                num_ratings_list=num_ratings,
                list_rank=rank,
                source_list=source_list,
            ))

        return listings

    @staticmethod
    def has_next_page(html: str, current_page: int) -> bool:
        """Check if there's a page N+1 to fetch.

        Goodreads list pagination uses a <div class="pagination"> block
        with numbered <a> elements. If we find a link for current_page+1, there's more.
        """
        soup = bs4.BeautifulSoup(html, "lxml")
        pagination = soup.select_one("div.pagination")
        if not pagination:
            return False  # no pagination element = single-page list
        for a in pagination.select("a"):
            txt = a.get_text(strip=True)
            if txt.isdigit() and int(txt) == current_page + 1:
                return True  # found a link to the next page
        return False


class BookPageParser:
    """Extracts BookDetail + Reviews from a Goodreads book page.

    The page embeds an Apollo GraphQL cache in <script id="__NEXT_DATA__">
    containing:
      - Book:<id>        → title, description, ISBN, format, genres, series info
      - Work:<id>        → aggregate stats (avg rating, distribution, awards)
      - Contributor:<id> → author name and URL
      - Review:<id>      → up to 30 reviews with text, rating, shelves

    We parse the whole cache once and resolve internal references manually
    (Apollo uses {"__ref": "Key:..."} for cross-references).
    """

    # regex to find the __NEXT_DATA__ JSON blob in the HTML
    NEXT_DATA_RE = re.compile(
        r'<script id="__NEXT_DATA__" type="application/json">\s*'
        r"(\{.*?\})\s*</script>",
        re.DOTALL,  # allow . to match newlines since the JSON spans multiple lines
    )

    @staticmethod
    def parse(html: str, max_reviews: int = 30) -> tuple[dict, list[dict]]:
        """Parse a book page and return (book_detail_dict, list_of_review_dicts)."""
        apollo = BookPageParser._extract_apollo(html)  # get the GraphQL cache dict
        if not apollo:
            logger.warning("No __NEXT_DATA__ payload found")
            return BookPageParser._fallback_from_ld(html), []  # try JSON-LD as fallback

        book = BookPageParser._find_first(apollo, "Book:")   # find the Book object in the cache
        work = BookPageParser._find_first(apollo, "Work:")   # find the Work object (has stats)
        if not book:
            logger.warning("No Book object in Apollo cache, falling back to JSON-LD")
            return BookPageParser._fallback_from_ld(html), []

        detail = BookPageParser._build_detail(book, work, apollo)   # extract metadata
        reviews = BookPageParser._build_reviews(apollo, max_reviews)  # extract reviews
        return detail, reviews

    # ---------- extraction helpers ----------

    @staticmethod
    def _extract_apollo(html: str) -> dict | None:
        """Pull the Apollo GraphQL cache dict from __NEXT_DATA__."""
        m = BookPageParser.NEXT_DATA_RE.search(html)
        if not m:
            return None
        try:
            data = json.loads(m.group(1))  # parse the entire __NEXT_DATA__ JSON
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse __NEXT_DATA__: %s", e)
            return None
        try:
            # the Apollo cache lives at props.pageProps.apolloState
            return data["props"]["pageProps"]["apolloState"]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _find_first(apollo: dict, prefix: str) -> dict | None:
        """Find the first key in the Apollo cache starting with the given prefix.

        Keys look like 'Book:kca://book/amzn1.gr.book.v1.abc123' — we match by prefix.
        """
        for k, v in apollo.items():
            if k.startswith(prefix):
                return v
        return None

    @staticmethod
    def _resolve(apollo: dict, maybe_ref: dict | None) -> dict | None:
        """Resolve an Apollo cross-reference.

        Apollo uses {"__ref": "Contributor:kca://..."} to link between objects.
        This function follows the reference and returns the actual object.
        """
        if not maybe_ref:
            return None
        if isinstance(maybe_ref, dict) and "__ref" in maybe_ref:
            return apollo.get(maybe_ref["__ref"])  # follow the reference
        return maybe_ref  # already a concrete object, not a reference

    # ---------- book detail ----------

    @staticmethod
    def _build_detail(book: dict, work: dict | None, apollo: dict) -> dict:
        """Build a flat detail dict from the Book + Work Apollo objects."""
        d: dict = {
            "title": book.get("title", "") or "",
            "title_complete": book.get("titleComplete", "") or "",
            "url": book.get("webUrl", "") or "",
            "book_id": book.get("id", "") or "",
            "legacy_id": str(book.get("legacyId", "")),
            "description": BookPageParser._clean_html(book.get("description", "") or ""),
            "image_url": book.get("imageUrl", "") or "",
        }

        # --- author: resolve the primaryContributorEdge reference ---
        pc_edge = book.get("primaryContributorEdge") or {}
        author = BookPageParser._resolve(apollo, pc_edge.get("node"))
        if author:
            d["author"] = author.get("name", "") or ""
            d["author_url"] = author.get("webUrl", "") or ""

        # --- book details: pages, format, publisher, ISBN, etc. ---
        details = book.get("details") or {}
        d["num_pages"] = str(details.get("numPages") or "")
        d["book_format"] = details.get("format", "") or ""
        d["publisher"] = details.get("publisher", "") or ""
        d["publication_date"] = BookPageParser._ms_to_date(details.get("publicationTime"))
        d["isbn"] = details.get("isbn", "") or ""
        d["isbn13"] = details.get("isbn13", "") or ""
        d["asin"] = details.get("asin", "") or ""
        lang = details.get("language") or {}
        d["language"] = lang.get("name", "") if isinstance(lang, dict) else ""

        # --- genres: flatten the bookGenres array into a pipe-separated string ---
        genres = [
            (bg.get("genre") or {}).get("name", "")
            for bg in (book.get("bookGenres") or [])
            if (bg.get("genre") or {}).get("name")
        ]
        d["genres"] = "|".join(genres)  # e.g. "Fiction|Fantasy|Adventure"

        # --- series: resolve the series reference if the book is part of one ---
        series_list = book.get("bookSeries") or []
        if series_list:
            first = series_list[0]
            series_obj = BookPageParser._resolve(apollo, first.get("series"))
            if series_obj:
                d["series_name"] = series_obj.get("title", "") or ""
            d["series_position"] = first.get("userPosition", "") or ""

        # --- work-level stats: ratings, distribution, awards ---
        if work:
            stats = work.get("stats") or {}
            d["avg_rating"] = str(stats.get("averageRating") or "")
            d["ratings_count"] = str(stats.get("ratingsCount") or "")
            d["reviews_count"] = str(stats.get("textReviewsCount") or "")
            dist = stats.get("ratingsCountDist") or []  # array of 5 ints: [1-star, 2-star, ..., 5-star]
            if len(dist) == 5:
                d["rating_dist_1"] = str(dist[0])
                d["rating_dist_2"] = str(dist[1])
                d["rating_dist_3"] = str(dist[2])
                d["rating_dist_4"] = str(dist[3])
                d["rating_dist_5"] = str(dist[4])

            # awards are nested inside work.details.awardsWon
            wdetails = work.get("details") or {}
            awards_won = wdetails.get("awardsWon") or []
            award_strs = []
            for a in awards_won:
                name = a.get("name", "")
                desig = a.get("designation", "")  # e.g. "Winner", "Nominee"
                if name:
                    award_strs.append(f"{name} ({desig})" if desig else name)
            d["awards"] = "|".join(award_strs)

        return d

    @staticmethod
    def _ms_to_date(ms: int | None) -> str:
        """Convert a millisecond epoch timestamp to YYYY-MM-DD string."""
        if not ms:
            return ""
        try:
            ts = int(ms) / 1000  # milliseconds → seconds
            return time.strftime("%Y-%m-%d", time.gmtime(ts))  # UTC to avoid timezone drift
        except (ValueError, OSError, OverflowError):
            return ""  # some pre-1970 dates cause OverflowError on Windows

    @staticmethod
    def _clean_html(s: str) -> str:
        """Strip HTML tags from a string and collapse whitespace."""
        if not s:
            return ""
        text = bs4.BeautifulSoup(s, "lxml").get_text(" ", strip=True)  # strip tags, join with spaces
        return re.sub(r"\s+", " ", text).strip()  # collapse runs of whitespace to single spaces

    # ---------- reviews ----------

    @staticmethod
    def _build_reviews(apollo: dict, max_reviews: int) -> list[dict]:
        """Extract review data from Review:* objects in the Apollo cache."""
        reviews: list[dict] = []
        for k, v in apollo.items():
            if not k.startswith("Review:"):
                continue  # skip non-review objects
            if len(reviews) >= max_reviews:
                break  # respect the cap

            # resolve the reviewer (creator) reference
            creator = BookPageParser._resolve(apollo, v.get("creator"))

            # extract shelf/tag info (what shelves the reviewer put this book on)
            shelving = v.get("shelving") or {}
            taggings = shelving.get("taggings") or []
            shelves = [
                (t.get("tag") or {}).get("name", "")
                for t in taggings
                if (t.get("tag") or {}).get("name")
            ]
            # prepend the base shelf (e.g. "read", "currently-reading")
            base_shelf = (shelving.get("shelf") or {}).get("name", "")
            if base_shelf:
                shelves = [base_shelf] + shelves

            reviews.append({
                "reviewer_name": (creator or {}).get("name", "") or "",
                "reviewer_id": str((creator or {}).get("legacyId", "")),
                "rating": str(v.get("rating") or ""),
                "review_text": BookPageParser._clean_html(v.get("text", "") or ""),
                "created_at": BookPageParser._ms_to_date(v.get("createdAt")),
                "updated_at": BookPageParser._ms_to_date(v.get("updatedAt")),
                "like_count": str(v.get("likeCount") or ""),
                "comment_count": str(v.get("commentCount") or ""),
                "shelves": "|".join(shelves),  # pipe-separated: "read|fiction|favorites"
            })
        return reviews

    # ---------- JSON-LD fallback ----------

    @staticmethod
    def _fallback_from_ld(html: str) -> dict:
        """If __NEXT_DATA__ is missing/broken, extract what we can from JSON-LD.

        JSON-LD is a simpler structured data block (<script type="application/ld+json">)
        that has basic info like title, author, rating, ISBN. It's less complete than
        Apollo but better than nothing.
        """
        d: dict = {}
        soup = bs4.BeautifulSoup(html, "lxml")
        tag = soup.find("script", type="application/ld+json")
        if not tag or not tag.string:
            return d
        try:
            ld = json.loads(tag.string)
        except json.JSONDecodeError:
            return d
        d["title"] = ld.get("name", "") or ""
        d["image_url"] = ld.get("image", "") or ""
        d["book_format"] = ld.get("bookFormat", "") or ""
        d["num_pages"] = str(ld.get("numberOfPages") or "")
        d["language"] = ld.get("inLanguage", "") or ""
        d["isbn"] = ld.get("isbn", "") or ""
        authors = ld.get("author") or []
        if isinstance(authors, list) and authors:
            d["author"] = authors[0].get("name", "") or ""
            d["author_url"] = authors[0].get("url", "") or ""
        agg = ld.get("aggregateRating") or {}
        d["avg_rating"] = str(agg.get("ratingValue") or "")
        d["ratings_count"] = str(agg.get("ratingCount") or "")
        d["reviews_count"] = str(agg.get("reviewCount") or "")
        return d


# new logger
gr_logger = setup_logger("goodreads-scraper", "goodreads-scraper.log")


class GoodreadsListDiscoverer(BaseDiscoverer):
    """Phase 1a: discovers Listopia list URLs by crawling genre-tagged index pages."""

    def __init__(self, config: GoodreadsConfig, client: SyncHttpClient | None = None):
        self._config = config
        # use provided client or create a default one with the config's settings
        self._client = client or SyncHttpClient(
            config.user_agents, config.request_delay,
            config.max_retries, config.retry_backoff,
        )
        self._parser = ListIndexPageParser()

    def discover(self) -> list[str]:
        """Crawl genre index pages and return a deduplicated list of Listopia URLs."""
        # build all the genre index page URLs we need to crawl
        index_urls = build_list_index_urls(
            list(self._config.genre_tags),
            self._config.list_index_pages_per_genre,
        )
        gr_logger.info("Phase 1a: crawling %d genre index pages", len(index_urls))

        all_lists: list[str] = []   # ordered list of unique list URLs
        seen: set[str] = set()      # fast lookup for deduplication

        for i, url in enumerate(index_urls, 1):
            try:
                html = self._client.get(url)  # synchronous fetch
            except Exception as e:
                gr_logger.warning("[1a %d/%d] failed %s: %s", i, len(index_urls), url, e)
                continue  # skip failed pages, keep going
            found = self._parser.parse(html)  # extract list URLs from the page
            new = [u for u in found if u not in seen]  # filter out duplicates
            for u in new:
                seen.add(u)
                all_lists.append(u)
            gr_logger.info(
                "[1a %d/%d] %s — +%d lists (total: %d)",
                i, len(index_urls), url.split("/tag/")[-1], len(new), len(all_lists),
            )

        gr_logger.info("Phase 1a complete: %d unique lists discovered", len(all_lists))
        return all_lists


class GoodreadsListScraper:
    """Phase 1b: scrapes each Listopia list for its book rows.

    Pushes BookListing objects into a shared asyncio.Queue as they're found,
    so Phase 2 can start scraping book detail pages immediately.
    Uses a separate AsyncHttpClient from Phase 2 so list-page traffic
    doesn't steal concurrency from book-page traffic.
    """

    def __init__(self, config: GoodreadsConfig, client: AsyncHttpClient):
        self._config = config
        self._client = client
        self._parser = ListPageParser()

    async def scrape_into_queue(
        self,
        list_urls: list[str],             # Listopia URLs to scrape
        queue: asyncio.Queue,             # where to push discovered BookListings
        seen_book_ids: set[str],          # shared set for cross-list deduplication
    ) -> int:
        """Scrape all lists concurrently and push new BookListings into the queue.

        Returns the total number of NEW (unique) listings pushed.
        Puts a None sentinel into the queue when finished to signal Phase 2.
        """
        gr_logger.info(
            "Phase 1b: scraping %d Listopia lists (target: %d books)",
            len(list_urls), self._config.target_books,
        )

        sem = asyncio.Semaphore(self._config.list_concurrency)  # cap concurrent list scrapes
        total_new = 0        # running count of unique books found across all lists
        lists_done = 0       # how many lists we've finished processing
        lock = asyncio.Lock()  # protects seen_book_ids and total_new from race conditions

        async def process_list(list_url: str) -> int:
            """Scrape one Listopia list, paginating up to max_list_pages."""
            nonlocal total_new, lists_done
            async with sem:  # wait here if we're at max concurrency
                new_here = 0  # books found from THIS list
                source_list = list_url.rstrip("/").split("/")[-1]  # e.g. "1.Best_Books_Ever"

                for page in range(1, self._config.max_list_pages + 1):
                    # check if we've hit the global target
                    async with lock:
                        if len(seen_book_ids) >= self._config.target_books:
                            return new_here  # enough books, stop early

                    page_url = f"{list_url}?page={page}" if page > 1 else list_url
                    try:
                        html = await self._client.get(page_url)
                    except Exception as e:
                        gr_logger.warning("List page failed %s: %s", page_url, e)
                        break  # move on to next list

                    listings = self._parser.parse(html, source_list=source_list)
                    if not listings:
                        break  # empty page = no more books on this list

                    per_list_cap = self._config.max_books_per_list
                    async with lock:  # protect shared state
                        for ls in listings:
                            if new_here >= per_list_cap:
                                break  # hit per-list cap
                            if not ls.book_id or ls.book_id in seen_book_ids:
                                continue  # skip duplicates or invalid IDs
                            seen_book_ids.add(ls.book_id)  # mark as seen
                            total_new += 1
                            new_here += 1
                            queue.put_nowait(ls)  # push into queue for Phase 2
                            if len(seen_book_ids) >= self._config.target_books:
                                break  # global target reached

                    if new_here >= per_list_cap:
                        break  # done with this list

                    # check if there's a next page of this list
                    if not self._parser.has_next_page(html, page):
                        break  # no more pages

                lists_done += 1
                if new_here:
                    gr_logger.info(
                        "[1b %d/%d] %s — +%d books (total queued: %d)",
                        lists_done, len(list_urls), source_list, new_here, total_new,
                    )
                return new_here

        # launch all list-scrape tasks concurrently (semaphore limits actual parallelism)
        tasks = [asyncio.create_task(process_list(u)) for u in list_urls]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            queue.put_nowait(None)  # sentinel: tells Phase 2 "no more books coming"

        gr_logger.info("Phase 1b complete: %d unique books queued", total_new)
        return total_new


class GoodreadsDetailScraper(BaseDetailScraper):
    """Phase 2: scrapes individual Goodreads book pages for full metadata + reviews."""

    def __init__(self, config: GoodreadsConfig, client: AsyncHttpClient):
        self._config = config
        self._client = client
        self._parser = BookPageParser()

    async def scrape_from_queue(
        self,
        queue: asyncio.Queue,     # BookListing objects arrive here from Phase 1b
        *writers: CsvWriter,      # writers[0] = books CSV, writers[1] = reviews CSV
    ) -> tuple[int, ...]:
        """Consume listings from the queue, scrape each book page, write to CSV.

        Returns (books_scraped, reviews_scraped).
        """
        book_writer, review_writer = writers[0], writers[1]
        flush_every = self._config.flush_every

        # load already-scraped URLs so we can skip them
        already_scraped = book_writer.get_existing_values("url")
        if already_scraped:
            gr_logger.info("Crash-resume: found %d already-scraped books, will skip them", len(already_scraped))

        # buffers — we accumulate rows and flush in batches for efficiency
        book_buf: list[BookDetail] = []
        review_buf: list[Review] = []
        books_done = 0        # successful scrapes
        books_failed = 0      # failed scrapes
        total_reviews = 0     # total reviews collected across all books

        start_time = time.monotonic()  # for calculating scraping speed
        pending: set[asyncio.Task] = set()  # currently in-flight scrape tasks
        list_done = False  # whether Phase 1b has finished (sentinel received)

        while True:
            # --- fill up the pending set with new tasks ---
            while len(pending) < self._config.detail_concurrency and not list_done:
                try:
                    listing = queue.get_nowait()  # non-blocking dequeue
                except asyncio.QueueEmpty:
                    break  # queue is empty right now, wait for more

                if listing is None:
                    list_done = True  # Phase 1b is done, no more books coming
                    gr_logger.info("List-scrape phase complete, draining %d remaining", len(pending))
                    break

                # validate the URL before spending a network request on it
                if not listing.url or not listing.url.startswith("https://www.goodreads.com/book/show/"):
                    gr_logger.debug("Skipping invalid URL: %r for '%s'", listing.url, listing.title)
                    continue

                # skip if already scraped in prev run
                if listing.url in already_scraped:
                    gr_logger.debug("Skipping already-scraped: %s", listing.title)
                    continue


                task = asyncio.create_task(self._scrape_one(listing))  # launch async scrape
                pending.add(task)

            # --- check termination ---
            if not pending and list_done:
                break  # all done!
            if not pending:
                await asyncio.sleep(0.1)  # nothing to do yet, wait briefly
                continue

            # --- wait for at least one task to finish ---
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    detail, reviews, title = task.result()  # unpack successful result
                    book_buf.append(detail)
                    review_buf.extend(reviews)
                    books_done += 1
                    total_reviews += len(reviews)
                    gr_logger.info("[%d done] ✓ %s (%d reviews)", books_done, title, len(reviews))
                except Exception as e:
                    books_failed += 1
                    gr_logger.error("[%d failed] ✗ %s: %s", books_failed, type(e).__name__, e)

            # --- flush buffers to disk periodically ---
            if len(book_buf) >= flush_every:
                book_writer.append(book_buf)
                review_writer.append(review_buf)
                book_buf.clear()
                review_buf.clear()
                elapsed = time.monotonic() - start_time
                rate = books_done / elapsed if elapsed > 0 else 0  # books per second
                gr_logger.info("Flushed to CSV | %d scraped, %d failed (%.2f books/sec)",
                               books_done, books_failed, rate)

        # flush any remaining buffered data
        if book_buf:
            book_writer.append(book_buf)
            review_writer.append(review_buf)

        return books_done, total_reviews

    async def _scrape_one(self, listing: BookListing) -> tuple[BookDetail, list[Review], str]:
        """Scrape a single book's metadata + embedded reviews."""
        html = await self._client.get(listing.url)  # fetch the book page
        detail_dict, review_dicts = self._parser.parse(html, self._config.max_reviews_per_book)

        # merge list-discovery fields into the detail (source_list, list_rank)
        detail_dict.setdefault("url", listing.url)
        detail_dict["source_list"] = listing.source_list
        detail_dict["list_rank"] = listing.list_rank
        if not detail_dict.get("title"):
            detail_dict["title"] = listing.title  # fallback to list title if page title is missing

        # build BookDetail, ignoring any extra keys not in the dataclass
        valid_fields = {f.name for f in fields(BookDetail)}
        filtered = {k: v for k, v in detail_dict.items() if k in valid_fields}
        detail = BookDetail(**filtered)

        # build Review objects, attaching the book title and URL to each one
        book_url = detail.url or listing.url
        book_title = detail.title or listing.title
        reviews = [Review(book_title=book_title, book_url=book_url, **r) for r in review_dicts]

        return detail, reviews, book_title


class GoodreadsPipeline(BasePipeline):
    """Top-level orchestrator for the Goodreads scraping pipeline."""

    def __init__(self, config: GoodreadsConfig | None = None):
        self._config = config or GoodreadsConfig()

    def run(self) -> None:
        """Execute the full pipeline. Uses cache if available."""
        cache_path = os.path.join(self._config.output_dir, self._config.listings_cache)
        cached = load_cache(cache_path, BookListing)  # try loading previous discovery results

        if cached:
            gr_logger.info("Loaded %d book listings from cache: %s", len(cached), cache_path)
            gr_logger.info("Skipping Phases 1a+1b, feeding cache directly to scraper")
            asyncio.run(self._run_from_cache(cached))
        else:
            gr_logger.info("No cache found — running full pipeline (Phase 1a → 1b → 2)")
            asyncio.run(self._run_interweaved())

    async def _run_interweaved(self) -> None:
        """Run all phases: discovery feeds a queue that detail scraping consumes."""
        cache_path = os.path.join(self._config.output_dir, self._config.listings_cache)

        # Phase 1a: synchronous discovery of Listopia list URLs
        discoverer = GoodreadsListDiscoverer(self._config)
        list_urls = discoverer.discover()
        if not list_urls:
            gr_logger.error("Phase 1a returned no lists — aborting")
            return

        # shared queue: Phase 1b pushes BookListings, Phase 2 consumes them
        queue: asyncio.Queue[BookListing | None] = asyncio.Queue(maxsize=500)
        seen_book_ids: set[str] = set()  # prevents duplicate books across lists

        # separate HTTP clients for list vs detail scraping (independent rate limiting)
        list_client = AsyncHttpClient(
            self._config.user_agents, self._config.request_delay,
            self._config.max_retries, self._config.retry_backoff,
            self._config.default_rate_limit_wait,
        )
        detail_client = AsyncHttpClient(
            self._config.user_agents, self._config.request_delay,
            self._config.max_retries, self._config.retry_backoff,
            self._config.default_rate_limit_wait,
        )
        await list_client.start(self._config.list_concurrency)
        await detail_client.start(self._config.detail_concurrency)

        # open CSV output files
        out = self._config.output_dir
        book_writer = CsvWriter(os.path.join(out, self._config.books_csv), BookDetail)
        review_writer = CsvWriter(os.path.join(out, self._config.reviews_csv), Review)
        book_writer.open()
        review_writer.open()

        list_scraper = GoodreadsListScraper(self._config, list_client)
        detail_scraper = GoodreadsDetailScraper(self._config, detail_client)

        # tap into the queue to collect listings for caching
        cached_listings: list[BookListing] = []
        original_put = queue.put_nowait
        def tap_put(item):
            if isinstance(item, BookListing):
                cached_listings.append(item)  # save a copy for the cache
            return original_put(item)
        queue.put_nowait = tap_put  # monkey-patch the queue to intercept puts

        try:
            # run Phase 1b (list scraping) and Phase 2 (detail scraping) concurrently
            list_task = asyncio.create_task(
                list_scraper.scrape_into_queue(list_urls, queue, seen_book_ids)
            )
            detail_task = asyncio.create_task(
                detail_scraper.scrape_from_queue(queue, book_writer, review_writer)
            )

            list_total = await list_task         # wait for Phase 1b to finish
            books_n, reviews_n = await detail_task  # wait for Phase 2 to finish

            gr_logger.info(
                "Pipeline done! %d lists → %d books queued → %d scraped, %d reviews",
                len(list_urls), list_total, books_n, reviews_n,
            )
        finally:
            book_writer.close()
            review_writer.close()
            await list_client.close()
            await detail_client.close()
            save_cache(cache_path, cached_listings, dedup_key="book_id")  # persist for re-runs

    async def _run_from_cache(self, listings: list[BookListing]) -> None:
        """Run Phase 2 only, using cached listings from a prior run."""
        # filter out any invalid URLs that might have snuck into the cache
        valid = [ls for ls in listings
                 if ls.url and ls.url.startswith("https://www.goodreads.com/book/show/")]
        if len(valid) < len(listings):
            gr_logger.warning("Filtered %d invalid URLs from cache (%d → %d)",
                              len(listings) - len(valid), len(listings), len(valid))

        # load everything into the queue at once + sentinel
        queue: asyncio.Queue[BookListing | None] = asyncio.Queue()
        for ls in valid:
            queue.put_nowait(ls)
        queue.put_nowait(None)  # sentinel: no more items

        out = self._config.output_dir
        book_writer = CsvWriter(os.path.join(out, self._config.books_csv), BookDetail)
        review_writer = CsvWriter(os.path.join(out, self._config.reviews_csv), Review)
        book_writer.open()
        review_writer.open()

        detail_client = AsyncHttpClient(
            self._config.user_agents, self._config.request_delay,
            self._config.max_retries, self._config.retry_backoff,
            self._config.default_rate_limit_wait,
        )
        await detail_client.start(self._config.detail_concurrency)

        try:
            detail_scraper = GoodreadsDetailScraper(self._config, detail_client)
            books_n, reviews_n = await detail_scraper.scrape_from_queue(
                queue, book_writer, review_writer
            )
            gr_logger.info("Done from cache! %d books, %d reviews", books_n, reviews_n)
        finally:
            book_writer.close()
            review_writer.close()
            await detail_client.close()
