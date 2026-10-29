"""Shared scraping infrastructure: config, HTTP clients with retry/backoff, CSV writer, cache, base classes.

Sources plug in by subclassing BaseDiscoverer (phase 1: find item URLs), BaseDetailScraper
(phase 2: scrape each item) and BasePipeline (run both, stream rows to CSV, cache listings for resumes).
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any

import aiohttp
import requests

DATA_DIR = "data"  # default folder for raw CSVs and caches (relative to where you run from)


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)  # capture everything; handlers filter what they show
    lgr.propagate = False
    # only add handlers once — avoids duplicate log lines if the cell is re-run
    if not lgr.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s:%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        lgr.addHandler(console)

        if log_file:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8", delay=True)  # opened on first log line
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s:%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            lgr.addHandler(fh)

    return lgr

# shared logger for pipeline orchestration
logger = setup_logger("scraper")


@dataclass(frozen=True)
class BaseScraperConfig:
    """Common configuration fields shared across all scraper sources."""

    # --- rate limiting ---
    request_delay: float = 1.5    # seconds to wait between requests (prevents bans)
    max_retries: int = 4         # how many times to retry a failed request
    retry_backoff: float = 3.0     # exponential backoff base: 3s, 9s, 27s, 81s
    default_rate_limit_wait: float = 60.0  # fallback wait when 429 has no Retry-After header

    # --- concurrency ---
    concurrency: int = 6   # max simultaneous async requests (6 at a time)

    # --- output ---
    output_dir: str = DATA_DIR   # where CSV files and caches are written
    flush_every: int = 25     # write buffered rows to disk every N items

    # --- user agents ---
    # we rotate these per-request so the target site sees different "browsers"
    # and is less likely to fingerprint and block us
    user_agents: tuple[str, ...] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    )


def browser_headers(user_agent: str) -> dict[str, str]:
    """Build a full set of HTTP headers that mimic a real browser.

    Why bother? Sites like Goodreads and RT use Cloudflare or similar
    that check for missing Accept-Language, Sec-Fetch-* headers,
    etc. Sending realistic headers reduces blocking.
    """
    return {
        "User-Agent": user_agent, # identifies the "browser" type
        "Accept": (     # tells server what content types we want
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",    # English language preference
        "Accept-Encoding": "gzip, deflate, br", # we can handle compressed responses
        "Cache-Control": "no-cache",  # don't serve us stale cached pages
        "Pragma": "no-cache",   # same as above, for HTTP/1.0 compat
        "Sec-Fetch-Dest": "document",  # browser navigation metadata
        "Sec-Fetch-Mode": "navigate",  # we're navigating to a page
        "Sec-Fetch-Site": "none",     # not coming from another site
        "Sec-Fetch-User": "?1",     # user-initiated request
        "Upgrade-Insecure-Requests": "1", # prefer HTTPS
    }


class SyncHttpClient:
    """Blocking HTTP client with retry + exponential backoff.

    Used for small-volume discovery phases where async is overkill.
    like Goodreads Phase 1a fetches ~90 genre index pages sequentially.
    """

    def __init__(
        self,
        user_agents: tuple[str, ...],  # pool of user-agent strings to rotate through
        delay: float = 1.5,            # minimum seconds between requests
        max_retries: int = 4,          # give up after this many failures
        retry_backoff: float = 3.0,    # exponential backoff base (3^1=3s, 3^2=9s, ...)
    ):
        self._session = requests.Session()  # reuses TCP connections for speed
        self._user_agents = user_agents
        self._delay = delay
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def get(self, url: str) -> str:
        """Fetch a URL, retrying on 429s and server errors."""
        for attempt in range(self._max_retries + 1):
            time.sleep(self._delay)  # polite delay before every request
            ua = random.choice(self._user_agents)  # pick a random user-agent each time
            try:
                resp = self._session.get(
                    url, headers=browser_headers(ua), timeout=30  # 30s timeout per request
                )

                # --- 429 Too Many Requests: the site is rate-limiting us ---
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")  # server may tell us how long to wait
                    wait = (
                        float(retry_after)                         # use server's suggestion if available
                        if retry_after
                        else self._retry_backoff ** (attempt + 1)  # otherwise exponential backoff
                    )
                    logger.warning("429 (sync) on %s, waiting %.1fs", url, wait)
                    time.sleep(wait)
                    continue  # retry the request

                # --- 5xx Server Error: the server is having problems ---
                if resp.status_code >= 500:
                    wait = self._retry_backoff ** (attempt + 1)
                    logger.warning("HTTP %d on %s, retrying in %.1fs", resp.status_code, url, wait)
                    time.sleep(wait)
                    continue  # retry

                resp.raise_for_status()  # raise exception for any other non-2xx status
                return resp.text         # success — return the HTML

            except requests.RequestException as e:
                if attempt == self._max_retries:
                    raise  # give up after max retries
                wait = self._retry_backoff ** (attempt + 1)
                logger.warning(
                    "Sync request error on %s (attempt %d/%d): %s, retrying in %.1fs",
                    url, attempt + 1, self._max_retries, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"All retries exhausted for {url}")


class AsyncHttpClient:
    """Non-blocking HTTP client with concurrency control, retry, and
    global rate-limit cool-off.

    This is the workhorse for Phase 2 (detail scraping) where we're
    hitting thousands of pages so the scraping can be fast.
    Key features:

    - Semaphore caps how many requests fly simultaneously
    - Exponential backoff on 429, 403, and 5xx errors
    - Honors Retry-After header when the server sends one
    - Global cool-off: when ONE request gets a 429, ALL concurrent
      requests pause too (prevents cascading 429s)
    - Jittered delay: adds random 0-0.5s to the base delay so
      requests don't fire in synchronized bursts
    """

    def __init__(
        self,
        user_agents: tuple[str, ...],            # pool of UAs to rotate through
        delay: float = 1.5,                      # base delay between requests
        max_retries: int = 4,                    # max retry attempts per URL
        retry_backoff: float = 3.0,              # backoff multiplier (3^1, 3^2, ...)
        default_rate_limit_wait: float = 60.0,   # fallback wait for 429 without Retry-After
    ):
        self._user_agents = user_agents
        self._delay = delay
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._default_rate_limit_wait = default_rate_limit_wait
        self._session: aiohttp.ClientSession | None = None    # created in start()
        self._semaphore: asyncio.Semaphore | None = None      # created in start()
        self._rate_limit_until: float = 0.0  # monotonic timestamp: all requests pause until this time

    async def start(self, concurrency: int) -> None:
        """Initialize the aiohttp session and concurrency semaphore.

        Must be called before any get() calls. Separated from __init__
        because async operations can't happen in a constructor.
        """
        self._semaphore = asyncio.Semaphore(concurrency)  # at most N requests in flight
        connector = aiohttp.TCPConnector(limit=concurrency * 2)  # TCP connection pool (2× concurrency headroom)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=45),  # 45s total timeout per request
        )

    async def close(self) -> None:
        """Shut down the aiohttp session and release connections."""
        if self._session:
            await self._session.close()

    async def _wait_if_rate_limited(self) -> None:
        """If a recent 429 triggered a global cool-off, sleep until it expires.

        This prevents other concurrent tasks from hammering the server
        while it's already telling us to slow down.
        """
        now = time.monotonic()
        if now < self._rate_limit_until:
            wait = self._rate_limit_until - now  # how much longer to wait
            logger.debug("Honoring global rate-limit cool-off: %.1fs", wait)
            await asyncio.sleep(wait)

    def _set_rate_limit_cooloff(self, seconds: float) -> None:
        """Set a global cool-off: ALL requests pause for `seconds` from now.

        Only extends the deadline — never shortens it. This way if two
        429s arrive close together, we honor the longer wait.
        """
        target = time.monotonic() + seconds
        if target > self._rate_limit_until:
            self._rate_limit_until = target

    async def get(self, url: str) -> str:
        """Fetch a URL with rate limiting, retry, and global cool-off."""
        async with self._semaphore:  # block here if we're at max concurrency
            for attempt in range(self._max_retries + 1):
                await self._wait_if_rate_limited()  # honor any active global cool-off

                # jittered delay: base + random 0-0.5s prevents synchronized bursts
                await asyncio.sleep(self._delay + random.uniform(0, 0.5))

                ua = random.choice(self._user_agents)  # rotate user-agent per request
                try:
                    async with self._session.get(url, headers=browser_headers(ua)) as resp:

                        # --- 429 Too Many Requests ---
                        if resp.status == 429:
                            retry_after = resp.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait = float(retry_after)  # server says "wait N seconds"
                                except ValueError:
                                    wait = self._default_rate_limit_wait  # header was malformed
                            else:
                                wait = self._retry_backoff ** (attempt + 1)  # exponential backoff
                            self._set_rate_limit_cooloff(wait)  # pause ALL concurrent requests too
                            logger.warning(
                                "429 on %s, cooling off %.1fs (attempt %d/%d)",
                                url, wait, attempt + 1, self._max_retries,
                            )
                            await asyncio.sleep(wait)
                            continue

                        # --- 403 Forbidden (usually Cloudflare challenge) ---
                        if resp.status == 403:
                            wait = self._default_rate_limit_wait  # back way off
                            self._set_rate_limit_cooloff(wait)    # pause everyone
                            logger.warning("403 on %s (likely Cloudflare), cooling off %.1fs", url, wait)
                            await asyncio.sleep(wait)
                            continue

                        # --- 5xx Server Error ---
                        if resp.status >= 500:
                            wait = self._retry_backoff ** (attempt + 1)
                            logger.warning("HTTP %d on %s, retrying in %.1fs", resp.status, url, wait)
                            await asyncio.sleep(wait)
                            continue

                        resp.raise_for_status()      # raise on any other non-2xx
                        return await resp.text()     # success!

                except Exception as e:
                    if attempt == self._max_retries:
                        raise  # exhausted all retries — let the caller handle it
                    wait = self._retry_backoff ** (attempt + 1)
                    logger.warning(
                        "Request error on %s (attempt %d/%d): %s: %s, retrying in %.1fs",
                        url, attempt + 1, self._max_retries, type(e).__name__, e, wait,
                    )
                    await asyncio.sleep(wait)

            raise RuntimeError(f"All retries exhausted for {url}")


class CsvWriter:
    """Streaming CSV writer for dataclass instances.

    Usage:
        writer = CsvWriter("output/books.csv", BookDetail)
        writer.open()          # creates file, writes header row
        writer.append([b1, b2])  # converts dataclasses → dicts → CSV rows, flushes
        writer.close()         # closes file, logs row count
    """

    def __init__(self, filepath: str, cls: type):
        self._filepath = filepath
        self._cls = cls
        # extract field names from the dataclass to use as CSV column headers
        self._fieldnames = [f.name for f in fields(cls)]
        self._file = None      # file handle, set in open()
        self._writer = None    # csv.DictWriter, set in open()
        self._count = 0        # running count of rows written

    def open(self) -> None:
        """Open the CSV file. If it already exists, append to it.
        If it doesn't exist, create it and write the header."""
        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)

        if os.path.exists(self._filepath):
            # --- RESUME MODE: file exists from a previous run ---
            # count how many rows are already there
            with open(self._filepath, encoding="utf-8") as f:
                self._count = sum(1 for _ in f) - 1  # subtract the header row
            if self._count < 0:
                self._count = 0
            # open in append mode — no header written
            self._file = open(self._filepath, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            logger.info("Resuming CSV: %s (%d rows already present)", self._filepath, self._count)
        else:
            # --- FRESH MODE: new file ---
            self._file = open(self._filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._count = 0

    def get_existing_values(self, column: str) -> set[str]:
        """Read the existing CSV and return all values from the given column.
        Call this AFTER open() to find out what's already been scraped."""
        values: set[str] = set()
        if not os.path.exists(self._filepath):
            return values
        try:
            with open(self._filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val = row.get(column, "")
                    if val:
                        values.add(val)
        except Exception as e:
            logger.warning("Could not read existing CSV for resume check: %s", e)
        return values

    def append(self, rows: list) -> None:
        """Convert dataclass instances to dicts and write them as CSV rows."""
        if not self._writer:
            raise RuntimeError("CsvWriter not opened — call open() first")
        for row in rows:
            self._writer.writerow(asdict(row))  # dataclass to dict to CSV row
            self._count += 1
        self._file.flush()  # force write to disk so data survives crashes

    def close(self) -> None:
        """Close the file and log how many rows were written."""
        if self._file:
            self._file.close()
            logger.info("Saved %d rows to %s", self._count, self._filepath)
            self._file = None
            self._writer = None




def save_cache(path: str, items: list, dedup_key: str | None = None) -> None:
    """Save a list of dataclass instances to a JSON file.

    Args:
        path:      where to write the JSON file
        items:     list of dataclass instances to persist
        dedup_key: if set, only keep the first occurrence of each value
                   for this field (e.g. dedup_key="book_id" removes duplicate books)
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # create parent dirs if needed
        dicts: list[dict] = []
        seen: set[str] = set()  # tracks which dedup_key values we've already added
        for item in items:
            d = asdict(item)  # convert dataclass → dict for JSON serialization
            if dedup_key:
                val = d.get(dedup_key, "")
                if val and val in seen:
                    continue  # skip duplicate
                seen.add(val)
            dicts.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dicts, f, ensure_ascii=False)  # write to disk
        logger.info("Cache saved: %d items → %s", len(dicts), path)
    except Exception as e:
        logger.warning("Failed to save cache: %s", e)


def load_cache[T](path: str, cls: type[T]) -> list[T] | None:
    """Load cached dataclass instances from a JSON file.

    Returns None if the file doesn't exist or is corrupt — the pipeline
    will then fall through to a full discovery run.
    """
    if not os.path.exists(path):
        return None  # no cache file → run full discovery
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)  # list of dicts
        return [cls(**row) for row in data]  # reconstruct dataclass instances
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Cache file corrupt, ignoring: %s", e)
        return None


class BaseDiscoverer(ABC):
    """Phase 1: discover item URLs to scrape.

    Each source implements this differently:
      - Goodreads: crawl genre index pages → list pages → book URLs
      - Rotten Tomatoes: scroll through JS-rendered browse pages → movie URLs
      - A hypothetical Amazon scraper: search result pages → product URLs
    """

    @abstractmethod
    def discover(self) -> list[Any]:
        """Return a list of lightweight listing objects (BookListing, MovieListing, etc.)
        that will be scraped in the detail phase."""
        ...


class BaseDetailScraper(ABC):
    """Phase 2: scrape full detail pages from discovered listings.

    Consumes items from an asyncio.Queue (fed by Phase 1) and writes
    results to one or more CsvWriters.
    """

    @abstractmethod
    async def scrape_from_queue(
        self,
        queue: asyncio.Queue,   # items arrive here from the discoverer
        *writers: CsvWriter,    # one or more CSV writers for different output types
    ) -> tuple[int, ...]:
        """Consume listings from the queue, scrape each one, write to CSV.

        Returns a tuple of counts (e.g. (books_scraped, reviews_scraped)).
        A None sentinel in the queue signals "no more items coming".
        """
        ...


class BasePipeline(ABC):
    """Top-level orchestrator that wires discovery → detail scraping → CSV output.

    Each source implements run() to set up its specific phases, create
    HTTP clients, open CSV writers, and coordinate everything.
    """

    @abstractmethod
    def run(self) -> None:
        """Execute the full scraping pipeline end-to-end."""
        ...
