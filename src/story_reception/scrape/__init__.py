"""Scrapers for Goodreads and Rotten Tomatoes.

    from story_reception.scrape import run_pipelines
    run_pipelines(["goodreads"])                 # or ["rottentomatoes"], or None for all

or from a terminal: python -m story_reception.scrape goodreads --output-dir data

Each source is imported only when it runs, so Goodreads works without Playwright installed.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import DATA_DIR, logger


def _goodreads(output_dir: str):
    from .goodreads import GoodreadsConfig, GoodreadsPipeline
    return GoodreadsPipeline(GoodreadsConfig(output_dir=output_dir))


def _rottentomatoes(output_dir: str):
    from .rottentomatoes import RottenTomatoesConfig, RottenTomatoesPipeline
    return RottenTomatoesPipeline(RottenTomatoesConfig(output_dir=output_dir))


# Adding a source: write a module with a config and a BasePipeline subclass, then register it here.
PIPELINES: dict[str, Callable[[str], object]] = {
    'goodreads': _goodreads,
    'rottentomatoes': _rottentomatoes,
}


def run_pipelines(names: list[str] | None = None, output_dir: str = DATA_DIR) -> None:
    """Run one or more scraping pipelines by name (all registered pipelines if names is None)."""
    requested = names or list(PIPELINES)
    unknown = [n for n in requested if n not in PIPELINES]
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}. Available: {', '.join(PIPELINES)}")

    for name in requested:
        logger.info('=' * 60)
        logger.info('Starting pipeline: %s', name)
        logger.info('=' * 60)
        try:
            PIPELINES[name](output_dir).run()
        except Exception as e:
            logger.error("Pipeline '%s' failed: %s: %s", name, type(e).__name__, e)
            raise
