"""python -m story_reception.scrape [goodreads] [rottentomatoes] [--output-dir data]"""
import argparse
import sys

from . import PIPELINES, run_pipelines
from .base import DATA_DIR


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m story_reception.scrape',
                                     description='Scrape Goodreads and/or Rotten Tomatoes into CSV files.')
    parser.add_argument('sources', nargs='*', metavar='source',
                        help=f"one or more of: {', '.join(PIPELINES)} (default: all)")
    parser.add_argument('--output-dir', default=DATA_DIR, help=f'folder for CSVs and caches (default: {DATA_DIR})')
    args = parser.parse_args(argv)
    unknown = [s for s in args.sources if s not in PIPELINES]
    if unknown:
        parser.error(f"unknown source(s): {', '.join(unknown)}; choose from {', '.join(PIPELINES)}")
    run_pipelines(args.sources or None, output_dir=args.output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
