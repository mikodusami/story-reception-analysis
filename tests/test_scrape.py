"""Offline tests for the scraping package: no network requests are made."""
import csv
import json
from dataclasses import dataclass

import pytest

pytest.importorskip('aiohttp')
pytest.importorskip('bs4')

from story_reception.scrape import PIPELINES, run_pipelines  # noqa: E402
from story_reception.scrape.base import CsvWriter, browser_headers, load_cache, save_cache  # noqa: E402
from story_reception.scrape.goodreads import (  # noqa: E402
    BookListing,
    GoodreadsConfig,
    GoodreadsPipeline,
    build_list_index_urls,
)
from story_reception.scrape.rottentomatoes import RottenTomatoesConfig, RottenTomatoesPipeline  # noqa: E402


@dataclass
class Row:
    name: str
    score: str = ''


def test_csv_writer_writes_header_then_resumes_without_one(tmp_path):
    path = tmp_path / 'out' / 'rows.csv'
    w = CsvWriter(str(path), Row)
    w.open()
    w.append([Row('a', '1'), Row('b', '2')])
    w.close()

    w = CsvWriter(str(path), Row)          # a second run appends (resume mode)
    w.open()
    assert w.get_existing_values('name') == {'a', 'b'}
    w.append([Row('c', '3')])
    w.close()

    rows = list(csv.DictReader(path.open()))
    assert [r['name'] for r in rows] == ['a', 'b', 'c']
    assert path.read_text().count('name,score') == 1


def test_cache_round_trip_with_dedup(tmp_path):
    path = tmp_path / 'cache.json'
    items = [BookListing(title='A', url='u1', book_id='1'), BookListing(title='A again', url='u1b', book_id='1'),
             BookListing(title='B', url='u2', book_id='2')]
    save_cache(str(path), items, dedup_key='book_id')
    loaded = load_cache(str(path), BookListing)
    assert [b.book_id for b in loaded] == ['1', '2']
    assert loaded[0].title == 'A'                       # first occurrence wins


def test_load_cache_missing_or_corrupt(tmp_path):
    assert load_cache(str(tmp_path / 'nope.json'), BookListing) is None
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    assert load_cache(str(bad), BookListing) is None
    wrong_shape = tmp_path / 'wrong.json'
    wrong_shape.write_text(json.dumps([{'unexpected_field': 1}]))
    assert load_cache(str(wrong_shape), BookListing) is None


def test_configs_and_url_builders():
    gr = GoodreadsConfig()
    assert gr.output_dir == 'data' and len(gr.genre_tags) == 30
    assert GoodreadsConfig(output_dir='elsewhere').output_dir == 'elsewhere'

    urls = build_list_index_urls(['fiction', 'horror'], max_pages_per_genre=2)
    assert urls == ['https://www.goodreads.com/list/tag/fiction?page=1',
                    'https://www.goodreads.com/list/tag/fiction?page=2',
                    'https://www.goodreads.com/list/tag/horror?page=1',
                    'https://www.goodreads.com/list/tag/horror?page=2']

    rt_urls = RottenTomatoesConfig().category_urls
    assert len(rt_urls) == len(set(rt_urls)) == 564
    assert all(u.startswith('https://www.rottentomatoes.com/browse/') for u in rt_urls)


def test_browser_headers_use_given_user_agent():
    headers = browser_headers('TestAgent/1.0')
    assert headers['User-Agent'] == 'TestAgent/1.0'
    assert 'Accept-Language' in headers


def test_registry_builds_pipelines_without_running(tmp_path):
    gr = PIPELINES['goodreads'](str(tmp_path))
    rt = PIPELINES['rottentomatoes'](str(tmp_path))
    assert isinstance(gr, GoodreadsPipeline) and isinstance(rt, RottenTomatoesPipeline)
    with pytest.raises(ValueError, match='Unknown source'):
        run_pipelines(['amazon'])
