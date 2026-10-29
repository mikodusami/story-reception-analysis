import pandas as pd
import pytest

pytest.importorskip('lingua')

from story_reception.language import detect_language  # noqa: E402


def test_detect_language():
    reviews = pd.Series([
        'This was one of the best books I have read all year, I loved every page of it.',
        'Me encantó este libro, los personajes son increíbles y la historia es preciosa.',
        'Ich habe dieses Buch so gerne mögen wollen, aber die Geschichte war zu langsam.',
        '<p>Read my full review here https://example.com</p> The pacing was slow but the ending worked.',
        '',
    ])
    assert detect_language(reviews).tolist() == ['en', 'es', 'de', 'en', 'unknown']
