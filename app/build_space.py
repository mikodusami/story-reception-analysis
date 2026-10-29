"""Assemble a Hugging Face Space (Docker, running Streamlit) in build/space/ with only what the app needs.

    python app/build_space.py          (or: make space)

Then publish it (needs a Hugging Face account; `hf` comes with huggingface_hub):
    hf auth login
    hf repos create <your-username>/story-reception --type space --sdk docker

Hugging Face no longer offers a built-in Streamlit SDK for new Spaces, so the bundle includes a Dockerfile.
Docker Spaces need a Hugging Face PRO subscription; the free route is Streamlit Community Cloud (see README).
    hf upload <your-username>/story-reception build/space . --repo-type space
"""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'build' / 'space'
REPO_URL = 'https://github.com/mikodusami/story-reception-analysis'

FILES = [
    'app/app.py', 'app/predict.py',
    'src/story_reception/__init__.py', 'src/story_reception/clean.py', 'src/story_reception/features.py',
    '.streamlit/config.toml',
]

# Hugging Face Docker Spaces run the container as user 1000 and route traffic to app_port (set in README.md)
DOCKERFILE = '''FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

README = f"""---
title: Story Reception
emoji: 📚
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
short_description: What makes critics, audiences and readers react differently
---

# Story Reception

Interactive front end for [story-reception-analysis]({REPO_URL}): three models trained on about 180,000 Goodreads
and Rotten Tomatoes reviews, each scored on books and films it never saw.

- **Review → rating:** predicts a review's star rating and shows which words drove it.
- **Critics vs. audiences:** predicts, from pre-release facts only, whether critics and audiences will disagree.
- **Divisive books:** how split a book's readers are, relative to its average rating.

This Space ships trained models and small samples only. Code, data pipeline and full analysis are on GitHub.
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, OUT / f)
    shutil.copytree(REPO / 'app' / 'artifacts', OUT / 'app' / 'artifacts')
    shutil.copy2(REPO / 'app' / 'requirements.txt', OUT / 'requirements.txt')   # pinned to the training versions
    (OUT / 'README.md').write_text(README)
    (OUT / 'Dockerfile').write_text(DOCKERFILE)
    size = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
    print(f'Built {OUT.relative_to(REPO)} ({size / 1e6:.1f} MB):')
    for f in sorted(OUT.rglob('*')):
        if f.is_file():
            print(f'  {f.relative_to(OUT)}')
    print((OUT / 'requirements.txt').read_text())


if __name__ == '__main__':
    main()
