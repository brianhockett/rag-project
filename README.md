# DS4221 Movie RAG

Question answering over 42,204 movie plot summaries from the CMU Movie Summary Corpus, for DS4221 Advanced Databases (UVA, Fall 2026).

**Read [CLAUDE.md](CLAUDE.md) first.** It has the project plan, current status, settled decisions and code style. Claude Code loads it automatically.

## Run

With Python 3.13, from the repo root:

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
pip install -r requirements.txt
python unzip.py    # extract the corpus from data/MovieSummaries.tar.gz
python ingest.py   # load it into data/movies.duckdb
```

Both scripts are safe to re-run. The rest of the pipeline (`build_index.py`, `movie_rag.py`, `ask.py`) isn't written yet.
