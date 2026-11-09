# DS4221 Movie RAG

Question answering over 42,204 movie plot summaries from the CMU Movie Summary Corpus, for DS4221 Advanced Databases (UVA, Fall 2026).

**Read [CLAUDE.md](CLAUDE.md) first.** It has the project plan, current status, settled decisions and code style. Claude Code loads it automatically.

## Run

With Python 3.13, from the repo root:

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
pip install -r requirements.txt
python unzip.py         # extract the corpus from data/MovieSummaries.tar.gz
python ingest.py        # load it into data/movies.duckdb
python build_index.py   # chunk, embed and index: roughly 45 min on a laptop GPU; resumes if stopped
python ask.py "In Jaws, how do they kill the shark?"
python ask.py           # keeps asking and loads the model only once: use this for the demo
```

Every script is safe to re-run. Without `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` set, `ask.py` prints the prompt it would send instead of an answer. Close the DuckDB CLI first, or open it with `duckdb -readonly`, because scripts can't open the database while the CLI holds it.
