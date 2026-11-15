# DS4221 Movie RAG

## Project goal

This is our final project for DS4221 Advanced Databases (UVA, Fall 2026), worth 30% of the grade. The team is Brian Hockett, Vinith Jayamani and Aidan Mayhue.

We are building a retrieval-augmented generation (RAG) system that answers questions about movie plots, such as "In Jaws, how do they kill the shark?", and cites the movies its answer came from.

- **Corpus:** the CMU Movie Summary Corpus, with 42,306 Wikipedia plot summaries plus Freebase metadata (title, release date, genres, languages).
- **Starting point:** Lab 9 (microvector: exact and IVF vector search) and Lab 10 (micro_rag: chunk → embed → retrieve → prompt → answer) from https://github.com/researcher111/AdvanceDatabase. The project spec is at https://researcher111.github.io/AdvanceDatabase/project.html.
- **Storage:** a local DuckDB file, `data/movies.duckdb`. Everything runs locally, with no hosted database and no paid APIs.
- **Retrieval:**
  - EmbeddingGemma 2 (`google/embeddinggemma-2`, 768 dimensions) embeddings in a FAISS IVF index, with an exact Flat index as the baseline.
  - BM25 keyword search (`rank_bm25`, approved by the professor), combined with the vector results using reciprocal rank fusion.
- **Generation:** the course's OpenAI-compatible LLM endpoint, configured with `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`.
- **Evaluation:**
  - document-level recall@k and MRR@10 on our own questions and the partner team's 10 questions;
  - answer-quality scoring;
  - a failure analysis.

**Proposal:** `DS4221 Project Proposal.pdf` in the repo root is the proposal we submitted. Read it before planning work on retrieval, evaluation or the schema, and check that the work still delivers what it promised:
- three kinds of question: plot details ("In Jaws, how do they kill the shark?"), naming a movie from its plot ("What is the space movie where the empire fights against the rebels?") and listing movies on a topic ("What are some movies about time travel?");
- an evaluation on 25+ questions of our own with known source summaries, reporting recall@k and MRR, comparing chunk sizes and Flat against IVF search, and scoring answers with an LLM judge plus spot checks;
- the four tables `documents`, `chunks`, `embeddings` and `eval_questions`.

The schema has grown since the proposal (for example `genres`, `languages` and `version`), so treat `ingest.py` as the current schema and the proposal as the original plan. If a change would drop something the proposal promised, say so before making it.

**Grading:**

| Part | Weight |
|---|---|
| End-to-end system | 25% |
| Retrieval layer | 25% |
| Evaluation | 25% |
| Report and presentation | 15% |
| Milestones | 10% |

**Deadlines (2026):**

| Date | Deadline |
|---|---|
| Thu Nov 12 | Retrieval milestone |
| Thu Nov 19 | Question swap with partner team |
| Mon Nov 23 | Team target for hybrid search |
| Thu Dec 3 | Workshop |
| Mon Dec 7, 11:59 pm | Final report |
| Tue Dec 8 | Presentation |

## Current plan and status

This section is the team's working plan, and every Claude session should follow it. Update it whenever a stage lands or a decision changes. `project_plan.pdf` is a one-page copy of the same plan; if the two disagree, this section wins.

### Status (Sun Nov 15)

Ingestion is done. On Nov 15 we dropped the chunk-size comparison, so the code now has one chunking: windows of at most 256 tokens. Brian rebuilt everything with it on Nov 15 (90,207 chunks), and the three evaluation scripts ran on that build, except `measure_answers.py`, which waits on an LLM endpoint. Generation is written and tested against a fake endpoint, but no real LLM endpoint has been configured yet.

| File | What it does | Status |
|---|---|---|
| `unzip.py` | Extracts `plot_summaries.txt`, `movie.metadata.tsv` and `README.txt` from `data/MovieSummaries.tar.gz` | Done |
| `ingest.py` | Cleans the text and loads 42,204 movies into `documents` (102 dropped: 99 with no metadata, 3 empty after cleaning). Creates all four tables, and loads `data/eval_questions.jsonl` if it exists. | Done |
| `build_index.py` | Chunks, embeds, fills `chunks` and `embeddings`, and builds the FAISS and BM25 indexes | Built Nov 15 with one chunking and a hard 256-token cap: 90,207 chunks in 28.6 min. Each run appends its step times, CPU threads, and GPU name and memory to `data/build_times.json`. |
| `movie_rag.py` | The model, `tokenize()`, `Retriever` (vector, BM25 and hybrid search), `build_prompt()`, `chat()` and `answer()`: Lab 10's `micro_rag.py` for our corpus | Retrieval worked on the Nov 9 full-corpus build; `chat()` tested against a fake endpoint only |
| `ask.py` | Command line: a question in, a cited answer out. With no question it keeps asking, for the demo. | Worked on the Nov 9 build; waits on the rebuild, then on an LLM endpoint for real answers |
| `measure_retrieval.py` | Recall@5 and MRR for five runs (vector-Flat, vector-IVF, BM25, hybrid-Flat, hybrid-IVF), ours and the partner team's side by side; writes `data/retrieval_results.csv` and `data/retrieval_by_question.csv` | Ran Nov 15 at `nprobe` 64: hybrid-Flat found 20 of 25; vector-Flat and BM25 19; vector-IVF and hybrid-IVF 18. At `nprobe` 8, vector-IVF found 16. |
| `measure_speed.py` | `nprobe` sweep (ANN recall@5 against Flat, and query time) and queries per second per search run; writes `data/nprobe_sweep.csv` and `data/query_speed.csv` | Ran Nov 15: ANN recall@5 is 0.63 at `nprobe` 8 and 0.91 at 64 (1.5 ms, against Flat's 9.8 ms); `NPROBE` set to 64 |
| `measure_answers.py` | Runs the real system on every question; writes `data/answers.csv` for the team to grade, and won't overwrite it | Written Nov 15 and tested against a fake endpoint; waits on the LLM endpoint |
| `data/eval_questions.jsonl` | Our 25+ evaluation questions | Written Nov 15: 25 questions (Aidan 9, Brian 9, Vinith 7), all doc IDs checked |

`ingest.py` recreates `movies.duckdb` with 42,204 rows in `documents` and the 25 questions in `eval_questions`; `build_index.py` then fills `chunks` and `embeddings`. A database from before Nov 15 still has a `config` column in `chunks`, and `build_index.py` stops with a message instead of using it. Brian's machine has an RTX 4070 laptop GPU (8 GB), so embedding runs on the GPU.

**Measured in the 500-movie test and on the full corpus's text**, for planning:
- **Full build time:** 28.6 minutes on Nov 15 (`data/build_times.json`): chunking 94 s, embedding 26 min at 57 chunks/s, IVF training 19 s, BM25 16 s. Embedding measured 70 chunks/s on its own; a test running alongside slowed it, so a clean rebuild should take about 25 minutes.
- **BM25 on the full corpus:** a 51 MB pickle that loads in about 2.5 s. A query takes 120–200 ms, because `rank_bm25` scores every chunk in Python.
- **`ask.py` timing:** about 15 s before the first answer (imports take ~13 s, loading the model and indexes ~7 s), then about 0.1 s per question plus the LLM's time. Use the keep-asking mode for the demo.
- **Opening the database:** while the DuckDB CLI has `movies.duckdb` open in read-write mode, no other process can open it, even read-only. Close the CLI or start it with `duckdb -readonly`.

### How the files connect

```
data/MovieSummaries.tar.gz            (committed to git)
        │  unzip.py
        ▼
data/MovieSummaries/plot_summaries.txt, movie.metadata.tsv
        │  ingest.py  ◄── data/eval_questions.jsonl
        ▼
data/movies.duckdb   documents · chunks · embeddings · eval_questions
        │  build_index.py   chunk → embed → fill chunks + embeddings → build indexes
        ▼
FAISS + BM25 index files in data/, each saved with its chunk_id order
        │  movie_rag.py     Retriever, build_prompt(), chat(), answer()
        ▼
ask.py         question in → answer citing [doc_id]
measure_*.py   metrics over eval_questions
```

Every script after `ingest.py` reads from `movies.duckdb`, never from the raw files. Retrieval returns chunk IDs, and joining `chunks` to `documents` gives each hit the `doc_id` and title that the answer cites. `build_index.py` (for chunks) and `movie_rag.py` (for questions) must share one `tokenize()` and one way of loading the embedding model.

### Retrieval design

- **Chunking** (our own code, no LangChain): one chunking for the whole corpus, windows of at most 256 of the embedding model's tokens with 32 tokens of overlap, based on Lab 9's `chunk_text()`. Decoding a window and encoding it again can add a token or two where the window cuts a word, so a window whose text comes to more than 256 tokens is shortened until it fits. `chunk_id` is `doc_id:version:tok256:ordinal`, for example `74830:v1:tok256:3`. The spec asks for the chunking configuration in every `chunk_id`, so `tok256` (`CHUNKING` in `build_index.py`) stays there, and must change if the window or overlap changes.
- **Embedding:** `google/embeddinggemma-2`, 768 dimensions, through `sentence-transformers`, loaded the way Lab 9's `embedding_demo.py` does it. Each chunk is embedded as `title: <title> | text: <chunk text>` because summaries rarely name their own movie; questions use the model's `SearchQuery` prompt. On an NVIDIA GPU the model runs in bfloat16, which is 2–3× faster than float32 and gives vectors that match float32's to a mean cosine of 0.9999; elsewhere it runs in float32. Vectors are re-normalized to unit length in float32, so inner product equals cosine. They're stored in `embeddings.vector`, so the indexes can be rebuilt without re-embedding. `build_index.py` embeds the longest chunks first, in GPU batches of 64 (`EMBED_BATCH`), and saves every 2,048 chunks, so a re-run after a crash resumes where it stopped.
- **Vector index:** FAISS `IndexIVFFlat` with inner product is the product, and `IndexFlatIP` is the exact baseline the spec requires. `nlist` is about 4√N, capped so each list gets at least 39 training vectors. `NPROBE` in `movie_rag.py` is 64, chosen Nov 15 from `measure_speed.py`'s sweep of ANN recall@5 (against the exact top 5) against latency. The full index has 1,201 lists: 8 gave only 0.63, 64 gives 0.91 at 1.5 ms per search (Flat: 9.8 ms), and 128 adds 0.03 for twice the time.
- **Index files**, all in `data/` (paths in `movie_rag.py`'s `INDEX_PATHS`, `CHUNK_IDS_PATH` and `BM25_PATH`):
  - `flat.faiss` and `ivf.faiss`: the two FAISS indexes;
  - `chunk_ids.json`: the `chunk_id` of each FAISS id and BM25 row, in `chunk_id` order (FAISS id `i` is `chunk_ids[i]`);
  - `bm25.pkl`: the BM25 index.

  `Retriever` checks on load that these match the `chunks` table, and stops with a message if they don't.
- **Keyword search** (our one stretch goal): `rank_bm25`'s `BM25Okapi` over each chunk's title, text and genres. The title is included because a question like "In Jaws, …" can't otherwise match on "Jaws". The tokenizer lowercases, strips accents, splits on `[a-z0-9]+`, drops stopwords (Lab 10's `STOPWORDS` plus question words such as "movie" and "film") and Snowball-stems, with a cache on the stemmer, which made tokenizing the corpus about 30 times faster.
- **Fusion:** reciprocal rank fusion over each search's top 50, score = Σ 1/(60 + rank). `Retriever.retrieve(question, k, mode)` takes `mode` = `"vector"`, `"keyword"` or `"hybrid"` (the default), so the comparison can run each one. Vector-only is the fallback.
- **Generation:** Lab 10's prompt shape: `Answer using ONLY the sources below. Cite as [doc_id].`, then one `[doc_id] Title (year): text` line per chunk, then the question. The year tells remakes apart, such as the four Titanics. `chat(prompt)` makes Lab 10's request: one user message posted to `LLM_BASE_URL/chat/completions` with the standard library, plus `temperature` 0 so evaluation runs repeat. Any failure raises `RuntimeError` with a one-line reason. `answer()` calls `chat()` and adds the list of retrieved movies, so citations like `[74830]` can be read. It echoes the prompt when `LLM_BASE_URL` isn't set or the request fails, so the demo keeps going. The spec allows LangChain or LlamaIndex in this step only; the labs use neither, and neither do we.
- **Metrics** are document-level, with one k everywhere: `TOP_K` = 5 in `movie_rag.py`, the number of chunks the LLM gets. Retrieve those 5 chunks, dedupe them to `doc_id`s in rank order, and compute recall@5 and MRR over those movies. The `nprobe` sweep's ANN recall uses the same k. Questions whose `q_id` starts with `p` are the partner team's and are reported separately.
- **Answer quality** is graded by people, not an LLM judge. `measure_answers.py` runs the real system (hybrid, IVF, `TOP_K` chunks, as in `ask.py`) on every question and writes a CSV with the question, correct and retrieved movies, recall@5, reciprocal rank, the LLM's answer, and two blank columns, a 1–10 score and an explanation, for the team to fill in. The failure analysis is written from that sheet.

### How we follow the labs

The pipeline has the same stages and function boundaries as Lab 10's `micro_rag.py`. The differences are where the data lives, the embedding model and the index library.

| Stage | Labs | Ours |
|---|---|---|
| Corpus | `corpus.py`: Python lists | `unzip.py` + `ingest.py` → DuckDB tables |
| Chunk | `chunk_corpus()`: groups of ~60 words | Windows of at most 256 tokens, 32 overlapping |
| Embed | hashed word counts, 2048 dimensions | EmbeddingGemma 2, 768 dimensions |
| Index | microvector `BruteForceIndex` / `IVFIndex`, rebuilt in memory on every run | FAISS `IndexFlatIP` / `IndexIVFFlat` (`n_clusters` → `nlist`, `probe` → `nprobe`), built once by `build_index.py` and saved to disk |
| Retrieve | `Retriever.retrieve()` | same class and return shape, plus BM25 and RRF |
| Prompt and answer | `build_prompt()`, `answer()` | reused almost unchanged |
| Command line and metrics | `ask.py`, `measure_rag.py` | `ask.py`, `measure_*.py` |

`build_index.py` is the only new piece. The lab embeds every chunk each time it starts, which works for 24 documents but not for 42k summaries and a neural model.

### Work left, by deadline

| By | Work |
|---|---|
| Thu Nov 12 | Run `build_index.py` on the full corpus (about 45 minutes), then show 5 example queries with their retrieved chunks from `ask.py`. |
| Thu Nov 19 | `data/eval_questions.jsonl`: at least 25 questions, about 9 each, written by people who read the summaries (the spec requires human-written questions). Send 10 to the partner team with their `doc_id`s and corpus access. |
| Mon Nov 23 | BM25 and RRF fusion are written; check them on the full corpus with the team's questions |
| Thu Dec 3 | `measure_retrieval.py` on our and the partner team's questions; `measure_speed.py` (the `nprobe` sweep and queries per second; build times come from `data/build_times.json`); `measure_answers.py` and the team's grading of its sheet. |
| Dec 7–8 | README with the full run order, the 6–8 page report with failure analysis, and the live demo |

### Settled decisions

Don't reopen these or flag them as problems:
- **The `embeddings` primary key is `chunk_id` alone.** Every embedding uses the same model, so `model` is recorded but isn't part of the key, as in our proposal. The spec's table shows `(chunk_id, model)`, so the report should say why we differ.
- **`data/MovieSummaries.tar.gz` stays committed**, so a fresh clone can reproduce everything without a download. The extracted files, `movies.duckdb`, the index files and the measurement results (`build_times.json` and the `measure_*.py` CSVs) stay gitignored. `data/answers.csv` doesn't, because it holds the team's grades.
- **The layout is flat and lab-style:** the scripts above sit in the repo root, with no packages and no Makefile. The README lists the scripts in run order.
- **BM25 uses `rank_bm25`**, which the professor approved. Don't hand-roll it.
- **One chunking, windows of at most 256 tokens** (decided Nov 15). The proposal's chunk-size comparison is dropped; the comparisons are Flat vs. IVF and vector vs. BM25 vs. hybrid. The report should say why 256 tokens and that the plan changed.
- **One k, no LLM judge** (decided Nov 15). Recall and MRR are reported at k = 5 only (`TOP_K`), with no comparison across k values. Answer quality is human-graded (1–10 plus an explanation), which the spec allows as a "graded rubric"; the proposal's LLM judge is dropped.

## Formatting rules

`ingest.py` is the reference: every new Python script should look like it. Some of these rules break PEP 8 on purpose, so **never run black, ruff format, autopep8 or isort** on this repo. They would undo rules 1 and 2.

### 1. Imports go longest line first

Put all imports in one block under a `# Imports` comment.
- Sort them by the length of the whole line, longest first. Break ties alphabetically.
- Mix `from x import y` and `import x` lines together. Don't split them into standard-library, third-party and local groups.

```python
# Imports
from pathlib import Path
import pandas as pd
import duckdb
import html
import json
import csv
import re
```

### 2. Spaces around every `=`

Every `=` in Python code gets a space on both sides. That includes keyword arguments and default parameter values, not just assignments.

```python
def search(question, k = 10):
with open(EVAL_QUESTIONS, encoding = "utf-8") as f:
for row in csv.reader(f, delimiter = "\t", quoting = csv.QUOTE_NONE):
```

The rule doesn't apply inside strings, SQL or URLs (`?curid=74830`), or to version pins in `requirements.txt`.

### 3. requirements.txt

- Pin every package with `==`, one per line.
- Sort the lines longest first, breaking ties alphabetically.
- Add a package only when code actually imports it or the team uses its command. Four exceptions are pinned deliberately; don't remove them:
  - `torchvision` and `pillow`: `sentence-transformers` loads EmbeddingGemma 2's image processor even for text, and fails without them;
  - `pyarrow`: DuckDB's `to_arrow_table()` needs it, and `build_index.py` uses that to read the vectors back about 13× faster;
  - `transformers`: pinned to the version Lab 9 verified with the model.
- Keep `duckdb` and `duckdb-cli` on the same version, so the command-line tool can always open files the Python code wrote.

```
duckdb-cli==2.0.0
duckdb==2.0.0
pandas==3.1.1
```

### 4. Script layout

Every script has these parts, in this order (see the template at the end):

1. **A module docstring.** It says:
   - what the script does;
   - which files it reads and writes, listed and indented four spaces with descriptions lined up in a column;
   - what happens when it's run again;
   - how to run it, as `Run:` or `Run after <script>:` followed by the indented command.
2. **The `# Imports` block.**
3. **Constants**, in UPPER_SNAKE_CASE, grouped under short section comments such as `# Data paths` or `# Schema: ...`. Build every path from `DATA = Path(__file__).parent / "data"` so the script works from any working directory.
4. **Functions**, each with a `# Function to <what it does>` comment line directly above the `def`.
5. **`main()`**, with `# Main function` above it.
6. **The entry point**, with `# Run the main function if this script is executed directly.` above it.

Leave two blank lines between functions.

### 5. Comments and docstrings

- **Docstrings:** every function except `main()` has one.
  - Start with a one-sentence summary ending in a period, then a blank line.
  - Then give the details: input and output formats, edge cases, and a concrete example where it helps (`Example: "Orca{{'}}s" -> "Orca's"`).
- **Block comments:** inside a function, split the code into short blocks separated by blank lines. Put a comment above each block saying what it does and why, written as a sentence.
- **Arrows:** show transformations with `->`, for example `# blank -> NULL` or `# ([[Aishwarya , -> (Aishwarya ,`.
- **Trailing comments:** when consecutive lines have short trailing comments, line them up in one column.
- **No type hints.**

### 6. Strings, output and errors

- Use double quotes for strings. Use single quotes only where they're needed inside an f-string expression or a SQL string literal.
- Use f-strings for all formatted output.
  - Put thousands separators on counts (`{n_rows:,}`).
  - Use `:<` and `:>` widths for aligned tables.
- Split a long `print` into adjacent f-strings, with the second line lined up under the first quote.
- Scripts print their progress and a short summary of what they did, in the style of the labs: row counts, and what was skipped and why.
- When an input file has an expected kind of problem, stop with a clear message, not a traceback: `raise SystemExit(f"{EVAL_QUESTIONS.name} line {line_number}: {e}")`.
- Open text files with `encoding = "utf-8"`.

### 7. Names

- Use snake_case for functions and variables, and UPPER_SNAKE_CASE for constants.
- Name counts `n_<things>` (`n_rows`, `n_documents`).
- Give DataFrame names a `_df` ending (`documents_df`, `questions_df`).
- Call the DuckDB connection `con`.

### 8. SQL and DuckDB

- Write SQL keywords in upper case and function names in lower case: `SELECT count(*) FILTER (WHERE release_date IS NULL)`.
- Put multi-line SQL in a triple-quoted string, indented inside the call.
- In `CREATE TABLE` statements, line up column names, types and `--` comments in columns, and comment every column.
- Insert DataFrames in one statement: `INSERT INTO <table> BY NAME SELECT * FROM <name>_df`. Never insert rows one at a time with `executemany`. For the documents table that took 116 s, against 0.3 s for one statement.
- Use f-strings in SQL only for fixed names written in the code, such as table names. Pass values as `?` parameters.
- Only one process can write to `movies.duckdb` at a time. Close the connection with `con.close()` when you're done.

### Template for a new script

```python
"""One-sentence summary of what this script does.

Reads:
    data/movies.duckdb      documents table

and writes:
    data/example.json       what is in it

What happens on a re-run, and how to start over.

Run after ingest.py:
    python example.py
"""

# Imports
from pathlib import Path
import duckdb
import json

# Data paths
DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "movies.duckdb"
OUTPUT = DATA / "example.json"


# Function to count movies per genre
def count_genres(con, limit = 10):
    """Return the most common genres as {genre: n_movies}.

    Example: {"Drama": 19135, "Comedy": 10468, ...}
    """
    # One row per (movie, genre) pair, then count each genre.
    rows = con.execute("""
        SELECT genre, count(*) AS n_movies
        FROM (SELECT unnest(genres) AS genre FROM documents)
        GROUP BY genre
        ORDER BY n_movies DESC
        LIMIT ?
    """, [limit]).fetchall()
    return dict(rows)


# Main function
def main():
    con = duckdb.connect(DB_PATH, read_only = True)
    genres = count_genres(con)
    con.close()

    with open(OUTPUT, "w", encoding = "utf-8") as f:
        json.dump(genres, f, indent = 2)
    print(f"Wrote {len(genres)} genres to {OUTPUT.name}")


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
```
