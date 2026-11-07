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
- Add a package only when code actually imports it or the team uses its command.
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
