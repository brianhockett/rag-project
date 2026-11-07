"""Load the CMU Movie Summary Corpus into DuckDB.

Reads the two files unzip.py extracted:
    data/MovieSummaries/plot_summaries.txt   wiki_id <TAB> plot summary
    data/MovieSummaries/movie.metadata.tsv   9 tab-separated columns, no header

and writes data/movies.duckdb:
    documents            one row per movie: id, title, release date,
                         genres, languages and the summary text
    eval_questions       loaded from data/eval_questions.jsonl, if it exists
    chunks, embeddings   created empty here; build_index.py fills them

The summaries file drives everything and the metadata is a lookup. 102 movies
are dropped: 99 summaries have no metadata row, so there is no title to cite
them by, and 3 are empty once their markup is removed (the whole "summary" was
a citation). Summaries and titles get two cleanups and nothing else: leftover
Wikipedia markup such as {{cite web}} is removed ({{'}} becomes an apostrophe),
and runs of whitespace are collapsed to one space. Each row records
SOURCE_VERSION, which becomes part of every chunk_id; bump it whenever the
cleaning changes. Tokenizing and embedding happen later in build_index.py.
Every later script reads from movies.duckdb, never from the raw files.

Safe to re-run. Documents are loaded once: if the table already has rows they
are left alone, so chunks and embeddings are never wiped (delete
data/movies.duckdb to start over). Eval questions are reloaded from the file on
every run, so after editing eval_questions.jsonl just run ingest.py again.

Run after unzip.py:
    python ingest.py
"""

# Imports
from pathlib import Path
import pandas as pd
import duckdb
import html
import json
import csv
import re

# Data paths
DATA = Path(__file__).parent / "data"
SUMMARIES = DATA / "MovieSummaries" / "plot_summaries.txt"
METADATA = DATA / "MovieSummaries" / "movie.metadata.tsv"
EVAL_QUESTIONS = DATA / "eval_questions.jsonl"
DB_PATH = DATA / "movies.duckdb"

# Source version: stored on every document and built into every chunk_id.
# Bump it whenever a change to clean_text or read_documents changes the summary
# text, so chunk ids from the new text never match ids from older evaluation runs.
SOURCE_VERSION = "v1"

# Schema: all four tables. IF NOT EXISTS makes re-running this a no-op.
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        BIGINT PRIMARY KEY,                    -- Wikipedia page ID, PK for the document
    title         VARCHAR NOT NULL,                      -- Movie title as given in the metadata file
    release_date  VARCHAR,                               -- Format varies by source; we don't parse it, just store it
    genres        VARCHAR[],                             -- List of genres
    languages     VARCHAR[],                             -- List of languages
    summary       VARCHAR NOT NULL,                      -- The plot summary text
    version       VARCHAR NOT NULL,                      -- Source version of the cleaned text (SOURCE_VERSION), part of every chunk_id
    source        VARCHAR,                               -- The link to the movie's wikipedia page: https://en.wikipedia.org/?curid=<doc_id>
    ingested_at   TIMESTAMP DEFAULT current_timestamp    -- When this row was inserted into the database
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  VARCHAR PRIMARY KEY,                       -- doc_id:version:config:ordinal, e.g. 74830:v1:tok256:3, PK for the chunk
    doc_id    BIGINT NOT NULL REFERENCES documents (doc_id), -- FK to the document this chunk belongs to
    config    VARCHAR NOT NULL,                          -- Chunk size: 'full' (whole summary) or 'tok<N>' (N-token windows), e.g. 'tok256'
    ordinal   INTEGER NOT NULL,                          -- Chunk number within the document, starting at 0
    text      VARCHAR NOT NULL,                          -- The chunk text, trimmed of whitespace
    n_tokens  INTEGER                                    -- Number of tokens in this chunk, computed by build_index.py
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  VARCHAR PRIMARY KEY REFERENCES chunks (chunk_id), -- doc_id:version:config:ordinal, PK, and FK to the chunk this embedding belongs to
    model     VARCHAR NOT NULL,                          -- e.g. google/embeddinggemma-2
    dim       INTEGER NOT NULL,                          -- Number of dimensions in the vector: 768 for EmbeddingGemma 2
    vector    FLOAT[768] NOT NULL                        -- embedding of "title: <documents.title> | text: <chunks.text>", unit length
);


CREATE TABLE IF NOT EXISTS eval_questions (
    q_id              VARCHAR PRIMARY KEY,               -- Evaluation question ID, e.g. "q1", PK for the question
    question          VARCHAR NOT NULL,                  -- The question text
    relevant_doc_ids  BIGINT[] NOT NULL,                 -- The correct document IDs that answer the question, e.g. [2345,67890]
    author            VARCHAR                            -- Team member who wrote the question
);
"""

# Text cleaning
# Templates that stand for punctuation. Every other {{...}} template is removed.
TEMPLATE_TEXT = {
    "{{spaced ndash}}": " – ",
    "{{emdash}}": "—",
    "{{mdash}}": "—",
    "{{'s}}": "'s",
    "{{'}}": "'",
    "{{’}}": "’",
}

# Function to clean up plot summaries and titles
def clean_text(text):
    """Remove leftover Wikipedia markup from a summary or title and collapse whitespace.

    Example: "the Orca{{'}}s mast.<ref name...|title...}}  {{cite web}} Brody"
          -> "the Orca's mast. Brody"
    """
    # HTML entities become characters: &mdash; -> —, &#39; -> ', &nbsp; -> non-breaking space
    text = html.unescape(text)

    # Templates that stand for punctuation become that punctuation: Orca{{'}}s -> Orca's
    for template, replacement in TEMPLATE_TEXT.items():
        text = text.replace(template, replacement)

    # Broken citations: <ref name...|title...}} or <ref name... ]
    # The pattern can't cross a { or }, so it can't run on into the next template.
    text = re.sub(r"<ref[^{}\]]{0,1000}?(?:\}\}|\])", " ", text)

    # Bare links: space.http://www.comingsoon.net/films.php?id=64232 -> space.
    text = re.sub(r"\[?https?://\S+", " ", text)

    # Every other template: {{cite web}}, {{Plot}}, {{Expand section}}.
    # Repeat until nothing changes, so nested templates like {{a {{b}} }} go too.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)

    # Stray brackets from broken links: ([[Aishwarya , -> (Aishwarya ,
    text = re.sub(r"\{\{|\}\}|\[\[|\]\]", "", text)

    # Collapse runs of whitespace (including non-breaking spaces) into one space.
    return " ".join(text.split())


# Function to read the movie metadata
def read_metadata():
    """Read movie.metadata.tsv into {doc_id: {"title", "release_date", "genres", "languages"}}.

    The file has no header. Each line has 9 tab-separated columns; we keep 5:
        0  Wikipedia movie ID   975900
        2  title                Ghosts of Mars
        3  release date         2001-08-24, 2000-02, 1938 or blank
        6  languages (JSON)     {"/m/02h40lc": "English Language"}
        8  genres (JSON)        {"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction"}
    The JSON columns map Freebase IDs to names; we keep only the names.
    """
    metadata = {}
    with open(METADATA, encoding = "utf-8") as f:
        # QUOTE_NONE: quote marks in titles are part of the title, not CSV quoting.
        for row in csv.reader(f, delimiter = "\t", quoting = csv.QUOTE_NONE):
            metadata[int(row[0])] = {
                "title": clean_text(row[2]),                        # [[Caltiki]] -> Caltiki
                "release_date": row[3] or None,                     # blank -> NULL
                "genres": list(json.loads(row[8]).values()),        # {} -> []
                "languages": list(json.loads(row[6]).values()),
            }
    return metadata


# Function to join the summaries to their metadata
def read_documents(metadata):
    """Join plot_summaries.txt to the metadata; return one row per movie as a DataFrame.

    Each line of the summaries file is "wiki_id<TAB>summary". A movie is dropped
    when it has no metadata row (no title to cite it by) or when its summary is
    empty after cleaning (the whole "summary" was a citation). The column names
    match the documents table, so the DataFrame can go straight into
    INSERT INTO documents BY NAME.
    """
    rows = []
    no_metadata = 0
    empty_summary = 0
    with open(SUMMARIES, encoding = "utf-8") as f:
        for line in f:
            wiki_id, text = line.rstrip("\n").split("\t", 1)
            doc_id = int(wiki_id)

            # Skip summaries with no metadata row.
            if doc_id not in metadata:
                no_metadata += 1
                continue

            # Skip summaries that are empty once the markup is removed.
            summary = clean_text(text)
            if not summary:
                empty_summary += 1
                continue

            movie = metadata[doc_id]
            rows.append({
                "doc_id": doc_id,
                "title": movie["title"],
                "release_date": movie["release_date"],
                "genres": movie["genres"],
                "languages": movie["languages"],
                "summary": summary,
                "version": SOURCE_VERSION,
                "source": f"https://en.wikipedia.org/?curid={doc_id}",
            })

    print(f"Read {len(rows):,} movies; dropped {no_metadata} with no metadata "
          f"and {empty_summary} with an empty summary after cleaning")
    return pd.DataFrame(rows)


# Function to load the evaluation questions
def load_eval_questions(con):
    """Replace the eval_questions table with the contents of eval_questions.jsonl.

    One JSON object per line:
        {"q_id": "q1", "question": "...", "relevant_doc_ids": [74830], "author": "Brian"}
    Blank lines and lines starting with # are skipped. Every relevant_doc_id must
    exist in documents, since a typo would otherwise just look like a retrieval
    miss. Does nothing if the file doesn't exist.
    """
    if not EVAL_QUESTIONS.exists():
        print(f"No {EVAL_QUESTIONS.name}; skipping eval questions")
        return

    # Read the questions, skipping blank and comment lines.
    questions = []
    with open(EVAL_QUESTIONS, encoding = "utf-8") as f:
        for line_number, line in enumerate(f, start = 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{EVAL_QUESTIONS.name} line {line_number}: {e}")

    # Check that every relevant document exists before touching the table.
    known_ids = {row[0] for row in con.execute("SELECT doc_id FROM documents").fetchall()}
    for q in questions:
        unknown = [d for d in q["relevant_doc_ids"] if d not in known_ids]
        if unknown:
            raise SystemExit(f"{q['q_id']}: relevant_doc_ids not in documents: {unknown}")

    # The file is the source of truth: clear the table, then insert what's in the file.
    # Both happen in one transaction, so a failed insert leaves the old questions in place.
    con.begin()
    con.execute("DELETE FROM eval_questions")
    if questions:
        questions_df = pd.DataFrame(questions)
        con.execute("INSERT INTO eval_questions BY NAME SELECT * FROM questions_df")
    con.commit()
    print(f"Loaded {len(questions)} eval questions")


# Function to print a summary of what's in the database
def print_report(con):
    """Print row counts, summary lengths, missing metadata and one random movie.

    These are the numbers for the corpus section of the final report.
    """
    print("\nRows per table:")
    for table in ["documents", "chunks", "embeddings", "eval_questions"]:
        n_rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<15} {n_rows:>7,}")

    # Summaries are single-spaced after cleaning, so splitting on " " counts words exactly.
    shortest, median, p90, longest, total = con.execute("""
        SELECT min(n), median(n), quantile_cont(n, 0.9), max(n), sum(n)
        FROM (SELECT len(string_split(summary, ' ')) AS n FROM documents)
    """).fetchone()
    print(f"\nSummary length in words: min {shortest}, median {median:.0f}, "
          f"90th percentile {p90:.0f}, max {longest:,} ({total:,} words in total)")

    no_date, no_genres = con.execute("""
        SELECT count(*) FILTER (WHERE release_date IS NULL),
               count(*) FILTER (WHERE len(genres) = 0)
        FROM documents
    """).fetchone()
    print(f"Missing metadata: {no_date:,} movies with no release date, {no_genres:,} with no genres")

    # A different movie each run, for a quick eyeball check.
    title, release_date, genres, source, summary = con.execute("""
        SELECT title, release_date, genres, source, summary FROM documents USING SAMPLE 1 ROWS
    """).fetchone()
    preview = summary if len(summary) <= 300 else summary[:300] + "..."
    print(f"\nRandom movie: {title} ({release_date or 'no date'}) {genres}\n  {source}\n  {preview}")


# Main function
def main():
    con = duckdb.connect(DB_PATH)
    con.execute(SCHEMA)

    # Documents are loaded once. Re-inserting would fail on the primary key, and
    # deleting them would mean deleting every chunk and embedding first.
    n_documents = con.execute("SELECT count(*) FROM documents").fetchone()[0]
    if n_documents:
        print(f"documents already has {n_documents:,} rows; leaving them alone "
              f"(delete {DB_PATH.name} to start over)")
    else:
        documents_df = read_documents(read_metadata())
        con.execute("INSERT INTO documents BY NAME SELECT * FROM documents_df")
        print(f"Inserted {len(documents_df):,} documents into {DB_PATH.name}")

    # Eval questions are reloaded every run, so edits to the file show up.
    load_eval_questions(con)

    print_report(con)
    con.close()


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
