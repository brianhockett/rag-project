"""Chunk the summaries, embed the chunks, and build the FAISS and BM25 indexes.

Reads:
    data/movies.duckdb      documents table

and writes:
    data/movies.duckdb      chunks and embeddings tables
    data/flat.faiss         exact index (IndexFlatIP), the baseline
    data/ivf.faiss          IVF index (IndexIVFFlat), the product
    data/chunk_ids.json     chunk_id of each FAISS id and BM25 row, in order
    data/bm25.pkl           BM25 index over each chunk's title, text and genres
    data/build_times.json   one record per run: each step's time, CPU threads, and the GPU's name and memory

Every chunk is a window of at most 256 of the embedding model's tokens, with
32 tokens of overlap. Embedding is the slow step (about 20 minutes for the
whole corpus on a laptop GPU), so it saves its work every SAVE_EVERY chunks.
Embedding runs on the GPU when there is one; FAISS (faiss-cpu) and BM25 run
on the CPU.

Safe to re-run, and a re-run resumes: chunks are made once, only chunks
without an embedding get embedded, and the FAISS and BM25 files are always
rebuilt from the tables, which takes about a minute. To re-chunk, delete the
rows from embeddings and then from chunks. Each run adds its record to
build_times.json, with skipped steps marked, so the full build's times are
in the record of the run that did all the work.

Run after ingest.py:
    python build_index.py
"""

# Imports
from movie_rag import BM25_PATH, CHUNK_IDS_PATH, DATA, DB_PATH, DIM, INDEX_PATHS, MODEL_ID, embed_documents, load_model, tokenize
from rank_bm25 import BM25Okapi
from datetime import datetime
import pandas as pd
import numpy as np
import duckdb
import pickle
import faiss
import torch
import json
import math
import time
import os

# Data paths: the timing log; the index paths come from movie_rag.py.
BUILD_TIMES = DATA / "build_times.json"

# Chunking: windows over the embedding model's own tokens, so 256 means 256 to the model too.
WINDOW_TOKENS = 256                             # most tokens in one chunk
OVERLAP_TOKENS = 32                             # tokens shared with the previous chunk
CHUNKING = "tok256"                             # names the chunking in every chunk_id; change it if the numbers above change

# Embedding: 256-token chunks run at 70/s in GPU batches of 64 (2.6 GB in bfloat16 on an 8 GB RTX 4070 laptop).
EMBED_BATCH = 64                                # chunks per GPU batch
SAVE_EVERY = 2048                               # chunks embedded between inserts, so a crash loses little

# IVF index: nlist is about 4 * sqrt(N), but FAISS wants at least 39 training vectors per list.
MIN_VECTORS_PER_LIST = 39


# Function to cut one summary's tokens into windows
def split_tokens(ids, tokenizer):
    """Return (text, n_tokens) for each window over a summary's token ids.

    Windows hold at most WINDOW_TOKENS tokens and overlap by OVERLAP_TOKENS,
    so a sentence cut at one window's end is whole in the next. A summary
    shorter than one window is one chunk. n_tokens counts the stored text's
    own tokens, because decoding a window and encoding it again can add a
    token or two where the window cuts a word.
    Example: 600 tokens -> windows [0, 256), [224, 480), [448, 600)
    """
    pieces = []
    start = 0
    while True:
        end = min(start + WINDOW_TOKENS, len(ids))

        # A window whose text re-encodes to too many tokens loses the overflow from its end, until it fits.
        while True:
            text = tokenizer.decode(ids[start:end], skip_special_tokens = True).strip()
            n_tokens = len(tokenizer(text, add_special_tokens = False)["input_ids"])
            if n_tokens <= WINDOW_TOKENS:
                break
            end -= n_tokens - WINDOW_TOKENS
        pieces.append((text, n_tokens))

        # The window that reaches the end of the summary is the last.
        if end == len(ids):
            return pieces
        start = end - OVERLAP_TOKENS


# Function to chunk every document
def chunk_documents(documents_df, tokenizer):
    """Return one row per chunk as a DataFrame whose columns match the chunks table.

    chunk_id is doc_id:version:CHUNKING:ordinal, e.g. 74830:v1:tok256:3.
    """
    # One batched call tokenizes every summary; the fast tokenizer runs it in parallel.
    all_ids = tokenizer(documents_df["summary"].tolist(), add_special_tokens = False)["input_ids"]

    rows = []
    for doc_id, version, ids in zip(documents_df["doc_id"], documents_df["version"], all_ids):
        for ordinal, (text, n_tokens) in enumerate(split_tokens(ids, tokenizer)):
            rows.append({"chunk_id": f"{doc_id}:{version}:{CHUNKING}:{ordinal}", "doc_id": doc_id,
                         "ordinal": ordinal, "text": text, "n_tokens": n_tokens})
    return pd.DataFrame(rows)


# Function to make the chunks, unless they exist already
def make_chunks(con, documents_df, tokenizer):
    """Fill the chunks table in one statement; skip it if the table has chunks already.

    Returns the step's record for build_times.json.
    Example: {"seconds": 171.4, "n_chunks": 90210}
    """
    n_chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if n_chunks:
        print(f"{n_chunks:,} chunks already in {DB_PATH.name}")
        return {"skipped": True, "n_chunks": n_chunks}

    start = time.perf_counter()
    chunks_df = chunk_documents(documents_df, tokenizer)
    con.execute("INSERT INTO chunks BY NAME SELECT * FROM chunks_df")
    seconds = time.perf_counter() - start
    print(f"{len(chunks_df):,} chunks from {len(documents_df):,} documents in "
          f"{seconds:.0f} s (median {chunks_df['n_tokens'].median():.0f} tokens, "
          f"max {chunks_df['n_tokens'].max():,})")
    return {"seconds": round(seconds, 2), "n_chunks": len(chunks_df)}


# Function to embed every chunk that has no embedding yet
def embed_missing(con, model):
    """Embed the chunks that have no embedding yet, longest first, saving every SAVE_EVERY chunks.

    Longest first means a batch too big for the GPU fails in the first
    minute, not the last. Returns the step's record for build_times.json;
    after a crash, the resumed run's record covers only the chunks it embedded.
    Example: {"seconds": 1288.0, "n_chunks": 90210, "chunks_per_second": 70.0}
    """
    missing_df = con.execute("""
        SELECT c.chunk_id, d.title, c.text
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE e.chunk_id IS NULL
        ORDER BY c.n_tokens DESC, c.chunk_id
    """).df()
    if missing_df.empty:
        print("Every chunk is embedded already")
        return {"skipped": True}

    print(f"Embedding {len(missing_df):,} chunks with {MODEL_ID} on {model.device}")
    start = time.perf_counter()
    for first in range(0, len(missing_df), SAVE_EVERY):
        group_df = missing_df.iloc[first:first + SAVE_EVERY]
        vectors = embed_documents(model, group_df["title"].tolist(), group_df["text"].tolist(), EMBED_BATCH)

        # One insert per group, so a crash loses at most SAVE_EVERY chunks of work.
        embeddings_df = pd.DataFrame({"chunk_id": group_df["chunk_id"].tolist(), "model": MODEL_ID,
                                      "dim": DIM, "vector": list(vectors)})
        con.execute("INSERT INTO embeddings BY NAME SELECT * FROM embeddings_df")
        n_done = first + len(group_df)
        rate = n_done / (time.perf_counter() - start)
        print(f"  {n_done:>7,} / {len(missing_df):,} chunks  {rate:5.0f} chunks/s  "
              f"about {(len(missing_df) - n_done) / rate / 60:.0f} min left")
    seconds = time.perf_counter() - start
    print(f"Embedded {len(missing_df):,} chunks in {seconds / 60:.1f} min")

    return {"seconds": round(seconds, 2), "n_chunks": len(missing_df),
            "chunks_per_second": round(len(missing_df) / seconds, 1)}


# Function to build and save the FAISS indexes
def build_faiss(con):
    """Build the exact and IVF indexes from the embeddings table; save them with the chunk_id list.

    Vectors go in ORDER BY chunk_id, so FAISS id i is chunk_ids[i]. The BM25
    index is built in the same order and shares the list. Returns the
    chunk_ids and the step's record for build_times.json.
    """
    # Arrow hands back the vectors as one float32 block: 2 s for 100,000, against 24 s as Python tuples.
    start = time.perf_counter()
    table = con.sql("""
        SELECT chunk_id, vector
        FROM embeddings
        ORDER BY chunk_id
    """).to_arrow_table()
    chunk_ids = table.column("chunk_id").to_pylist()
    vectors = np.ascontiguousarray(table.column("vector").combine_chunks().flatten().to_numpy().reshape(-1, DIM))
    n_vectors = len(vectors)
    read_seconds = time.perf_counter() - start

    # Exact baseline: every query is compared with every vector.
    start = time.perf_counter()
    flat = faiss.IndexFlatIP(DIM)
    flat.add(vectors)
    flat_seconds = time.perf_counter() - start

    # IVF: k-means into nlist lists (training), then each vector goes into the list of its nearest centroid.
    nlist = max(1, min(round(4 * math.sqrt(n_vectors)), n_vectors // MIN_VECTORS_PER_LIST))
    quantizer = faiss.IndexFlatIP(DIM)
    ivf = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    start = time.perf_counter()
    ivf.train(vectors)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    ivf.add(vectors)
    add_seconds = time.perf_counter() - start

    # Save both indexes and the list that maps FAISS ids back to chunk_ids.
    faiss.write_index(flat, str(INDEX_PATHS["flat"]))
    faiss.write_index(ivf, str(INDEX_PATHS["ivf"]))
    with open(CHUNK_IDS_PATH, "w", encoding = "utf-8") as f:
        json.dump(chunk_ids, f)
    print(f"Flat index of {n_vectors:,} vectors in {flat_seconds:.1f} s; "
          f"IVF with nlist {nlist} trained in {train_seconds:.1f} s and filled in {add_seconds:.1f} s")

    record = {"n_vectors": n_vectors, "read_vectors_seconds": round(read_seconds, 2),
              "flat_seconds": round(flat_seconds, 2),
              "ivf_train_seconds": round(train_seconds, 2), "ivf_add_seconds": round(add_seconds, 2)}
    return chunk_ids, record


# Function to build and save the BM25 index
def build_bm25(con, chunk_ids):
    """Build BM25 over each chunk's title, text and genres, in chunk_ids order, and pickle it.

    The title is included because summaries rarely name their own movie, so a
    question like "In Jaws, ..." could not otherwise match on "Jaws". Returns
    the step's record for build_times.json.
    """
    rows = con.execute("""
        SELECT c.chunk_id, d.title, c.text, array_to_string(d.genres, ' ')
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY c.chunk_id
    """).fetchall()
    if [row[0] for row in rows] != chunk_ids:
        raise SystemExit("Some chunks have no embedding; run build_index.py again to finish embedding")

    start = time.perf_counter()
    bm25 = BM25Okapi([tokenize(f"{title} {text} {genres}") for _, title, text, genres in rows])
    seconds = time.perf_counter() - start
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 over {len(rows):,} chunks in {seconds:.0f} s ({BM25_PATH.stat().st_size / 1e6:.0f} MB pickled)")
    return {"seconds": round(seconds, 2), "n_chunks": len(rows)}


# Function to describe the machine doing the build
def describe_hardware(model):
    """Return the CPU's thread count and, when embedding runs on an NVIDIA GPU, its name and memory.

    The GPU fields are None without one.
    Example: {"cpu_threads": 20, "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "gpu_memory_gb": 8.6}
    """
    hardware = {"cpu_threads": os.cpu_count(), "gpu": None, "gpu_memory_gb": None}
    if model.device.type == "cuda":
        properties = torch.cuda.get_device_properties(model.device)
        hardware["gpu"] = properties.name
        hardware["gpu_memory_gb"] = round(properties.total_memory / 1e9, 1)
    return hardware


# Function to add this run's record to the timing log
def save_timings(run):
    """Append one run's record to build_times.json, a JSON list with one record per run, oldest first.

    Earlier runs are kept, so builds on different machines or settings can be compared.
    """
    runs = []
    if BUILD_TIMES.exists():
        try:
            with open(BUILD_TIMES, encoding = "utf-8") as f:
                runs = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{BUILD_TIMES.name} isn't valid JSON ({e}); fix or delete it, then run again")
    runs.append(run)
    with open(BUILD_TIMES, "w", encoding = "utf-8") as f:
        json.dump(runs, f, indent = 2)
    print(f"Timings saved to {BUILD_TIMES.name} (run {len(runs)})")


# Main function
def main():
    run_start = time.perf_counter()
    started_at = datetime.now().isoformat(timespec = "seconds")
    con = duckdb.connect(DB_PATH)

    # A database from before Nov 15 has a config column in chunks, so it can't take these rows.
    if "config" in [row[0] for row in con.execute("DESCRIBE chunks").fetchall()]:
        raise SystemExit(f"{DB_PATH.name} has the old chunks table with a config column. "
                         f"Delete it, then run ingest.py and build_index.py again.")

    documents_df = con.execute("SELECT doc_id, version, title, summary FROM documents ORDER BY doc_id").df()
    print(f"Read {len(documents_df):,} documents from {DB_PATH.name}")

    start = time.perf_counter()
    model = load_model()
    load_seconds = time.perf_counter() - start
    print(f"Loaded {MODEL_ID} on {model.device} in {load_seconds:.0f} s\n")

    # Each step returns its own record; together they make this run's entry in build_times.json.
    run = {"started_at": started_at, "hardware": describe_hardware(model),
           "model_load_seconds": round(load_seconds, 2)}
    run["chunking"] = make_chunks(con, documents_df, model.tokenizer)
    run["embedding"] = embed_missing(con, model)
    chunk_ids, run["faiss"] = build_faiss(con)
    run["bm25"] = build_bm25(con, chunk_ids)

    print("\nRows per table:")
    for table in ["documents", "chunks", "embeddings"]:
        n_rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<15} {n_rows:>7,}")
    con.close()

    run["total_seconds"] = round(time.perf_counter() - run_start, 2)
    print(f"\nTotal {run['total_seconds'] / 60:.1f} min")
    save_timings(run)


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
