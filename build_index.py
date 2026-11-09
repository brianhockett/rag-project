"""Chunk the summaries, embed the chunks, and build the FAISS and BM25 indexes.

Reads:
    data/movies.duckdb              documents table

and writes:
    data/movies.duckdb              chunks and embeddings tables, for both chunk configs
    data/<config>_flat.faiss        exact index (IndexFlatIP), the baseline
    data/<config>_ivf.faiss         IVF index (IndexIVFFlat), the product
    data/<config>_chunk_ids.json    chunk_id of each FAISS id and BM25 row, in order
    data/<config>_bm25.pkl          BM25 index over each chunk's title, text and genres

There are two chunk configs: "full" (one chunk per summary) and "tok256"
(256-token windows with 32 tokens of overlap). Embedding is the slow step
(over an hour for the whole corpus on a laptop GPU), so it saves its work
every SAVE_EVERY chunks.

Safe to re-run, and a re-run resumes: a config's chunks are made once, only
chunks without an embedding get embedded, and the FAISS and BM25 files are
always rebuilt from the tables, which takes about a minute. To re-chunk a
config, delete its rows from embeddings and then from chunks.

Run after ingest.py:
    python build_index.py
"""

# Imports
from movie_rag import DATA, DB_PATH, DIM, MODEL_ID, embed_documents, load_model, tokenize
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
import duckdb
import pickle
import faiss
import json
import math
import time

# Chunk configs
#CONFIGS = ["full", "tok256"]
CONFIGS = ['tok256'] # testing
WINDOW_TOKENS = 256                             # tok256: tokens per window
OVERLAP_TOKENS = 32                             # tok256: tokens shared with the previous window

# Embedding: chunks per GPU batch by chunk length. Measured in bfloat16 on an 8 GB RTX 4070 laptop:
# 256-token chunks run at 70/s in batches of 64 (2.6 GB); whole summaries at 32/s in batches of 8;
# the 203 summaries over 2,048 tokens at about 2/s, however they are batched.
EMBED_BATCHES = [(300, 64), (2048, 8), (math.inf, 2)]   # (up to this many tokens, chunks per GPU batch)
SAVE_EVERY = 2048                               # chunks embedded between inserts, so a crash loses little

# IVF index: nlist is about 4 * sqrt(N), but FAISS wants at least 39 training vectors per list.
MIN_VECTORS_PER_LIST = 39


# Function to cut one summary's tokens into windows
def split_tokens(ids, tokenizer):
    """Return the text of each WINDOW_TOKENS window over a summary's token ids.

    Windows overlap by OVERLAP_TOKENS, so a sentence cut at one window's end
    is whole in the next. A summary shorter than one window is one chunk.
    Example: 600 tokens -> windows [0, 256), [224, 480), [448, 600)
    """
    pieces = []
    start = 0
    while True:
        end = min(start + WINDOW_TOKENS, len(ids))
        pieces.append(tokenizer.decode(ids[start:end], skip_special_tokens = True).strip())
        if end == len(ids):
            return pieces
        start = end - OVERLAP_TOKENS


# Function to chunk every document
def chunk_documents(documents_df, config, tokenizer):
    """Return one row per chunk as a DataFrame whose columns match the chunks table.

    chunk_id is doc_id:version:config:ordinal, e.g. 74830:v1:tok256:3.
    """
    # Windows are over the model's own tokens, so 256 means 256 to the model too.
    # One batched call tokenizes every summary; the fast tokenizer runs it in parallel.
    summaries = documents_df["summary"].tolist()
    if config == "full":
        pieces_per_doc = [[summary] for summary in summaries]
    else:
        all_ids = tokenizer(summaries, add_special_tokens = False)["input_ids"]
        pieces_per_doc = [split_tokens(ids, tokenizer) for ids in all_ids]

    rows = []
    for doc_id, version, pieces in zip(documents_df["doc_id"], documents_df["version"], pieces_per_doc):
        for ordinal, text in enumerate(pieces):
            rows.append({"chunk_id": f"{doc_id}:{version}:{config}:{ordinal}", "doc_id": doc_id,
                         "config": config, "ordinal": ordinal, "text": text})
    chunks_df = pd.DataFrame(rows)

    # Count the stored text's tokens: decoding and re-encoding can move a window by a token or two.
    all_ids = tokenizer(chunks_df["text"].tolist(), add_special_tokens = False)["input_ids"]
    chunks_df["n_tokens"] = [len(ids) for ids in all_ids]
    return chunks_df


# Function to make one config's chunks, unless they exist already
def make_chunks(con, config, documents_df, tokenizer):
    """Fill the chunks table for one config, in one statement; skip a config that has chunks."""
    n_chunks = con.execute("SELECT count(*) FROM chunks WHERE config = ?", [config]).fetchone()[0]
    if n_chunks:
        print(f"{config}: {n_chunks:,} chunks already in {DB_PATH.name}")
        return

    start = time.perf_counter()
    chunks_df = chunk_documents(documents_df, config, tokenizer)
    con.execute("INSERT INTO chunks BY NAME SELECT * FROM chunks_df")
    print(f"{config}: {len(chunks_df):,} chunks from {len(documents_df):,} documents in "
          f"{time.perf_counter() - start:.0f} s (median {chunks_df['n_tokens'].median():.0f} tokens, "
          f"max {chunks_df['n_tokens'].max():,})")


# Function to embed every chunk that has no embedding yet
def embed_missing(con, config, model):
    """Embed one config's unembedded chunks, longest first, saving every SAVE_EVERY chunks.

    Each length band in EMBED_BATCHES gets its own GPU batch size. Longest
    first means a batch too big for the GPU fails in the first minute, not
    the last.
    """
    missing_df = con.execute("""
        SELECT c.chunk_id, d.title, c.text, c.n_tokens
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE c.config = ? AND e.chunk_id IS NULL
        ORDER BY c.n_tokens DESC, c.chunk_id
    """, [config]).df()
    if missing_df.empty:
        print(f"{config}: every chunk is embedded already")
        return

    print(f"{config}: embedding {len(missing_df):,} chunks with {MODEL_ID} on {model.device}")
    start = time.perf_counter()
    n_done = 0

    # Length bands from EMBED_BATCHES, longest band first: (2048, inf] -> (512, 2048] -> [0, 512]
    lowers = [-1] + [upper for upper, _ in EMBED_BATCHES[:-1]]
    for lower, (upper, batch_size) in reversed(list(zip(lowers, EMBED_BATCHES))):
        band_df = missing_df[(missing_df["n_tokens"] > lower) & (missing_df["n_tokens"] <= upper)]
        for first in range(0, len(band_df), SAVE_EVERY):
            group_df = band_df.iloc[first:first + SAVE_EVERY]
            vectors = embed_documents(model, group_df["title"].tolist(), group_df["text"].tolist(), batch_size)

            # One insert per group, so a crash loses at most SAVE_EVERY chunks of work.
            embeddings_df = pd.DataFrame({"chunk_id": group_df["chunk_id"].tolist(), "model": MODEL_ID,
                                          "dim": DIM, "vector": list(vectors)})
            con.execute("INSERT INTO embeddings BY NAME SELECT * FROM embeddings_df")
            n_done += len(group_df)
            rate = n_done / (time.perf_counter() - start)
            print(f"  {n_done:>7,} / {len(missing_df):,} chunks  {rate:5.0f} chunks/s  "
                  f"about {(len(missing_df) - n_done) / rate / 60:.0f} min left")
    print(f"{config}: embedded {n_done:,} chunks in {(time.perf_counter() - start) / 60:.1f} min")


# Function to build and save one config's FAISS indexes
def build_faiss(con, config):
    """Build the exact and IVF indexes from the embeddings table; save them with the chunk_id list.

    Vectors go in ORDER BY chunk_id, so FAISS id i is chunk_ids[i]. The BM25
    index is built in the same order and shares the list.
    """
    # Arrow hands back the vectors as one float32 block: 2 s for 100,000, against 24 s as Python tuples.
    table = con.sql("""
        SELECT e.chunk_id, e.vector
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE c.config = ?
        ORDER BY e.chunk_id
    """, params = [config]).to_arrow_table()
    chunk_ids = table.column("chunk_id").to_pylist()
    vectors = np.ascontiguousarray(table.column("vector").combine_chunks().flatten().to_numpy().reshape(-1, DIM))
    n_vectors = len(vectors)

    # Exact baseline: every query is compared with every vector.
    start = time.perf_counter()
    flat = faiss.IndexFlatIP(DIM)
    flat.add(vectors)
    flat_seconds = time.perf_counter() - start

    # IVF: k-means into nlist lists, then each vector goes into the list of its nearest centroid.
    nlist = max(1, min(round(4 * math.sqrt(n_vectors)), n_vectors // MIN_VECTORS_PER_LIST))
    start = time.perf_counter()
    quantizer = faiss.IndexFlatIP(DIM)
    ivf = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)
    ivf_seconds = time.perf_counter() - start

    # Save both indexes and the list that maps FAISS ids back to chunk_ids.
    faiss.write_index(flat, str(DATA / f"{config}_flat.faiss"))
    faiss.write_index(ivf, str(DATA / f"{config}_ivf.faiss"))
    with open(DATA / f"{config}_chunk_ids.json", "w", encoding = "utf-8") as f:
        json.dump(chunk_ids, f)
    print(f"{config}: Flat index of {n_vectors:,} vectors in {flat_seconds:.1f} s; "
          f"IVF with nlist {nlist} trained and filled in {ivf_seconds:.1f} s")
    return chunk_ids


# Function to build and save one config's BM25 index
def build_bm25(con, config, chunk_ids):
    """Build BM25 over each chunk's title, text and genres, in chunk_ids order, and pickle it.

    The title is included because summaries rarely name their own movie, so a
    question like "In Jaws, ..." could not otherwise match on "Jaws".
    """
    rows = con.execute("""
        SELECT c.chunk_id, d.title, c.text, array_to_string(d.genres, ' ')
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.config = ?
        ORDER BY c.chunk_id
    """, [config]).fetchall()
    if [row[0] for row in rows] != chunk_ids:
        raise SystemExit(f"{config}: some chunks have no embedding; run build_index.py again to finish embedding")

    start = time.perf_counter()
    bm25 = BM25Okapi([tokenize(f"{title} {text} {genres}") for _, title, text, genres in rows])
    seconds = time.perf_counter() - start
    path = DATA / f"{config}_bm25.pkl"
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"{config}: BM25 over {len(rows):,} chunks in {seconds:.0f} s ({path.stat().st_size / 1e6:.0f} MB pickled)")


# Main function
def main():
    con = duckdb.connect(DB_PATH)
    documents_df = con.execute("SELECT doc_id, version, title, summary FROM documents ORDER BY doc_id").df()
    print(f"Read {len(documents_df):,} documents from {DB_PATH.name}")

    start = time.perf_counter()
    model = load_model()
    print(f"Loaded {MODEL_ID} on {model.device} in {time.perf_counter() - start:.0f} s\n")

    for config in CONFIGS:
        make_chunks(con, config, documents_df, model.tokenizer)
        embed_missing(con, config, model)
        chunk_ids = build_faiss(con, config)
        build_bm25(con, config, chunk_ids)
        print()

    print("Rows per table:")
    for table in ["documents", "chunks", "embeddings"]:
        n_rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<15} {n_rows:>7,}")
    con.close()


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
