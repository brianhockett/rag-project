"""Measure search speed: the nprobe sweep (ANN recall against speed) and queries per second per search run.

Reads:
    data/movies.duckdb                  eval_questions, chunks and documents tables
    data/flat.faiss, data/ivf.faiss     both FAISS indexes, through movie_rag.Retriever
    data/chunk_ids.json, data/bm25.pkl  the chunk_id list and the BM25 index

and writes:
    data/nprobe_sweep.csv               per nprobe: ANN recall@5 against the exact index, ms per query, queries per second
    data/query_speed.csv                per search run: ms per query and queries per second, query embedding included

The queries are the evaluation questions (ours and any partner team's), each
embedded once for the sweep. ANN recall@5 is the share of the exact (Flat)
top 5 chunks that IVF also returns. It needs no labels: it measures how well
IVF approximates brute force, not whether the chunks are right. Times are
for one query at a time, the way ask.py searches, averaged over several
passes after one untimed warm-up search. Index build times are in
data/build_times.json.

Reads the database without changing it; a re-run overwrites both CSV files.

Run after build_index.py:
    python measure_speed.py
"""

# Imports
from movie_rag import DATA, DB_PATH, NPROBE, TOP_K, Retriever, embed_query, load_model
from measure_retrieval import RUNS, load_questions
import pandas as pd
import numpy as np
import duckdb
import time

# Data paths
SWEEP_CSV = DATA / "nprobe_sweep.csv"
SPEED_CSV = DATA / "query_speed.csv"

# Settings
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]         # IVF lists searched per query, out of nlist (about 1,200)
FAISS_PASSES = 20                               # passes over the questions when timing FAISS alone (under 10 ms a search)
RETRIEVE_PASSES = 3                             # passes when timing full searches (BM25 alone takes 120-200 ms)
ANN_RECALL = f"ann_recall_at_{TOP_K}"           # column name, e.g. ann_recall_at_5


# Function to time single-query searches
def time_searches(search, queries, passes):
    """Return the mean milliseconds per call of search(query), over every query, repeated passes times.

    One untimed call comes first, so the first search's warm-up isn't counted.
    """
    search(queries[0])
    start = time.perf_counter()
    for _ in range(passes):
        for query in queries:
            search(query)
    return (time.perf_counter() - start) * 1000 / (passes * len(queries))


# Function to sweep nprobe
def sweep_nprobe(flat, ivf, vectors):
    """Return one row for Flat and one per nprobe: ANN recall@k against Flat, ms per query, queries per second.

    vectors holds one embedded question per row. FAISS alone is timed, so the
    rows compare the indexes, not the embedding model.
    """
    # The exact top k for every question, for IVF to be compared with.
    _, exact_ids = flat.search(vectors, TOP_K)
    flat_ms = time_searches(lambda vector: flat.search(vector[None], TOP_K), vectors, FAISS_PASSES)
    rows = [{"index": "Flat", "nprobe": None, ANN_RECALL: 1.0,
             "ms_per_query": round(flat_ms, 3), "queries_per_second": round(1000 / flat_ms)}]

    # nprobe can't usefully exceed the number of lists, which matters for small test indexes.
    for nprobe in [n for n in NPROBES if n <= ivf.nlist]:
        ivf.nprobe = nprobe
        _, ivf_ids = ivf.search(vectors, TOP_K)
        recall = np.mean([len(set(found) & set(exact)) / TOP_K for found, exact in zip(ivf_ids, exact_ids)])
        ms = time_searches(lambda vector: ivf.search(vector[None], TOP_K), vectors, FAISS_PASSES)
        rows.append({"index": "IVF", "nprobe": nprobe, ANN_RECALL: round(float(recall), 3),
                     "ms_per_query": round(ms, 3), "queries_per_second": round(1000 / ms)})
    ivf.nprobe = NPROBE

    sweep_df = pd.DataFrame(rows)
    sweep_df["nprobe"] = sweep_df["nprobe"].astype("Int64")
    return sweep_df


# Function to time each search run end to end
def time_runs(retrievers, questions):
    """Return ms per query and queries per second for each run in RUNS, embedding and DuckDB lookup included."""
    rows = []
    for name, mode, index in RUNS:
        ms = time_searches(lambda question: retrievers[index].retrieve(question, TOP_K, mode), questions, RETRIEVE_PASSES)
        rows.append({"run": name, "ms_per_query": round(ms, 1), "queries_per_second": round(1000 / ms, 1)})
        print(f"  {name:<12} {ms:7.1f} ms")
    return pd.DataFrame(rows)


# Main function
def main():
    con = duckdb.connect(DB_PATH, read_only = True)
    questions = load_questions(con)["question"].tolist()
    con.close()

    # One model, shared by a Retriever for each FAISS index; the IVF one has nprobe = NPROBE.
    model = load_model()
    retrievers = {index: Retriever(index, model) for index in ["flat", "ivf"]}
    ivf = retrievers["ivf"].index
    print(f"{len(questions)} questions; IVF index of {ivf.ntotal:,} vectors in {ivf.nlist:,} lists")

    # The sweep: FAISS alone, on questions embedded once.
    vectors = np.vstack([embed_query(model, question) for question in questions])
    sweep_df = sweep_nprobe(retrievers["flat"].index, ivf, vectors)
    sweep_df.to_csv(SWEEP_CSV, index = False)

    print(f"\n{'Index':<6} {'nprobe':>6} {f'ANN recall@{TOP_K}':>14} {'ms/query':>9} {'queries/s':>10}")
    for _, row in sweep_df.iterrows():
        nprobe = "-" if pd.isna(row["nprobe"]) else row["nprobe"]
        marker = "   <- NPROBE now" if nprobe == NPROBE else ""
        print(f"{row['index']:<6} {nprobe:>6} {row[ANN_RECALL]:>14.3f} {row['ms_per_query']:>9.3f} "
              f"{row['queries_per_second']:>10,}{marker}")

    # Whole searches, the way ask.py runs them.
    print(f"\nFull searches, {RETRIEVE_PASSES} passes over the questions:")
    speed_df = time_runs(retrievers, questions)
    speed_df.to_csv(SPEED_CSV, index = False)

    print(f"\n{'Run':<12} {'ms/query':>9} {'queries/s':>10}")
    for _, row in speed_df.iterrows():
        print(f"{row['run']:<12} {row['ms_per_query']:>9.1f} {row['queries_per_second']:>10.1f}")
    print(f"\nWrote {SWEEP_CSV.name} and {SPEED_CSV.name}")


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
