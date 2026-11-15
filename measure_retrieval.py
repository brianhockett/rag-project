"""Measure retrieval quality: recall@5 and MRR for each search mode, on our questions and the partner team's.

Reads:
    data/movies.duckdb                  eval_questions, chunks and documents tables
    data/flat.faiss, data/ivf.faiss     both FAISS indexes, through movie_rag.Retriever
    data/chunk_ids.json, data/bm25.pkl  the chunk_id list and the BM25 index

and writes:
    data/retrieval_results.csv          recall@5, MRR and questions found, per question set and run
    data/retrieval_by_question.csv      each question's rank of the first correct movie, per run

Five runs: vector search on the exact (Flat) and IVF indexes, BM25 alone, and
hybrid (BM25 and vector, fused by RRF) on each index. Each run retrieves the
TOP_K chunks the LLM would get and dedupes them to movies, in rank order.
Metrics are at the document level, against eval_questions.relevant_doc_ids:
    recall@5    share of the question's correct movies among those movies
    MRR         mean over questions of 1 / rank of the first correct movie (0 if none)
Questions whose q_id starts with "p" are the partner team's, reported separately.

Reads the database without changing it; a re-run overwrites both CSV files.

Run after build_index.py:
    python measure_retrieval.py
"""

# Imports
from movie_rag import DATA, DB_PATH, TOP_K, Retriever, load_model
import pandas as pd
import duckdb

# Data paths
SUMMARY_CSV = DATA / "retrieval_results.csv"
BY_QUESTION_CSV = DATA / "retrieval_by_question.csv"

# Runs: (name, search mode, FAISS index). BM25 doesn't use FAISS; it just runs on the IVF Retriever.
RUNS = [
    ("vector-Flat", "vector", "flat"),
    ("vector-IVF", "vector", "ivf"),
    ("BM25", "keyword", "ivf"),
    ("hybrid-Flat", "hybrid", "flat"),
    ("hybrid-IVF", "hybrid", "ivf"),
]
RECALL = f"recall_at_{TOP_K}"                   # column name, e.g. recall_at_5


# Function to load the evaluation questions
def load_questions(con):
    """Return eval_questions as a DataFrame, with a question_set column of "ours" or "partner".

    The partner team's questions are the ones whose q_id starts with "p".
    """
    questions_df = con.execute("SELECT q_id, question, relevant_doc_ids FROM eval_questions ORDER BY q_id").df()
    if questions_df.empty:
        raise SystemExit("eval_questions is empty; write data/eval_questions.jsonl, then run ingest.py")
    questions_df["question_set"] = ["partner" if q_id.startswith("p") else "ours" for q_id in questions_df["q_id"]]
    return questions_df


# Function to score one question's retrieved chunks
def score(retrieved, relevant_doc_ids):
    """Return recall, reciprocal rank and rank for one question, at the document level.

    The chunks are deduped to movies in rank order first, so a movie with two
    chunks in the top 5 counts once. rank is the position of the first correct
    movie among those movies, or None when none is correct.
    Example: movies [1173339, 74830, 3556701], relevant [74830] -> recall 1.0, reciprocal rank 0.5, rank 2
    """
    movies = list(dict.fromkeys(c["doc_id"] for c in retrieved))
    relevant = {int(doc_id) for doc_id in relevant_doc_ids}
    found = [movie for movie in movies if movie in relevant]
    rank = movies.index(found[0]) + 1 if found else None
    return {"recall": len(found) / len(relevant), "reciprocal_rank": 1 / rank if rank else 0.0, "rank": rank}


# Function to run every question through every run
def evaluate(retrievers, questions_df):
    """Return one row per (question, run), with that run's recall, reciprocal rank and rank."""
    rows = []
    for name, mode, index in RUNS:
        for q in questions_df.itertuples():
            retrieved = retrievers[index].retrieve(q.question, TOP_K, mode)
            rows.append({"q_id": q.q_id, "question_set": q.question_set, "question": q.question, "run": name,
                         **score(retrieved, q.relevant_doc_ids)})
        print(f"  {name:<12} {len(questions_df)} questions")
    return pd.DataFrame(rows)


# Function to summarize each run
def summarize(results_df):
    """Return recall@k, MRR and the number of questions with a correct movie, per question set and run."""
    rows = []
    for question_set in ["ours", "partner"]:
        set_df = results_df[results_df["question_set"] == question_set]
        if set_df.empty:
            continue
        for name, _, _ in RUNS:
            run_df = set_df[set_df["run"] == name]
            rows.append({"question_set": question_set, "run": name, "n_questions": len(run_df),
                         RECALL: round(run_df["recall"].mean(), 3), "mrr": round(run_df["reciprocal_rank"].mean(), 3),
                         "n_found": int(run_df["rank"].notna().sum())})
    return pd.DataFrame(rows)


# Main function
def main():
    # Questions first, so an empty table stops the script before the slow loading.
    con = duckdb.connect(DB_PATH, read_only = True)
    questions_df = load_questions(con)
    con.close()
    n_partner = int((questions_df["question_set"] == "partner").sum())
    print(f"{len(questions_df)} questions: {len(questions_df) - n_partner} ours, {n_partner} partner")

    # One model, shared by a Retriever for each FAISS index.
    model = load_model()
    retrievers = {index: Retriever(index, model) for index in ["flat", "ivf"]}
    print(f"\n{len(RUNS)} runs, {TOP_K} chunks per question:")
    results_df = evaluate(retrievers, questions_df)

    # Summary table, one row per question set and run.
    summary_df = summarize(results_df)
    summary_df.to_csv(SUMMARY_CSV, index = False)

    # One row per question, one column per run, holding the rank of the first correct movie (blank if none).
    ranks_df = results_df.pivot(index = ["q_id", "question_set", "question"], columns = "run", values = "rank")
    ranks_df = ranks_df[[name for name, _, _ in RUNS]].astype("Int64").reset_index()
    ranks_df.to_csv(BY_QUESTION_CSV, index = False)

    print(f"\n{'Set':<9} {'Run':<12} {f'Recall@{TOP_K}':>9} {'MRR':>6} {'Found':>7}")
    for _, row in summary_df.iterrows():
        print(f"{row['question_set']:<9} {row['run']:<12} {row[RECALL]:>9.3f} {row['mrr']:>6.3f} "
              f"{f'{row.n_found}/{row.n_questions}':>7}")
    print(f"\nWrote {SUMMARY_CSV.name} and {BY_QUESTION_CSV.name}")


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
