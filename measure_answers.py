"""Run the real system on every evaluation question and write a sheet for the team to grade.

Reads:
    data/movies.duckdb      eval_questions, chunks and documents tables
    data/ivf.faiss          the IVF index, with chunk_ids.json and bm25.pkl, through movie_rag.Retriever

and writes:
    data/answers.csv        one row per question: the question, its correct and retrieved movies,
                            recall@5, reciprocal rank, the LLM's answer, and blank score and
                            explanation columns for a person to fill in

The system is the one ask.py runs: hybrid search on the IVF index, TOP_K
chunks, Lab 10's prompt, and the LLM at LLM_BASE_URL. Graders give each
answer a score from 1 to 10 and explain it; the failure analysis is written
from the low scores. Questions whose q_id starts with "p" are the partner
team's.

Needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, in .env or the shell. Won't
overwrite answers.csv, because by then it may hold the team's grades: rename
or delete it to make a new one.

Run after build_index.py:
    python measure_answers.py
"""

# Imports
from movie_rag import DATA, DB_PATH, TOP_K, Retriever, build_prompt, chat
from measure_retrieval import RECALL, load_questions, score
import pandas as pd
import duckdb
import time
import os

# Data paths
ANSWERS_CSV = DATA / "answers.csv"

# The search ask.py uses: hybrid, on the IVF index that Retriever loads by default.
MODE = "hybrid"


# Function to list movies the way the sheet shows them
def describe_movies(movies):
    """Return movies as one readable string, in order: "Title (year) [doc_id]; ...".

    movies is a list of (doc_id, title, year) tuples; year may be None.
    Example: [(74830, "Jaws", "1975")] -> "Jaws (1975) [74830]"
    """
    return "; ".join(f"{title} ({year or 'no date'}) [{doc_id}]" for doc_id, title, year in movies)


# Main function
def main():
    # Check both ways this can go wrong before loading anything.
    if ANSWERS_CSV.exists():
        raise SystemExit(f"{ANSWERS_CSV.name} already exists and may hold the team's grades; "
                         f"rename or delete it to make a new one")
    if not os.environ.get("LLM_BASE_URL"):
        raise SystemExit("Put LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env first (copy .env.example): answers need the LLM")

    # The questions, and the titles of their correct movies.
    con = duckdb.connect(DB_PATH, read_only = True)
    questions_df = load_questions(con)
    titles = {row[0]: row for row in con.execute("""
        SELECT doc_id, title, left(release_date, 4)
        FROM documents
        WHERE doc_id IN (SELECT unnest(relevant_doc_ids) FROM eval_questions)
    """).fetchall()}
    con.close()

    retriever = Retriever()
    print(f"Answering {len(questions_df)} questions with {os.environ.get('LLM_MODEL') or 'the default model'}:")
    rows = []
    for q in questions_df.itertuples():
        retrieved = retriever.retrieve(q.question, TOP_K, MODE)

        # A failed request stops the run, so the sheet never mixes answers with errors.
        start = time.perf_counter()
        try:
            reply = chat(build_prompt(q.question, retrieved))
        except RuntimeError as e:
            raise SystemExit(f"{q.q_id}: {e}")
        seconds = time.perf_counter() - start

        # The retrieved movies, deduped in rank order, as the LLM saw them.
        retrieved_movies = list(dict.fromkeys((c["doc_id"], c["title"], c["year"]) for c in retrieved))
        metrics = score(retrieved, q.relevant_doc_ids)
        rows.append({"q_id": q.q_id, "question_set": q.question_set, "question": q.question,
                     "correct_movies": describe_movies([titles[int(doc_id)] for doc_id in q.relevant_doc_ids]),
                     "retrieved_movies": describe_movies(retrieved_movies),
                     RECALL: round(metrics["recall"], 3), "reciprocal_rank": round(metrics["reciprocal_rank"], 3),
                     "answer": reply, "score": "", "explanation": ""})
        print(f"  {q.q_id:<4} {seconds:5.1f} s   recall {metrics['recall']:.2f}")

    # utf-8-sig adds the marker Excel needs to read accents and curly quotes (O’Hare) correctly.
    pd.DataFrame(rows).to_csv(ANSWERS_CSV, index = False, encoding = "utf-8-sig")
    print(f"\nWrote {len(rows)} answers to {ANSWERS_CSV.name}; fill in score (1-10) and explanation for each")


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
