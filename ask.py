"""Ask the movie RAG a question from the command line.

Reads the indexes build_index.py wrote, through movie_rag.Retriever, and
writes nothing. With LLM_BASE_URL, LLM_API_KEY and LLM_MODEL set (in .env or
the shell), it prints the model's answer and the movies it drew on. Without them, or if the
endpoint fails, it runs in Lab 10's echo mode and prints the prompt, so you
see exactly what retrieval chose. With no question on the command line it
keeps asking, so the model and indexes load only once (use this for the demo).

Run after build_index.py:
    python ask.py "In Jaws, how do they kill the shark?"
    python ask.py
"""

# Imports
from movie_rag import Retriever, answer
import time
import sys
import os


# Main function
def main():
    # Load everything, then run one throwaway search: the GPU's first search takes about
    # a second longer than the rest, and a demo audience shouldn't wait for it.
    start = time.perf_counter()
    retriever = Retriever()
    retriever.retrieve("warm-up question", 1)
    print(f"Loaded the model and the indexes in {time.perf_counter() - start:.1f} s")

    # Say where answers come from, so a missing endpoint shows up before the first question.
    if os.environ.get("LLM_BASE_URL"):
        print(f"Answers from {os.environ.get('LLM_MODEL') or 'the default model'} at {os.environ['LLM_BASE_URL']}")
    else:
        print("No LLM_BASE_URL in .env or the shell: echo mode prints the prompt instead of an answer")

    # One question from the command line, then exit.
    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:]), retriever))
        return

    # Otherwise keep asking until a blank line.
    while question := input("\nQuestion (blank line to quit): ").strip():
        start = time.perf_counter()
        print(answer(question, retriever))
        print(f"({time.perf_counter() - start:.2f} s)")


# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
