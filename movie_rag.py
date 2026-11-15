"""Shared pieces of the movie RAG: the embedding model, the BM25 tokenizer, retrieval and answers.

This is Lab 10's micro_rag.py for our corpus. build_index.py imports
load_model(), embed_documents() and tokenize() from here, so chunks are
embedded and tokenized exactly the way questions are. ask.py uses Retriever
and answer(); the measure_*.py scripts use Retriever, build_prompt() and
chat(), which also serves the LLM judge.

Reads what build_index.py wrote:
    data/movies.duckdb      chunks and documents tables
    data/flat.faiss         exact index (IndexFlatIP), the baseline
    data/ivf.faiss          IVF index (IndexIVFFlat), the product
    data/chunk_ids.json     chunk_id of each FAISS id and BM25 row, in order
    data/bm25.pkl           BM25 index over the same chunks

and writes nothing. Answers come from the OpenAI-compatible endpoint named by
LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, as in Lab 10.

Import it rather than running it:
    from movie_rag import Retriever, answer
"""

# Imports
from sentence_transformers import SentenceTransformer
from pathlib import Path
import snowballstemmer
import urllib.request
import urllib.error
import numpy as np
import unicodedata
import functools
import duckdb
import pickle
import faiss
import torch
import json
import os
import re

# Data paths: the database, and the index files build_index.py writes and Retriever reads.
DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "movies.duckdb"
INDEX_PATHS = {"flat": DATA / "flat.faiss", "ivf": DATA / "ivf.faiss"}
CHUNK_IDS_PATH = DATA / "chunk_ids.json"
BM25_PATH = DATA / "bm25.pkl"

# Embedding model: EmbeddingGemma 2, text only, loaded the way Lab 9's embedding_demo.py does.
MODEL_ID = "google/embeddinggemma-2"
DIM = 768
CONTEXT_TOKENS = 8192
QUERY_PROMPT = "SearchQuery"                    # the model's "task: search result | query: " prefix

# Retrieval settings
TOP_K = 5                                       # chunks the LLM gets per question, and the k in recall@k
NPROBE = 64                                     # IVF lists searched per query, of 1,201; measure_speed.py's sweep: ANN recall@5 0.91
N_CANDIDATES = 50                               # chunks each search hands to fusion
RRF_K = 60                                      # reciprocal rank fusion: score = sum of 1 / (RRF_K + rank)

# Generation settings
TEMPERATURE = 0                                 # no sampling, so reruns give the same answer for the evaluation
LLM_TIMEOUT = 60                                # seconds to wait for the endpoint, as in Lab 10

# BM25 tokenizer: Lab 10's STOPWORDS, plus words that most movie questions share.
STOPWORDS = set("""a an and are as at be because by can each for from has have
how in is it its like more most no not of on one only or so than that the
their them then there these they this to was what when where which while who
why will with you your""".split())
QUESTION_WORDS = {"about", "did", "do", "does", "film", "films", "movie", "movies", "plot", "some", "story"}
STEMMER = snowballstemmer.stemmer("english")


# Function to load the embedding model
def load_model(device = None):
    """Load EmbeddingGemma 2 for text, on the GPU when there is one.

    As in Lab 9's embedding_demo.py: the vision and audio towers are left out,
    the window is the model's 8,192 tokens, and anything longer is cut from
    the right. Uses an NVIDIA GPU (cuda), then an Apple GPU (mps), then the CPU.

    Weights are float32 as in the lab, except bfloat16 on an NVIDIA GPU: on an
    RTX 4070 laptop that embeds 2-3x faster (70 vs 31 chunks/s at 256 tokens),
    and the vectors match float32's to a mean cosine of 0.9999.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = SentenceTransformer(MODEL_ID, device = device,
                                config_kwargs = {"vision_config": None, "audio_config": None},
                                model_kwargs = {"torch_dtype": dtype})
    model.max_seq_length = CONTEXT_TOKENS
    model.tokenizer.truncation_side = "right"
    return model


# Function to format a chunk the way EmbeddingGemma 2 expects documents
def document_text(title, text):
    """Return the chunk in the model's document format, with the movie's title filled in.

    The model's own "Document" prompt is "title: none | text: ". We give it the
    real title, because summaries rarely name their own movie.
    Example: ("Jaws", "Brody shoots the tank.") -> "title: Jaws | text: Brody shoots the tank."
    """
    return f"title: {title} | text: {text}"


# Function to embed chunks
def embed_documents(model, titles, texts, batch_size = 16):
    """Embed chunks as unit-length float32 vectors, one row per chunk, in input order.

    batch_size is how many chunks go through the GPU at once; long chunks need
    a smaller one to fit in GPU memory.
    """
    inputs = [document_text(title, text) for title, text in zip(titles, texts)]
    vectors = model.encode(inputs, batch_size = batch_size, convert_to_numpy = True, show_progress_bar = False)
    return unit_rows(vectors)


# Function to embed a question
def embed_query(model, question):
    """Embed one question with the search-query prompt, as a (1, 768) float32 array."""
    vector = model.encode([question], prompt_name = QUERY_PROMPT, convert_to_numpy = True)
    return unit_rows(vector)


# Function to scale vectors to unit length
def unit_rows(vectors):
    """Return the rows as float32 unit vectors, so inner product is cosine.

    Normalizing here, in float32, rather than inside the model keeps lengths
    exact even when the model runs in bfloat16 (whose norms come out 1.001).
    """
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis = 1, keepdims = True)


# Function to stem one word
@functools.lru_cache(maxsize = None)
def stem(word):
    """Return the Snowball stem of a lowercase word.

    The cache matters: the corpus reuses a small vocabulary, and stemming
    each of its 13 million words afresh took 6-7 minutes.
    Example: "killing" -> "kill"
    """
    return STEMMER.stemWord(word)


# Function to split text into BM25 terms
def tokenize(text):
    """Lowercase, strip accents, split into words, drop stopwords and stem.

    build_index.py uses it for chunks and Retriever uses it for questions, so
    the two always match.
    Example: "In Jaws, how do they kill the shark?" -> ["jaw", "kill", "shark"]
    """
    # Accents come off so "Amélie" matches "Amelie": é -> e + combining accent -> e.
    # Plain ASCII text, which is most of it, has none to strip.
    text = text.lower()
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Words are runs of letters and digits; stopwords and question words carry no signal.
    words = re.findall(r"[a-z0-9]+", text)
    return [stem(w) for w in words if w not in STOPWORDS and w not in QUESTION_WORDS]


# Class to search the indexes build_index.py saved
class Retriever:
    """Loads the indexes once, then answers top-k searches by vector, keyword or both."""

    def __init__(self, index = "ivf", model = None):
        """Load the FAISS index, the chunk_id list, the BM25 index and the model.

        index is "ivf" (the product) or "flat" (the exact baseline). Pass model
        to share one loaded model between several Retrievers.
        """
        self.index_type = index

        # Without the index files there is nothing to search yet.
        missing = [path.name for path in [INDEX_PATHS[index], CHUNK_IDS_PATH, BM25_PATH] if not path.exists()]
        if missing:
            raise SystemExit(f"No {', '.join(missing)} in data/; run build_index.py first")

        # FAISS ids are positions; chunk_ids[i] is the chunk behind FAISS id i and BM25 row i.
        self.index = faiss.read_index(str(INDEX_PATHS[index]))
        if index == "ivf":
            self.index.nprobe = NPROBE
        with open(CHUNK_IDS_PATH, encoding = "utf-8") as f:
            self.chunk_ids = json.load(f)
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        self.model = model or load_model()
        self.con = duckdb.connect(DB_PATH, read_only = True)

        # The index files and the database must describe the same chunks.
        n_known = self.con.execute("SELECT count(*) FROM chunks WHERE chunk_id IN (SELECT unnest(?))",
                                   [self.chunk_ids]).fetchone()[0]
        if not (self.index.ntotal == len(self.chunk_ids) == n_known):
            raise SystemExit(f"The index files don't match {DB_PATH.name}; run build_index.py again")


    def vector_search(self, question, n):
        """Return the n nearest chunks as [(chunk_id, cosine)], best first."""
        scores, ids = self.index.search(embed_query(self.model, question), n)
        # FAISS pads with id -1 when the probed IVF lists hold fewer than n vectors.
        return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


    def keyword_search(self, question, n):
        """Return the n best BM25 chunks as [(chunk_id, BM25 score)], best first.

        Chunks that share no term with the question (score 0) are left out.
        """
        scores = self.bm25.get_scores(tokenize(question))
        top = np.argsort(-scores, kind = "stable")[:n]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


    def hybrid_search(self, question, n):
        """Fuse the vector and BM25 rankings by reciprocal rank fusion; return [(chunk_id, RRF score)].

        Each search contributes its top N_CANDIDATES. A chunk scores
        1 / (RRF_K + rank) for each ranking it appears in.
        """
        fused = {}
        for ranking in (self.vector_search(question, N_CANDIDATES), self.keyword_search(question, N_CANDIDATES)):
            for rank, (chunk_id, _) in enumerate(ranking, start = 1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key = lambda item: (-item[1], item[0]))[:n]


    def retrieve(self, question, k = TOP_K, mode = "hybrid"):
        """Return the k best chunks for the question as dicts, best first.

        mode is "vector", "keyword" or "hybrid". Each dict has chunk_id, doc_id,
        title, year, text and score (a cosine, BM25 or RRF score, by mode).
        """
        search = {"vector": self.vector_search, "keyword": self.keyword_search, "hybrid": self.hybrid_search}[mode]
        ranked = search(question, k)

        # The join to documents is what lets every chunk cite its movie.
        rows = self.con.execute("""
            SELECT c.chunk_id, c.doc_id, d.title, left(d.release_date, 4) AS year, c.text
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id IN (SELECT unnest(?))
        """, [[chunk_id for chunk_id, _ in ranked]]).fetchall()
        by_id = {row[0]: row for row in rows}
        return [{"chunk_id": chunk_id, "doc_id": by_id[chunk_id][1], "title": by_id[chunk_id][2],
                 "year": by_id[chunk_id][3], "text": by_id[chunk_id][4], "score": score}
                for chunk_id, score in ranked]


# Function to assemble the prompt
def build_prompt(question, retrieved):
    """Return Lab 10's prompt: the instruction, one source line per chunk, then the question.

    Each source line is "[doc_id] Title (year): text". The year tells remakes
    apart, such as the four Titanics.
    """
    sources = "\n".join(f"[{c['doc_id']}] {c['title']} ({c['year'] or 'no date'}): {c['text']}" for c in retrieved)
    return f"Answer using ONLY the sources below. Cite as [doc_id].\n\n{sources}\n\nQuestion: {question} \n\nIf the provided sources are insufficient to answer, say \"The provided sources do not contain enough information to answer the question.\""


# Function to send one prompt to the LLM
def chat(prompt):
    """Send the prompt to the OpenAI-compatible endpoint as one user message; return the reply text.

    The endpoint is LLM_BASE_URL, with LLM_API_KEY and LLM_MODEL, as in Lab 10.
    Every failure raises RuntimeError with a one-line reason: no endpoint set,
    a refused connection, a timeout, an HTTP error (401 for a bad key, 404 for
    a wrong URL) or a reply with no answer in it. answer() falls back to echo
    mode on it; the LLM judge should stop instead.
    Example: chat("Reply with one word: hi") -> "Hi"
    """
    base = os.environ.get("LLM_BASE_URL")
    if not base:
        raise RuntimeError("LLM_BASE_URL is not set")
    url = base.rstrip("/") + "/chat/completions"

    # Lab 10's request, plus a fixed temperature.
    request = urllib.request.Request(
        url,
        headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}",
                   "Content-Type": "application/json"},
        data = json.dumps({"model": os.environ.get("LLM_MODEL", ""),
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": TEMPERATURE}).encode())

    # HTTP errors carry the server's own explanation in the body, so show the start of it.
    try:
        with urllib.request.urlopen(request, timeout = LLM_TIMEOUT) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as e:
        detail = " ".join(e.read().decode("utf-8", errors = "replace").split())[:200]
        raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from e
    except (OSError, ValueError) as e:      # refused, unreachable, timed out, or not JSON
        raise RuntimeError(f"no usable reply from {url}: {getattr(e, 'reason', e)}") from e

    # The answer is the first choice's message; anything else usually means a wrong URL or model.
    try:
        return reply["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise RuntimeError(f"{url} replied without an answer: {str(reply)[:200]}") from e


# Function to list the movies behind an answer
def format_sources(retrieved):
    """Return the retrieved movies, best first, one per line, so a citation like [74830] can be read.

    A movie with several retrieved chunks is listed once.
    Example: "Sources:\\n  [1173339]  Jaws 3-D (1983)\\n  [74830]    Jaws (1975)"
    """
    movies = {}
    for c in retrieved:
        movies.setdefault(c["doc_id"], f"{c['title']} ({c['year'] or 'no date'})")
    return "Sources:\n" + "\n".join(f"  {f'[{doc_id}]':<10} {movie}" for doc_id, movie in movies.items())


# Function to answer a question
def answer(question, retriever, k = TOP_K, mode = "hybrid"):
    """Retrieve, build the prompt and generate, as in Lab 10; return the text to print.

    Without LLM_BASE_URL this runs in echo mode and returns the prompt, so you
    see exactly what retrieval chose. With it, the reply comes back followed by
    the retrieved movies (format_sources). If the endpoint fails, the prompt
    comes back with the reason instead, so a live demo keeps going.
    """
    retrieved = retriever.retrieve(question, k, mode)
    prompt = build_prompt(question, retrieved)
    if not os.environ.get("LLM_BASE_URL"):
        cited = ", ".join(str(c["doc_id"]) for c in retrieved)
        return f"{prompt}\n\n[echo mode: no LLM configured. A real model would answer from: {cited}]"

    # Real mode. A failed request shows the prompt rather than stopping ask.py.
    try:
        reply = chat(prompt)
    except RuntimeError as e:
        return f"{prompt}\n\n[LLM request failed, so this is the prompt instead. {e}]"
    return f"{reply}\n\n{format_sources(retrieved)}"
