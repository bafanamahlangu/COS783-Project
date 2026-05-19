"""
keyword_searcher.py — Semantic Forensic Search Module
======================================================
Uses SentenceTransformer embeddings and cosine similarity to perform
context-aware search over forensic text data (emails, logs, messages).

Why transformers require minimal preprocessing
----------------------------------------------
Transformer models like all-MiniLM-L6-v2 are pre-trained on large corpora
and internally handle tokenisation, subword splitting, and contextual
representation. Aggressive preprocessing (stemming, lowercasing, stopword
removal) would destroy the contextual signals the model relies on, reducing
retrieval quality. Only whitespace normalisation is applied here.

Why this is better than traditional keyword search
---------------------------------------------------
Exact keyword matching fails when suspects use synonyms, slang, or
paraphrasing (e.g. "wire the cash" instead of "transfer funds"). Semantic
search maps both the query and documents into a shared embedding space, so
conceptually related text scores highly even without lexical overlap.

Why semantic search is useful in digital forensics
---------------------------------------------------
Forensic investigators rarely know the exact phrasing used in criminal
communications. Semantic search lets an analyst query with natural-language
concepts ("financial fraud", "drug transaction") and surface relevant
evidence regardless of how it was worded, dramatically reducing manual
review time.

What remains for main.py
------------------------
main.py is responsible for:
  - Accepting user input (file name, query keywords)
  - Instantiating KeywordSearcher and calling search_keywords()
  - Iterating over the returned indices/scores to display results
  - Optionally calling extract_entities() on matched chunks for display
"""

import re
import spacy
from sentence_transformers import SentenceTransformer, util
from file_reader import FileReader

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once to avoid repeated I/O overhead
# ---------------------------------------------------------------------------
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Load spaCy English model; falls back gracefully if not installed
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    _NLP = None
    print(
        "[Warning] spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse redundant whitespace — the only preprocessing applied.

    Transformers handle everything else internally (tokenisation, casing,
    punctuation), so heavier cleaning would only hurt retrieval quality.
    """
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Entity extraction (NER)
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict:
    """Return named entities grouped by label using spaCy NER.

    Useful in forensic investigations to surface names, organisations,
    locations, dates, and monetary values without manual inspection.

    Args:
        text: Raw text from a matched chunk.

    Returns:
        Dict mapping entity label → list of entity strings, e.g.
        {"PERSON": ["Jamie"], "MONEY": ["$5,000"]}
    """
    if _NLP is None:
        return {}

    doc = _NLP(text)
    entities: dict = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities


# ---------------------------------------------------------------------------
# Core search class
# ---------------------------------------------------------------------------

class KeywordSearcher:
    """Semantic keyword searcher backed by SentenceTransformer embeddings.

    Architecture
    ------------
    Documents → minimal preprocessing → SentenceTransformer embeddings
    → cosine similarity against query embedding → ranked top-k results
    """

    def __init__(self, keywords: str, file_path: str):
        self.keywords = keywords
        self.file_path = file_path

    # ------------------------------------------------------------------
    # Public API — called as KeywordSearcher.search_keywords(kw, path)
    # from main.py (static-style call on the class)
    # ------------------------------------------------------------------

    @staticmethod
    def search_keywords(keywords: str, file_path: str, top_k: int = 5):
        """Embed the query and all document chunks, rank by cosine similarity.

        Args:
            keywords: Natural-language query string from the investigator.
            file_path: Path to the CSV file processed by FileReader.
            top_k: Number of top results to return (default 5).

        Returns:
            top_indices (list[int]): Indices into the chunks list, best first.
            top_scores  (list[float]): Corresponding cosine similarity scores.
        """
        # 1. Load and chunk the document
        reader = FileReader(file_path)
        message_ids, chunks = reader.read_file()

        if not chunks:
            print("[Warning] No text chunks found in the file.")
            return [], []

        # 2. Minimal preprocessing
        clean_chunks = [_normalise(c) for c in chunks]
        clean_query = _normalise(keywords)

        # 3. Encode query and all chunks into the same embedding space
        #    convert_to_tensor=True enables GPU acceleration if available
        query_embedding = _MODEL.encode(clean_query, convert_to_tensor=True)
        chunk_embeddings = _MODEL.encode(clean_chunks, convert_to_tensor=True)

        # 4. Cosine similarity between query and every chunk
        scores = util.cos_sim(query_embedding, chunk_embeddings)[0]  # shape: (N,)

        # 5. Rank and return top-k
        top_results = scores.topk(k=min(top_k, len(chunks)))
        top_indices = top_results.indices.tolist()
        top_scores = [round(float(s), 4) for s in top_results.values.tolist()]

        return top_indices, top_scores
