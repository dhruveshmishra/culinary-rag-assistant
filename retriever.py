import os
import re
import pickle
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure offline operation for HuggingFace if cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger("retriever")

def simple_stem(word: str) -> str:
    """Lightweight English suffix stemmer without external dependencies."""
    word = word.lower()
    for suffix in ["ing", "ies", "es", "ed", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def tokenize(text: str) -> List[str]:
    """Tokenize and remove stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in config.STOPWORDS]

def extract_meaningful_terms(query: str) -> List[str]:
    """Extract query terms excluding stopwords and generic question words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
    meaningful = [
        t for t in tokens
        if t not in config.GENERIC_WORDS and t not in config.STOPWORDS and len(t) > 2
    ]
    return meaningful

class CulinaryRetriever:
    def __init__(self):
        self._load_embedder()
        self._load_chroma()
        self._load_bm25()

    def _load_embedder(self):
        try:
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def _load_chroma(self):
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = self.chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    def _load_bm25(self):
        if not os.path.exists(config.BM25_PATH):
            raise FileNotFoundError(f"BM25 index not found at {config.BM25_PATH}. Run ingest.py first.")
        with open(config.BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunks = bm25_data["chunks"]

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Execute hybrid search (Dense Vector + BM25) with Reciprocal Rank Fusion (k=60),
        and evaluate the two-stage Refusal Guard.
        """
        query_text = query.strip()
        meaningful_terms = extract_meaningful_terms(query_text)

        # 1. Vector Search (using BGE search instruction prefix)
        prefixed_query = f"{config.BGE_QUERY_PREFIX}{query_text}"
        query_embedding = self.embedder.encode([prefixed_query], normalize_embeddings=True)[0]

        chroma_res = self.collection.get(include=["embeddings", "documents", "metadatas"])
        num_docs = len(chroma_res["ids"])

        vector_scores = []
        for i in range(num_docs):
            doc_emb = np.array(chroma_res["embeddings"][i])
            # Cosine similarity between unit-normalized vectors
            sim = float(np.dot(query_embedding, doc_emb))
            vector_scores.append((chroma_res["ids"][i], sim, chroma_res["documents"][i], chroma_res["metadatas"][i]))

        # Sort vector ranks (1-based: rank 1 is highest similarity)
        vector_scores.sort(key=lambda x: x[1], reverse=True)
        vector_rank_map = {item[0]: rank + 1 for rank, item in enumerate(vector_scores)}
        vector_sim_map = {item[0]: item[1] for item in vector_scores}

        best_sim = vector_scores[0][1] if vector_scores else 0.0

        # 2. BM25 Search
        query_tokens = tokenize(query_text)
        bm25_scores_list = self.bm25.get_scores(query_tokens)
        bm25_results = []
        for i, chunk in enumerate(self.bm25_chunks):
            bm25_results.append((chunk["chunk_id"], float(bm25_scores_list[i])))

        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_rank_map = {item[0]: rank + 1 for rank, item in enumerate(bm25_results)}
        bm25_score_map = {item[0]: item[1] for item in bm25_results}

        best_bm25 = bm25_results[0][1] if bm25_results else 0.0

        # 3. Reciprocal Rank Fusion (RRF)
        all_ids = set(vector_rank_map.keys()).union(set(bm25_rank_map.keys()))
        rrf_scores = []
        doc_info_map = {item[0]: (item[2], item[3]) for item in vector_scores}

        for cid in all_ids:
            v_rank = vector_rank_map.get(cid, num_docs + 1)
            b_rank = bm25_rank_map.get(cid, num_docs + 1)
            rrf = (1.0 / (config.RRF_K + v_rank)) + (1.0 / (config.RRF_K + b_rank))
            doc_text, doc_meta = doc_info_map[cid]
            rrf_scores.append({
                "chunk_id": cid,
                "page": doc_meta.get("page"),
                "section_title": doc_meta.get("section_title"),
                "chapter_module": doc_meta.get("chapter_module"),
                "text": doc_text,
                "sim_score": vector_sim_map.get(cid, 0.0),
                "bm25_score": bm25_score_map.get(cid, 0.0),
                "rrf_score": rrf
            })

        # Sort by RRF score descending and take top K
        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_chunks = rrf_scores[:config.TOP_K]

        # 4. Refusal Guard (Critical)
        # Check 1: Semantic check
        semantic_pass = (best_sim >= config.SIM_THRESHOLD)

        # Check 2: Keyword check
        # At least one meaningful query term (or simple stem) must appear in top chunks' text,
        # OR best BM25 score >= BM25_THRESHOLD
        top_chunks_text = " ".join([c["text"].lower() for c in top_chunks])
        top_chunks_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", top_chunks_text))
        top_chunks_stems = {simple_stem(w) for w in top_chunks_words}

        keyword_matched = False
        if meaningful_terms:
            for term in meaningful_terms:
                term_stem = simple_stem(term)
                if term in top_chunks_words or term_stem in top_chunks_stems or term in top_chunks_text:
                    keyword_matched = True
                    break

        keyword_pass = keyword_matched or (best_bm25 >= config.BM25_THRESHOLD)

        # Refusal Decision: Use BOTH checks and refuse if EITHER fails
        refused = not (semantic_pass and keyword_pass)

        refusal_reasons = []
        if not semantic_pass:
            refusal_reasons.append(f"Semantic similarity {best_sim:.4f} < threshold {config.SIM_THRESHOLD:.4f}")
        if not keyword_pass:
            refusal_reasons.append(f"No meaningful keyword match in top chunks and BM25 {best_bm25:.4f} < threshold {config.BM25_THRESHOLD:.4f}")

        refusal_reason = "; ".join(refusal_reasons) if refused else "Passed both checks"

        return {
            "query": query,
            "refused": refused,
            "refusal_reason": refusal_reason,
            "best_sim": best_sim,
            "best_bm25": best_bm25,
            "meaningful_terms": meaningful_terms,
            "keyword_matched": keyword_matched,
            "chunks": top_chunks
        }
