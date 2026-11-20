import os
import re
import pickle
import logging
import shutil
from pathlib import Path

# Ensure offline operation for HuggingFace if already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

def normalize_text(text: str) -> str:
    """Normalize curly quotes, apostrophes, dashes, and whitespace."""
    replacements = {
        "\u2019": "'",  # right single quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2014": "--", # em dash
        "\u2013": "-",  # en dash
        "\u00a0": " ",  # non-breaking space
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Normalize multiple spaces per line but preserve line breaks
    lines = [re.sub(r"[ \t]+", " ", line) for line in text.split("\n")]
    return "\n".join(lines)

def clean_page(raw_text: str):
    """
    Clean page strictly by pattern:
    a) Remove header, Document Page N of 50, footer and '=' lines.
    b) Remove three boilerplate sections (Reference Clause headings & text).
    c) Remove Operational Compliance Note (Ref-...) paragraph.
    d) Keep CHAPTER | MODULE title as metadata only.
    """
    text = normalize_text(raw_text)

    # 1. Extract CHAPTER | MODULE as metadata
    chap_match = re.search(r"CHAPTER\s+\d+\s*\|\s*MODULE\s+[\d\.]+:[^=\n]+(?:\n[^=\n]+)*", text, re.IGNORECASE)
    chapter_module = chap_match.group(0).strip() if chap_match else ""
    chapter_module = " ".join(chapter_module.split())

    # 2. Separate body after '===' separator line
    parts = re.split(r"={10,}", text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else text

    # Remove footer
    body = re.sub(r"Page \d+ of \d+\s*\|\s*Confidential Assessment Corpus - Do Not Distribute.*", "", body, flags=re.IGNORECASE | re.DOTALL)

    # 3. Remove the three boilerplate sections matching Reference Clause headings and text
    # Pattern: \d+\.\s+... (Reference Clause ...) followed by text up to next numbered heading or end
    body = re.sub(
        r"(?:^|\n)\d+\.\s+[\s\S]*?\([^\)]*Reference Clause[^\)]*\)[\s\S]*?(?=(?:\n\d+\.\s+[A-Z])|\Z)",
        "",
        body,
        flags=re.IGNORECASE
    )
    # Catch any remaining Reference Clause lines
    body = re.sub(r"(?:^|\n)\d+\.\s+[\s\S]*?Reference Clause[\s\S]*?(?=(?:\n\d+\.\s+[A-Z])|\Z)", "", body, flags=re.IGNORECASE)
    body = re.sub(r".*?Reference Clause.*?\n?", "", body, flags=re.IGNORECASE)

    # 4. Remove Operational Compliance Note (Ref-...) paragraph
    body = re.sub(r"Operational Compliance Note\s*\(Ref-[^\)]+\):?[\s\S]*", "", body, flags=re.IGNORECASE)

    # 5. Clean whitespace
    body = body.strip()

    # 6. Extract section title (e.g. RECIPE CARD 2.4..., SECTION 5.1..., SECTION 7.2...)
    sec_match = re.search(r"^((?:RECIPE CARD|SECTION)\s+[\d\.]+:[^\n]+)", body, re.IGNORECASE | re.MULTILINE)
    section_title = sec_match.group(1).strip() if sec_match else ""

    return body, chapter_module, section_title

def tokenize_for_bm25(text: str):
    """Tokenize, lowercase, and remove stopwords for BM25."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in config.STOPWORDS]

def chunk_text(page_num: int, section_title: str, chapter_module: str, text: str):
    """
    Produce one chunk per surviving page. If > 1200 chars, split by section heading or blank line.
    """
    if len(text) <= 1200:
        return [{
            "chunk_id": f"page_{page_num}_chunk_1",
            "page": page_num,
            "section_title": section_title,
            "chapter_module": chapter_module,
            "text": text,
            "embed_text": f"{section_title}\n{text}" if not text.startswith(section_title) else text
        }]
    
    # If longer than ~1200 chars, split by blank lines or section headings
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sub_chunks = []
    current_chunk = []
    current_len = 0
    chunk_idx = 1

    for p in paragraphs:
        if current_len + len(p) > 1200 and current_chunk:
            combined = "\n\n".join(current_chunk)
            sub_chunks.append({
                "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                "page": page_num,
                "section_title": section_title,
                "chapter_module": chapter_module,
                "text": combined,
                "embed_text": f"{section_title}\n{combined}"
            })
            chunk_idx += 1
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p)

    if current_chunk:
        combined = "\n\n".join(current_chunk)
        sub_chunks.append({
            "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
            "page": page_num,
            "section_title": section_title,
            "chapter_module": chapter_module,
            "text": combined,
            "embed_text": f"{section_title}\n{combined}"
        })

    return sub_chunks

def run_ingest(pdf_path: str = config.PDF_PATH):
    """Load, clean, chunk, embed, and store culinary manual."""
    logger.info("Starting ingestion from: %s", pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    doc = fitz.open(pdf_path)
    logger.info("PDF opened. Total pages: %d", len(doc))

    surviving_chunks = []
    kept_pages = []
    dropped_pages = []

    for i in range(len(doc)):
        page_num = i + 1
        raw_text = doc[i].get_text()
        cleaned_text, chapter_module, section_title = clean_page(raw_text)

        # Drop condition: shorter than ~40 chars or no digits/list/table structure
        has_digits = bool(re.search(r"\d", cleaned_text))
        has_list_or_table = bool(re.search(r"[-•\*:]", cleaned_text))

        if len(cleaned_text) < 40 or not (has_digits and has_list_or_table):
            dropped_pages.append(page_num)
        else:
            kept_pages.append(page_num)
            page_chunks = chunk_text(page_num, section_title, chapter_module, cleaned_text)
            surviving_chunks.extend(page_chunks)

    logger.info("Cleaning complete. Kept pages (%d): %s", len(kept_pages), kept_pages)
    logger.info("Dropped pages (%d): %s", len(dropped_pages), dropped_pages)
    logger.info("Total chunks created: %d", len(surviving_chunks))

    # Initialize Embedding Model
    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
    try:
        embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    texts_to_embed = [c["embed_text"] for c in surviving_chunks]
    logger.info("Computing embeddings for %d chunks...", len(texts_to_embed))
    embeddings = embedder.encode(texts_to_embed, normalize_embeddings=True)

    # Initialize ChromaDB (Idempotent: reset or recreate collection)
    chroma_dir = Path(config.CHROMA_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if it exists to guarantee clean rebuild without duplicates
    try:
        chroma_client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        logger.info("Existing Chroma collection deleted for idempotent rebuild.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in surviving_chunks]
    documents = [c["text"] for c in surviving_chunks]
    metadatas = [
        {
            "page": c["page"],
            "section_title": c["section_title"],
            "chapter_module": c["chapter_module"],
            "embed_text": c["embed_text"]
        }
        for c in surviving_chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )
    logger.info("Stored %d chunks in ChromaDB at %s", collection.count(), config.CHROMA_DIR)

    # Build BM25 Index
    tokenized_corpus = [tokenize_for_bm25(c["embed_text"]) for c in surviving_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
        "chunks": surviving_chunks
    }
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info("Saved BM25 corpus and index to %s", config.BM25_PATH)

    logger.info("Ingestion completed successfully.")
    return {
        "kept_pages": kept_pages,
        "dropped_pages": dropped_pages,
        "chunks_count": len(surviving_chunks)
    }

if __name__ == "__main__":
    run_ingest()
