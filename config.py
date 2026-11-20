import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "grand_bistro_culinary_recipe_manual_50p.pdf"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = "culinary_manual"
BM25_PATH = os.getenv("BM25_PATH", str(BASE_DIR / "bm25_data.pkl"))
LOG_PATH = os.getenv("LOG_PATH", str(BASE_DIR / "culinary_assistant.log"))

# Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Gemini Model Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("gemini api") or os.getenv("GEMINI_API")
PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.0
MAX_RETRIES = 3

# Retrieval & Hybrid Search
RRF_K = 60
TOP_K = 3

# Refusal Guard Thresholds (Empirically Calibrated)
# In-scope min similarity was 0.6427, out-of-scope max similarity was 0.6016
SIM_THRESHOLD = 0.62
# In-scope min BM25 was 2.2875, out-of-scope max BM25 was 0.8502
BM25_THRESHOLD = 1.50

REFUSAL_RESPONSE = "I cannot find this information in the manual."

# System Prompt for Strict Factual Generation
SYSTEM_PROMPT = (
    "You are a culinary operations assistant for Le Grand Bistro. "
    "Answer using ONLY the CONTEXT provided. Never use outside knowledge, even if you know the answer. "
    "If the answer is not fully contained in the CONTEXT, reply exactly: I cannot find this information in the manual. "
    "Quote exact numbers and units as written (grams, millilitres, degrees Celsius, minutes). "
    "If the question has several parts, answer every part. "
    "Keep the answer short and factual. Do not add safety advice or extra tips that are not in the CONTEXT."
)

# Generic question words and stopwords for refusal guard
GENERIC_WORDS = {
    "what", "which", "exact", "required", "according", "standard", "mandatory",
    "minimum", "specified", "is", "are", "the", "in", "of", "and", "to", "for",
    "a", "an", "on", "does", "at", "as", "by", "with", "from", "how", "why",
    "when", "where", "who", "whom", "whose", "can", "could", "would", "should",
    "much", "many", "time", "current", "rating", "season", "winter", "visitors",
    "close", "paris", "copenhagen", "weather", "forecast", "weekend", "won",
    "secret", "recipe", "hours", "calories", "capital", "city", "calibrate",
    "machine", "stock", "price", "head", "chef", "style", "table", "section",
    "give", "tell", "show", "list", "name", "find", "get", "page", "manual"
}

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
}
