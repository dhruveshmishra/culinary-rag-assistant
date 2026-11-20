import logging
from typing import Dict, Any, List

import config
from retriever import CulinaryRetriever
from generator import GeminiGenerator

# Set up logging to logfile and console
logging.basicConfig(
    filename=config.LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("assistant")

# Also add stream handler to show logs in console if not already added
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

class CulinaryAssistant:
    def __init__(self):
        logger.info("Initializing CulinaryAssistant...")
        self.retriever = CulinaryRetriever()
        self.generator = GeminiGenerator()
        logger.info("CulinaryAssistant initialized.")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main query entry point:
        1. Retrieve top chunks and evaluate Refusal Guard.
        2. If refused by guard, return standard refusal without calling Gemini.
        3. If guard passes, call Gemini with retrieved context.
        4. Return dict: {answer, sources, refused, retrieved}.
        5. Log all steps to logfile.
        """
        logger.info("--- NEW QUESTION: '%s' ---", question)

        retrieval_res = self.retriever.retrieve(question)
        refused_by_guard = retrieval_res["refused"]
        retrieved_chunks = retrieval_res["chunks"]

        logger.info(
            "Retrieval Guard: refused=%s, best_sim=%.4f (thresh=%.2f), best_bm25=%.4f (thresh=%.2f), keyword_match=%s, reason='%s'",
            refused_by_guard,
            retrieval_res["best_sim"],
            config.SIM_THRESHOLD,
            retrieval_res["best_bm25"],
            config.BM25_THRESHOLD,
            retrieval_res["keyword_matched"],
            retrieval_res["refusal_reason"]
        )

        formatted_retrieved = [
            {
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "section_title": c["section_title"],
                "score": round(c["rrf_score"], 5),
                "sim_score": round(c["sim_score"], 4),
                "bm25_score": round(c["bm25_score"], 4),
                "text": c["text"]
            }
            for c in retrieved_chunks
        ]

        if refused_by_guard:
            logger.info("Decision: Refused by Refusal Guard. Returning refusal message without LLM call.")
            return {
                "answer": config.REFUSAL_RESPONSE,
                "sources": [],
                "refused": True,
                "retrieved": formatted_retrieved
            }

        # Guard passed: call Generator
        gen_res = self.generator.generate(question, retrieved_chunks)
        answer = gen_res["answer"]
        sources = gen_res.get("sources", [])
        is_refused = gen_res.get("refused", False)

        logger.info(
            "Decision: LLM called. refused=%s, model=%s, sources=%s",
            is_refused,
            gen_res.get("model"),
            sources
        )
        logger.info("Answer:\n%s", answer)

        return {
            "answer": answer,
            "sources": sources,
            "refused": is_refused,
            "retrieved": formatted_retrieved
        }

# Module-level convenience function
_default_assistant = None

def ask(question: str) -> Dict[str, Any]:
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = CulinaryAssistant()
    return _default_assistant.ask(question)

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the ingredients in Signature Truffle Ganache?"
    result = ask(q)
    print("Answer:\n", result["answer"])
    print("Sources:", result["sources"])
    print("Refused:", result["refused"])
