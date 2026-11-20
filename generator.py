import os
import re
import time
import logging
from typing import List, Dict, Any, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

import config

logger = logging.getLogger("generator")

class GeminiGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. API calls will fail.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def _determine_source_pages(self, answer_text: str, chunks: List[Dict[str, Any]]) -> List[int]:
        """
        Determine which retrieved chunk(s) the answer drew facts from.
        Uses token and key factual overlap (quantities, dish names, temperatures, allergens).
        """
        if not chunks:
            return []

        # Tokenize answer
        answer_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.%]+\b", answer_text.lower()))
        content_tokens = [t for t in answer_tokens if t not in config.STOPWORDS and len(t) > 1]

        scored_chunks = []
        for c in chunks:
            c_text_lower = c["text"].lower()
            score = sum(1 for t in content_tokens if t in c_text_lower)
            scored_chunks.append((score, c["page"]))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        if scored_chunks and scored_chunks[0][0] > 0:
            return [scored_chunks[0][1]]
        return [chunks[0]["page"]]

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer from retrieved chunks using Gemini with strict factual prompt.
        """
        if not self.client:
            return {
                "answer": "Error: GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your .env file.",
                "sources": [],
                "refused": True,
                "error": "Missing GEMINI_API_KEY"
            }

        # Build user message with labelled context blocks
        context_blocks = []
        for c in chunks:
            block = f"[Page {c['page']} | {c['section_title']}]\n{c['text']}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)
        user_message = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"

        generation_config = types.GenerateContentConfig(
            temperature=config.TEMPERATURE,
            system_instruction=config.SYSTEM_PROMPT
        )

        model_to_use = config.PRIMARY_MODEL
        response_text = ""
        last_error = None

        # Retry loop with backoff and automatic model fallback if primary model is unavailable
        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info("Calling Gemini (%s), attempt %d...", model_to_use, attempt + 1)
                response = self.client.models.generate_content(
                    model=model_to_use,
                    contents=user_message,
                    config=generation_config
                )
                response_text = response.text.strip()
                break
            except ClientError as e:
                err_msg = str(e)
                # If primary model is unavailable / retired for new users, fall back to fallback model
                if ("404" in err_msg or "not available" in err_msg or "NOT_FOUND" in err_msg) and model_to_use != config.FALLBACK_MODEL:
                    logger.warning("Primary model %s returned 404 (%s). Falling back to %s.", model_to_use, err_msg, config.FALLBACK_MODEL)
                    model_to_use = config.FALLBACK_MODEL
                    continue
                last_error = e
                logger.error("ClientError on attempt %d: %s", attempt + 1, err_msg)
                time.sleep(1.0 * (2 ** attempt))
            except Exception as e:
                last_error = e
                logger.error("Error on attempt %d: %s", attempt + 1, e)
                time.sleep(1.0 * (2 ** attempt))

        if not response_text:
            if last_error:
                return {
                    "answer": f"Error communicating with Gemini API: {last_error}",
                    "sources": [],
                    "refused": True,
                    "error": str(last_error)
                }
            response_text = config.REFUSAL_RESPONSE

        # Check if model returned the exact refusal sentence
        clean_ans = response_text.strip()
        is_refusal = (
            clean_ans == config.REFUSAL_RESPONSE or
            "cannot find this information in the manual" in clean_ans.lower()
        )

        if is_refusal:
            return {
                "answer": config.REFUSAL_RESPONSE,
                "sources": [],
                "refused": True,
                "model": model_to_use
            }

        # Determine source page(s) based on factual content overlap
        sources = self._determine_source_pages(clean_ans, chunks)
        source_cite = f"Source: Page {sources[0]}" if sources else ""

        final_answer = f"{clean_ans}\n\n{source_cite}" if source_cite else clean_ans

        return {
            "answer": final_answer,
            "raw_answer": clean_ans,
            "sources": sources,
            "refused": False,
            "model": model_to_use
        }
