"""
src/llm_inference.py
--------------------
LLM-based classification using GPT-4o-mini with structured output.
Supports standalone use and hybrid routing (baseline + LLM).

Environment variable required:
    OPENAI_API_KEY  — set in .env or shell
"""
import os
import time
from typing import List, Literal

from dotenv import load_dotenv
load_dotenv()

# ── Pydantic schema for structured LLM output ─────────────────────────────────
from pydantic import BaseModel, Field


class ArticleClassification(BaseModel):
    label: Literal["Fake", "Real"] = Field(
        description="Binary classification: Fake or Real."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 (uncertain) and 1 (certain)."
    )
    reasons: List[str] = Field(
        description="Up to 3 short text-grounded reasons for the decision."
    )
    needs_human_review: bool = Field(
        description="True if article is ambiguous, very short, or lacks clear signals."
    )


# ── System prompt (EDA-grounded) ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert NLP classification assistant. Your task is binary text classification.

Task definition:
- Label = Fake: content that is misleading, fabricated, manipulative, or low-credibility
  based solely on its internal textual signals.
- Label = Real: content that is coherent, structured, evidence-based, and consistent
  with credible journalism or factual communication.

IMPORTANT: Judge only from the article text itself. Do not fact-check against external knowledge.

Signals that suggest Fake:
- Sensational, emotionally manipulative, or clickbait phrasing
- Dramatic claims without specific evidence or named attribution
- Vague or anonymous sourcing ("experts say", "sources claim") with no verifiable detail
- Internal inconsistency or contradiction
- Social-media artifacts (retweet-style content, embedded Twitter/image captions)
- Informal vocabulary: exclamation marks, "just", "like", "guys"

Signals that suggest Real:
- Named, specific attribution (people, institutions, dates, locations)
- Formal journalistic tone (Reuters/AP style)
- Balanced, cautious, evidence-based language
- Specific numerical claims tied to named sources
- Consistent and coherent structure

Instructions:
1. Classify as exactly one of: Fake or Real.
2. Return a confidence score (0.0 = very uncertain, 1.0 = very certain).
3. Provide up to 3 short reasons grounded in the article text.
4. Set needs_human_review=true if the article is ambiguous, very short, or lacks clear signals.
5. Return ONLY valid JSON matching the required schema.
"""

LLM_MODEL    = "gpt-4o-mini"
LLM_MAX_WORDS = 800   # truncate to control token cost (~$0.15/1M input tokens)


def _get_client():
    """Lazy-load OpenAI client to avoid import errors in no-LLM mode."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to .env or export it in your shell.\n"
            "To run without LLM: set USE_LLM=False or NO_LLM=1."
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def classify_with_llm(
    article_text: str,
    model: str = LLM_MODEL,
    max_words: int = LLM_MAX_WORDS,
) -> dict:
    """
    Classify a single article using GPT-4o-mini structured output.

    Returns a dict with keys:
        label, confidence, reasons, needs_human_review,
        latency_s, prompt_tokens, completion_tokens
    """
    client = _get_client()

    truncated = " ".join(str(article_text).split()[:max_words])

    t0 = time.time()
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Article:\n\n{truncated}"},
        ],
        response_format=ArticleClassification,
        temperature=0.1,
        max_tokens=300,
    )
    latency = time.time() - t0

    parsed = response.choices[0].message.parsed
    usage  = response.usage

    return {
        "label"              : parsed.label,
        "confidence"         : parsed.confidence,
        "reasons"            : parsed.reasons,
        "needs_human_review" : parsed.needs_human_review,
        "latency_s"          : round(latency, 3),
        "prompt_tokens"      : usage.prompt_tokens,
        "completion_tokens"  : usage.completion_tokens,
    }


def hybrid_classify(
    article_text: str,
    baseline_pipeline,
    threshold: float = 0.80,
    use_llm: bool = True,
) -> dict:
    """
    Hybrid router:
      - baseline_confidence >= threshold  →  return baseline result immediately
      - baseline_confidence <  threshold  →  escalate to LLM

    Args:
        article_text       : raw article string
        baseline_pipeline  : fitted sklearn Pipeline with predict_proba
        threshold          : confidence cutoff for LLM escalation
        use_llm            : set False to always use baseline (no API calls)

    Returns a dict with:
        final_label, source, baseline_confidence, llm_confidence,
        reasons, needs_human_review, latency_s, prompt_tokens, completion_tokens
    """
    proba = baseline_pipeline.predict_proba([article_text])[0]
    conf  = float(max(proba))
    bl_label = "Real" if proba.argmax() == 1 else "Fake"

    if not use_llm or conf >= threshold:
        return {
            "final_label"        : bl_label,
            "source"             : "baseline",
            "baseline_confidence": round(conf, 4),
            "llm_confidence"     : None,
            "reasons"            : [],
            "needs_human_review" : False,
            "latency_s"          : 0.0,
            "prompt_tokens"      : 0,
            "completion_tokens"  : 0,
        }

    llm = classify_with_llm(article_text)
    return {
        "final_label"        : llm["label"],
        "source"             : "llm",
        "baseline_confidence": round(conf, 4),
        "llm_confidence"     : llm["confidence"],
        "reasons"            : llm["reasons"],
        "needs_human_review" : llm["needs_human_review"],
        "latency_s"          : llm["latency_s"],
        "prompt_tokens"      : llm["prompt_tokens"],
        "completion_tokens"  : llm["completion_tokens"],
    }
