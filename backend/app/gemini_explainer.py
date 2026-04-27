"""Gemini AI explainability layer for Meta-Guard.

Provides optional natural-language explanations using Google's Gemini API.
If the API key is missing or the call fails, the system falls back gracefully
with no impact on predictions.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

_gemini_model = None
_gemini_available = False


def _init_gemini() -> bool:
    """Lazy-initialise the Gemini client. Returns True if available."""
    global _gemini_model, _gemini_available

    if _gemini_model is not None:
        return _gemini_available

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.info("GEMINI_API_KEY not set — Gemini explanations disabled.")
        _gemini_available = False
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        _gemini_available = True
        logger.info("Gemini AI explainability layer initialised.")
        return True
    except Exception as exc:
        logger.warning(f"Gemini init failed: {exc}")
        _gemini_available = False
        return False


def gemini_batch_explanation(
    total: int,
    normal_detected: int,
    known_detected: int,
    zeroday_detected: int,
    avg_confidence: float,
    avg_distance: float | None,
    top_labels: list[str],
) -> str | None:
    """Generate a Gemini-powered batch-level explanation.

    Returns None if Gemini is unavailable or the API call fails.
    Gemini NEVER affects predictions — this is purely explanatory.
    """
    if not _init_gemini():
        return None

    dist_text = f"{avg_distance:.4f}" if avg_distance is not None else "N/A"
    labels_text = ", ".join(top_labels) if top_labels else "None"

    prompt = f"""You are a cybersecurity analyst reviewing results from Meta-Guard, an AI-based 
Intrusion Detection System for IoMT (Internet of Medical Things) networks.

The system uses a two-stage architecture:
- Stage 1: Binary classifier detecting Normal vs Suspicious traffic
- Stage 2: Open-set classifier distinguishing Known Attacks from Zero-Day (unknown) attacks using embedding distance

Here are the detection results from a batch of synthetic network traffic:
- Total samples: {total}
- Normal traffic detected: {normal_detected}
- Known attacks detected: {known_detected}  
- Zero-day (unknown) attacks detected: {zeroday_detected}
- Average model confidence: {avg_confidence:.2%}
- Average embedding distance: {dist_text}
- Detected attack families: {labels_text}

Provide a concise (3-4 sentences) professional security analysis of these results. 
Focus on:
1. Overall threat assessment
2. The significance of any zero-day detections
3. One actionable recommendation

Keep it professional and suitable for a SOC dashboard display."""

    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"Gemini batch explanation failed: {exc}")
        return None


def gemini_sample_explanation(
    stage1: str,
    stage2: str,
    confidence: float,
    distance: float | None,
    predicted_label: str | None,
    final_decision: str,
) -> str | None:
    """Generate a Gemini-powered single-sample explanation.

    Returns None if Gemini is unavailable or the API call fails.
    """
    if not _init_gemini():
        return None

    dist_text = f"{distance:.4f}" if distance is not None else "N/A"

    prompt = f"""You are a cybersecurity analyst. Briefly explain (2-3 sentences) this Meta-Guard IDS detection result:
- Stage 1 (Edge): {stage1}
- Stage 2 (Cloud): {stage2}
- Confidence: {confidence:.2%}
- Embedding distance: {dist_text}
- Predicted label: {predicted_label or 'N/A'}
- Final decision: {final_decision}

Be concise and professional. Focus on what this means for network security."""

    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"Gemini sample explanation failed: {exc}")
        return None


def gemini_dataset_summary(
    num_normal: int,
    num_known: int,
    num_zero_day: int,
    total: int,
) -> str | None:
    """Generate a Gemini-powered summary of the generated dataset.

    Returns None if Gemini is unavailable.
    """
    if not _init_gemini():
        return None

    prompt = f"""You are a cybersecurity data analyst. A synthetic IoMT network traffic dataset was generated for testing an Intrusion Detection System called Meta-Guard.

Dataset composition:
- Normal traffic samples: {num_normal}
- Known attack samples: {num_known}
- Zero-day (unknown attack) samples: {num_zero_day}
- Total: {total}

The dataset uses 45 network flow features (packet lengths, inter-arrival times, flag counts, protocol fields) matching the CIC IoMT 2024 format.

Normal samples have low variance patterns. Known attacks have moderate anomaly signatures pushed toward known attack prototypes. Zero-day samples have high-deviation out-of-distribution patterns far from all known prototypes.

Provide a concise (2-3 sentences) professional summary of this dataset's composition and what detection challenges it presents for the IDS."""

    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"Gemini dataset summary failed: {exc}")
        return None


def is_gemini_available() -> bool:
    """Check if Gemini API is configured and available."""
    return _init_gemini()
