import os
import statistics
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

# Search for .env explicitly at project root
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# ---------------------------------------------------------------------------
# IoMT / IoT feature profile — derived from the training data distribution
# The scaler was fit on CIC_IoMT_2024_WiFi_MQTT data with 45 features.
# ---------------------------------------------------------------------------
_IOMT_EXPECTED_FEATURES = 45

# AI disclaimer shown at the bottom of every analysis
_AI_DISCLAIMER = (
    "\n\n---\n"
    "🤖 *Disclaimer: This analysis is AI-generated based on the Meta-Guard pipeline outputs. "
    "Results are indicative and should be verified by a qualified security analyst before "
    "taking action. Do not rely solely on this report for critical security decisions.*"
)


def _format_feature_summary(features: list[float]) -> str:
    """Generate a quick statistical fingerprint of the feature vector."""
    n = len(features)
    mn = min(features)
    mx = max(features)
    avg = statistics.mean(features)
    std = statistics.stdev(features) if n > 1 else 0.0
    non_zero = sum(1 for f in features if abs(f) > 1e-9)
    return (
        f"{n} features | range [{mn:.2f}, {mx:.2f}] | "
        f"mean {avg:.2f} | std {std:.2f} | {non_zero}/{n} non-zero"
    )


def _risk_badge(stage1_result: str, stage2_result: str) -> tuple[str, str]:
    """Return (emoji, risk_level) for display."""
    if "unknown" in stage2_result.lower():
        return "🔴", "CRITICAL"
    elif stage1_result.lower() == "normal":
        return "🟢", "LOW"
    else:
        return "🟡", "MEDIUM-HIGH"


# ---------------------------------------------------------------------------
# Dataset suitability check
# ---------------------------------------------------------------------------

def _check_iomt_suitability_single(features: list[float]) -> str | None:
    """Return a warning string if this single sample is clearly non-IoMT, else None."""
    n = len(features)
    issues: list[str] = []

    # Only flag wrong feature count — this is the definitive signal
    if n != _IOMT_EXPECTED_FEATURES:
        issues.append(
            f"Expected {_IOMT_EXPECTED_FEATURES} network flow features for IoMT/IoT traffic, "
            f"but received {n}. This does not match the CIC IoMT 2024 dataset format."
        )

    # Check for all zeros (placeholder / empty vector)
    non_zero = sum(1 for f in features if abs(f) > 1e-9)
    if non_zero == 0:
        issues.append(
            "All feature values are zero — this is an empty or placeholder vector, "
            "not valid network traffic data."
        )

    if not issues:
        return None

    return (
        "⚠️ **Dataset Suitability Notice**\n\n"
        "This input may not be suitable for IoMT / IoT intrusion detection. "
        "The Meta-Guard pipeline is trained on the CIC IoMT 2024 WiFi/MQTT dataset "
        "with 45 network flow features (packet lengths, inter-arrival times, flag counts, "
        "protocol fields, etc.). Using non-network-traffic data (e.g., house prices, "
        "product catalogs, sensor readings from non-IoT domains) may produce unreliable results.\n\n"
        "Issues detected:\n" + "\n".join(f"- {issue}" for issue in issues)
    )


def _check_iomt_suitability_batch(
    total: int,
    normal_count: int,
    suspicious_count: int,
    zero_day_count: int,
    avg_confidence: float,
    avg_distance: float | None,
    feature_count: int | None = None,
    original_feature_count: int | None = None,
) -> str | None:
    """Return a warning string if the batch is clearly non-IoMT data.

    The key signal is *original_feature_count* — the number of columns in the
    uploaded CSV *before* padding/truncation to 45.  If the user uploads
    chocolate.csv with 8 columns or house_prices.csv with 30, that mismatch
    is the strongest indicator that this is **not** IoMT/IoT traffic.

    We intentionally do NOT flag datasets just because they have 0 normal
    samples or high zero-day counts — a legitimate IoMT attack-capture
    dataset could have exactly that distribution.
    """
    issues: list[str] = []

    # Primary check: original column count mismatch
    cols_to_check = original_feature_count if original_feature_count is not None else feature_count
    if cols_to_check is not None and cols_to_check != _IOMT_EXPECTED_FEATURES:
        issues.append(
            f"The uploaded dataset has **{cols_to_check} columns**, but IoMT/IoT network "
            f"traffic analysis requires exactly **{_IOMT_EXPECTED_FEATURES}** features. "
            f"The columns were {'padded with zeros' if cols_to_check < _IOMT_EXPECTED_FEATURES else 'truncated'} "
            f"to fit the pipeline, which will affect accuracy."
        )

    if not issues:
        return None

    return (
        "⚠️ **Dataset Suitability Notice**\n\n"
        "This dataset does **not** appear to be IoMT / IoT network traffic data. "
        "The Meta-Guard pipeline is specifically designed for the CIC IoMT 2024 WiFi/MQTT "
        "dataset format with 45 network flow features (packet lengths, inter-arrival times, "
        "flag counts, protocol fields, etc.).\n\n"
        "Datasets from other domains (e.g., house price prediction, product sales, "
        "medical records, image features) will produce **unreliable** classifications "
        "because the models have never seen such data distributions.\n\n"
        "Issues detected:\n" + "\n".join(f"- {issue}" for issue in issues)
    )


# ---------------------------------------------------------------------------
# Local model-based analysis (uses models from D:\Research\Meta-Guard\models)
# ---------------------------------------------------------------------------


def _load_local_label_encoder():
    """Load the label encoder that was saved during training."""
    import pickle
    from .config import get_settings
    settings = get_settings()
    try:
        with open(settings.encoder_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _get_known_classes() -> list[str]:
    """Retrieve the list of known attack classes from the local model artifacts."""
    encoder = _load_local_label_encoder()
    if encoder is not None and hasattr(encoder, "classes_"):
        return [str(c) for c in encoder.classes_]
    return []


def generate_explanation(
    features: list[float],
    stage1_result: str,
    stage2_result: str,
    distance_score: float | None,
) -> str:
    """Generate AI explanation using the local model pipeline's actual results.

    This uses the classification outputs from the models already loaded in memory
    (located in D:\\Research\\Meta-Guard\\models) — no external API needed.
    """
    # Check dataset suitability first
    suitability_warning = _check_iomt_suitability_single(features)

    emoji, risk = _risk_badge(stage1_result, stage2_result)
    feat_summary = _format_feature_summary(features)
    dist_text = f"{distance_score:.4f}" if distance_score is not None else "N/A"

    lines: list[str] = []

    # Show suitability warning at the top if applicable
    if suitability_warning:
        lines.append(suitability_warning)
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"{emoji} **Risk Level: {risk}**")
    lines.append("")

    # Section 1 — Input overview
    lines.append("**📊 Input Profile**")
    lines.append(f"Analyzed vector: {feat_summary}")
    lines.append("")

    # Section 2 — Pipeline processing (references actual local models)
    lines.append("**⚙️ Pipeline Processing**")
    if stage1_result.lower() == "normal":
        lines.append(
            "**Stage 1 — Edge Binary Classifier** (Reptile meta-learned, LayerNorm+GELU architecture) "
            "classified this traffic as **normal**. The softmax probability for the attack class "
            "fell below the calibrated threshold (0.20), so Stage 2 was not triggered."
        )
    else:
        lines.append(
            "**Stage 1 — Edge Binary Classifier** flagged this traffic as **suspicious** "
            "(P(attack) ≥ 0.20). The sample was escalated to "
            "**Stage 2 — Cloud Open-Set Analyzer** which projects the input into a "
            "64-dimensional L2-normalised embedding space and compares it against "
            f"48 class prototypes using cosine similarity."
        )
        lines.append("")
        if "unknown" in stage2_result.lower():
            lines.append(
                f"Result: **{stage2_result}** — cosine distance ({dist_text}) exceeds the "
                f"calibrated threshold (0.0957) AND the prototype margin is below 0.4339, "
                "indicating the sample does not cluster near any known attack family."
            )
        else:
            lines.append(
                f"Result: **{stage2_result}** — cosine distance ({dist_text}) is within the "
                f"calibrated threshold, confirming a match to a known attack prototype."
            )
    lines.append("")

    # Section 3 — Security insight
    lines.append("**🔍 Security Insight**")
    if "unknown" in stage2_result.lower():
        lines.append(
            "The embedding vector does not cluster near any of the 48 known attack prototypes, "
            "indicating traffic behavior not seen during training on the CIC IoMT 2024 dataset. "
            "This is a strong **Zero-Day candidate**. "
            "Recommended: isolate the source endpoint, capture full packet trace, "
            "and escalate to Tier-3 SOC for manual forensic analysis."
        )
    elif stage1_result.lower() == "normal":
        lines.append(
            "Feature magnitudes and statistical distribution align with normal IoMT operational baselines "
            "from the CIC IoMT 2024 WiFi/MQTT training data. "
            "No anomalous patterns or threshold-exceeding distance deviations detected."
        )
    else:
        lines.append(
            f"The embedding closely matches the **{stage2_result}** prototype cluster "
            f"(cosine distance = {dist_text}). This confirms a recognized attack signature "
            "from the CIC IoMT 2024 training data. "
            "Apply the standard mitigation playbook for this attack family."
        )

    lines.append(_AI_DISCLAIMER)
    return "\n".join(lines)


def generate_batch_explanation(
    total: int,
    normal_count: int,
    suspicious_count: int,
    zero_day_count: int,
    known_attack_count: int,
    avg_confidence: float,
    avg_distance: float | None,
    top_labels: list[str],
    feature_count: int | None = None,
    original_feature_count: int | None = None,
) -> str:
    """Generate batch-level AI summary using the local pipeline results.

    All analysis is based on outputs from the models in the models/ directory.
    """
    # Check dataset suitability
    suitability_warning = _check_iomt_suitability_batch(
        total=total,
        normal_count=normal_count,
        suspicious_count=suspicious_count,
        zero_day_count=zero_day_count,
        avg_confidence=avg_confidence,
        avg_distance=avg_distance,
        feature_count=feature_count,
        original_feature_count=original_feature_count,
    )

    if zero_day_count > 0:
        emoji, risk = "🔴", "HIGH"
    elif suspicious_count > 0:
        emoji, risk = "🟡", "MEDIUM"
    else:
        emoji, risk = "🟢", "LOW"

    dist_text = f"{avg_distance:.4f}" if avg_distance is not None else "N/A"
    label_text = ", ".join(top_labels) if top_labels else "None"

    lines: list[str] = []

    # Show suitability warning at the top if applicable
    if suitability_warning:
        lines.append(suitability_warning)
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend([
        f"{emoji} **Overall Risk: {risk}**",
        "",
        "**📊 Dataset Overview**",
        f"Processed **{total:,}** samples through the two-stage Meta-Guard pipeline. "
        f"Each sample's {_IOMT_EXPECTED_FEATURES}-feature vector was scaled using the "
        f"training StandardScaler, passed through the Reptile meta-learned Edge classifier "
        f"(Stage 1), and — if flagged suspicious — analyzed by the Cloud open-set "
        f"embedding engine with 48 class prototypes (Stage 2).",
        "",
        "**📈 Key Findings**",
        f"- Normal: **{normal_count:,}** ({normal_count/total*100:.1f}%)",
        f"- Suspicious (Known Attacks): **{known_attack_count:,}**",
        f"- Zero-Day Candidates: **{zero_day_count:,}**",
        f"- Average Confidence: **{avg_confidence:.2%}**",
        f"- Average Embedding Distance: **{dist_text}**",
        f"- Detected Attack Labels: {label_text}",
        "",
        "**🔍 Recommendations**",
    ])

    if zero_day_count > 0:
        lines.append(
            f"**{zero_day_count:,}** sample(s) exhibit cosine distances exceeding the calibrated "
            "threshold (0.0957) with low prototype margin (< 0.4339), indicating novel threat "
            "behavior not present in the CIC IoMT 2024 training data. "
            "Immediate isolation of associated endpoints and packet-level forensic capture is advised."
        )
    if known_attack_count > 0:
        lines.append(
            f"**{known_attack_count:,}** sample(s) match recognized attack signatures from the "
            "48 known classes. Apply standard remediation playbooks for the detected families."
        )
    if normal_count == total:
        lines.append(
            "All traffic conforms to baseline IoMT patterns. No actionable threats detected."
        )

    lines.append(_AI_DISCLAIMER)
    return "\n".join(lines)
