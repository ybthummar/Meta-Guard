from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import get_settings
from .pipeline import MetaGuardPipeline
from .schemas import (
    BatchPredictionResponse,
    EvaluationSummary,
    GenerateRequest,
    GenerateResponse,
    PredictionResult,
)
from .ai_explainer import generate_explanation, generate_batch_explanation
from .data_generator import GenerationConfig, generate_synthetic_dataset
from .gemini_explainer import (
    gemini_batch_explanation,
    gemini_dataset_summary,
    is_gemini_available,
)


app = FastAPI(
    title="Meta-Guard IDS API",
    description="Two-stage IoMT intrusion detection demo (edge binary detector + cloud open-set analysis).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    settings = get_settings()
    app.state.pipeline = MetaGuardPipeline.from_settings(settings)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def get_metadata() -> dict[str, object]:
    pipeline = _get_pipeline()
    meta = pipeline.metadata()
    meta["gemini_available"] = is_gemini_available()
    return meta


@app.post("/predict", response_model=PredictionResult | BatchPredictionResponse)
def predict(payload: Any = Body(...)) -> PredictionResult | BatchPredictionResponse:
    pipeline = _get_pipeline()

    try:
        features, is_single = _normalize_payload(payload)
        raw_results = pipeline.predict(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected inference error: {exc}") from exc

    # Extract original feature count if provided by frontend (for suitability check)
    original_feature_count = None
    if isinstance(payload, dict):
        original_feature_count = payload.get("original_feature_count")
    
    # Generate optional AI explanation and determine final decision
    for i, res in enumerate(raw_results):
        if res.get("zero_day"):
            final_decision = "Zero-Day Alert"
        elif res["stage1"] == "normal":
            final_decision = "Normal"
        else:
            final_decision = "Known Attack"
            
        res["final_decision"] = final_decision
        
        ai_exp = generate_explanation(
            features=features[i],
            stage1_result=res["stage1"],
            stage2_result=res["stage2"],
            distance_score=res.get("distance")
        )
        res["ai_analysis"] = ai_exp

    if is_single:
        return PredictionResult.model_validate(raw_results[0])

    suspicious_count = sum(1 for item in raw_results if item["stage1"] == "suspicious")
    zero_day_count = sum(1 for item in raw_results if item["zero_day"])
    normal_count = sum(1 for item in raw_results if item["stage1"] == "normal")
    known_attack_count = suspicious_count - zero_day_count

    # Compute batch-level stats for AI summary
    confidences = [float(item["confidence"]) for item in raw_results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    distances = [float(item["distance"]) for item in raw_results if item.get("distance") is not None]
    avg_distance = sum(distances) / len(distances) if distances else None

    # Collect unique predicted labels
    top_labels = list({
        str(item["predicted_label"])
        for item in raw_results
        if item.get("predicted_label") is not None
    })

    batch_summary = generate_batch_explanation(
        total=len(raw_results),
        normal_count=normal_count,
        suspicious_count=suspicious_count,
        zero_day_count=zero_day_count,
        known_attack_count=known_attack_count,
        avg_confidence=avg_confidence,
        avg_distance=avg_distance,
        top_labels=top_labels,
        feature_count=len(features[0]) if features else None,
        original_feature_count=original_feature_count,
    )

    return BatchPredictionResponse(
        count=len(raw_results),
        suspicious_count=suspicious_count,
        zero_day_count=zero_day_count,
        results=[PredictionResult.model_validate(item) for item in raw_results],
        batch_ai_summary=batch_summary,
    )


def _get_pipeline() -> MetaGuardPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized yet.")
    return pipeline


# ---------------------------------------------------------------------------
# Synthetic Data Generation & Evaluation
# ---------------------------------------------------------------------------

_GENERATED_DIR = Path(__file__).resolve().parents[2] / "generated"


@app.post("/generate", response_model=GenerateResponse)
def generate_dataset(req: GenerateRequest) -> GenerateResponse:
    """Generate synthetic IoMT dataset based on user-specified counts."""
    pipeline = _get_pipeline()

    total = req.num_normal + req.num_known + req.num_zero_day
    if total == 0:
        raise HTTPException(status_code=400, detail="At least one sample type must be > 0.")

    try:
        config = GenerationConfig(
            num_normal=req.num_normal,
            num_known=req.num_known,
            num_zero_day=req.num_zero_day,
            seed=req.seed,
        )
        df = generate_synthetic_dataset(pipeline, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {exc}") from exc

    # Save CSV
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _GENERATED_DIR / "generated_dataset.csv"
    df.to_csv(csv_path, index=False)

    # Store in app state for evaluation
    app.state.generated_df = df
    app.state.generated_csv_path = csv_path

    # Gemini dataset summary (optional, non-blocking)
    gemini_summary = gemini_dataset_summary(
        req.num_normal, req.num_known, req.num_zero_day, total
    )

    preview = df.head(10).to_dict(orient="records")
    msg = f"Generated {total} synthetic samples ({req.num_normal} normal, {req.num_known} known, {req.num_zero_day} zero-day)."
    if gemini_summary:
        msg += f"\n\n**Gemini Analysis:** {gemini_summary}"

    return GenerateResponse(
        total_samples=total,
        num_normal=req.num_normal,
        num_known=req.num_known,
        num_zero_day=req.num_zero_day,
        csv_filename="generated_dataset.csv",
        preview=preview,
        message=msg,
    )


@app.get("/download-dataset")
def download_dataset():
    """Download the most recently generated dataset CSV."""
    csv_path = getattr(app.state, "generated_csv_path", None)
    if csv_path is None or not Path(csv_path).exists():
        raise HTTPException(status_code=404, detail="No generated dataset found. Run /generate first.")
    return FileResponse(
        path=str(csv_path),
        filename="generated_dataset.csv",
        media_type="text/csv",
    )


@app.post("/evaluate", response_model=EvaluationSummary)
def evaluate_dataset(custom_threshold: float | None = Body(default=None, embed=True)) -> EvaluationSummary:
    """Run all generated samples through Meta-Guard and compute evaluation metrics."""
    pipeline = _get_pipeline()

    generated_df = getattr(app.state, "generated_df", None)
    if generated_df is None:
        raise HTTPException(status_code=400, detail="No generated dataset found. Run /generate first.")

    # Extract feature columns (exclude ground_truth)
    feature_cols = [c for c in generated_df.columns if c != "ground_truth"]
    features = generated_df[feature_cols].values.tolist()
    ground_truths = generated_df["ground_truth"].tolist() if "ground_truth" in generated_df.columns else None

    # Optionally override distance threshold for demo
    original_threshold = pipeline.cloud_analyzer.distance_threshold
    if custom_threshold is not None:
        pipeline.cloud_analyzer.distance_threshold = custom_threshold

    try:
        raw_results = pipeline.predict(features)
    except Exception as exc:
        pipeline.cloud_analyzer.distance_threshold = original_threshold
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
    finally:
        if custom_threshold is not None:
            pipeline.cloud_analyzer.distance_threshold = original_threshold

    # Compute final decisions and AI explanations
    for i, res in enumerate(raw_results):
        if res.get("zero_day"):
            res["final_decision"] = "Zero-Day Alert"
        elif res["stage1"] == "normal":
            res["final_decision"] = "Normal"
        else:
            res["final_decision"] = "Known Attack"

        if ground_truths:
            res["ground_truth"] = ground_truths[i]

        ai_exp = generate_explanation(
            features=features[i],
            stage1_result=res["stage1"],
            stage2_result=res["stage2"],
            distance_score=res.get("distance"),
        )
        res["ai_analysis"] = ai_exp

    # Aggregate counts
    normal_detected = sum(1 for r in raw_results if r["stage1"] == "normal")
    suspicious_count = sum(1 for r in raw_results if r["stage1"] == "suspicious")
    zero_day_detected = sum(1 for r in raw_results if r.get("zero_day"))
    known_detected = suspicious_count - zero_day_detected

    # Batch AI summary
    confidences = [float(r["confidence"]) for r in raw_results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    distances = [float(r["distance"]) for r in raw_results if r.get("distance") is not None]
    avg_distance = sum(distances) / len(distances) if distances else None
    top_labels = list({str(r["predicted_label"]) for r in raw_results if r.get("predicted_label")})

    batch_summary = generate_batch_explanation(
        total=len(raw_results),
        normal_count=normal_detected,
        suspicious_count=suspicious_count,
        zero_day_count=zero_day_detected,
        known_attack_count=known_detected,
        avg_confidence=avg_confidence,
        avg_distance=avg_distance,
        top_labels=top_labels,
        feature_count=len(features[0]) if features else None,
    )

    # Gemini batch explanation (optional)
    gemini_text = gemini_batch_explanation(
        total=len(raw_results),
        normal_detected=normal_detected,
        known_detected=known_detected,
        zeroday_detected=zero_day_detected,
        avg_confidence=avg_confidence,
        avg_distance=avg_distance,
        top_labels=top_labels,
    )

    # Build comparison table
    comparison = None
    if ground_truths:
        comparison = []
        for i, res in enumerate(raw_results):
            comparison.append({
                "sample": i + 1,
                "ground_truth": ground_truths[i],
                "predicted": res["final_decision"],
                "match": ground_truths[i].lower().replace("-", "").replace(" ", "")
                         in res["final_decision"].lower().replace("-", "").replace(" ", ""),
                "confidence": res["confidence"],
                "distance": res.get("distance"),
            })

    return EvaluationSummary(
        total_samples=len(raw_results),
        normal_detected=normal_detected,
        known_attacks_detected=known_detected,
        zero_day_detected=zero_day_detected,
        results=[PredictionResult.model_validate(r) for r in raw_results],
        comparison=comparison,
        batch_ai_summary=batch_summary,
        gemini_analysis=gemini_text,
    )


def _robust_feature_fix(features: list[Any], expected_size: int = 45) -> list[float]:
    cleaned = []
    for f in features:
        try:
            cleaned.append(float(f))
        except (ValueError, TypeError):
            cleaned.append(0.0)
            
    if len(cleaned) > expected_size:
        return cleaned[:expected_size]
    elif len(cleaned) < expected_size:
        return cleaned + [0.0] * (expected_size - len(cleaned))
    return cleaned

def _normalize_payload(payload: Any) -> tuple[list[list[float]], bool]:
    pipeline = _get_pipeline()
    expected_size = pipeline.metadata().get("expected_features", 45)

    raw_features = payload

    if isinstance(payload, Mapping):
        if "features" in payload:
            raw_features = payload["features"]
        else:
            raw_features = payload

    if isinstance(raw_features, Mapping):
        single_vector = _mapping_to_vector(raw_features)
        return [_robust_feature_fix(single_vector, expected_size)], True

    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raw_features = [0.0] * expected_size # gracefully fallback instead of error

    if len(raw_features) == 0:
        raw_features = [[0.0] * expected_size]  # ensure it never crashes

    first_item = raw_features[0]

    if isinstance(first_item, Mapping):
        return [_robust_feature_fix(_mapping_to_vector(item), expected_size) for item in raw_features], False

    if isinstance(first_item, Sequence) and not isinstance(first_item, (str, bytes)):
        return [_robust_feature_fix(list(row), expected_size) for row in raw_features], False

    return [_robust_feature_fix(list(raw_features), expected_size)], True

def _mapping_to_vector(feature_mapping: Mapping[str, Any]) -> list[Any]:
    keys = list(feature_mapping.keys())
    if not keys:
        raise ValueError("Feature mapping is empty.")

    # Keep f1, f2, ..., fN in natural order when provided.
    if all(key.startswith("f") and key[1:].isdigit() for key in keys):
        ordered_keys = sorted(keys, key=lambda item: int(item[1:]))
    else:
        ordered_keys = sorted(keys)

    try:
        return [feature_mapping[key] for key in ordered_keys]
    except Exception as exc:
        raise ValueError(f"Value error in mapping payload: {exc}") from exc
