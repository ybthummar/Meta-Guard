from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    stage1: str = Field(description="Stage-1 result: normal or suspicious")
    stage2: str = Field(description="Stage-2 result: known attack, unknown attack, or not_applicable")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for final decision")
    distance: float | None = Field(default=None, ge=0.0, description="Embedding distance to predicted prototype")
    stage1_confidence: float = Field(ge=0.0, le=1.0, description="Confidence for edge-stage decision")
    predicted_label: str | None = Field(default=None, description="Known class label when applicable")
    closest_known_label: str | None = Field(default=None, description="Closest known class in embedding space")
    zero_day: bool = Field(description="True when classified as unknown/zero-day attack")
    threshold: float | None = Field(default=None, ge=0.0, description="Distance threshold used in stage-2")
    final_decision: str | None = Field(default=None, description="Final status: Normal, Known Attack, or Zero-Day Alert")
    ai_analysis: str | None = Field(default=None, description="AI-generated cyber reasoning (optional)")
    ground_truth: str | None = Field(default=None, description="Intended category from synthetic generation (demo only)")


class BatchPredictionResponse(BaseModel):
    count: int = Field(ge=1)
    suspicious_count: int = Field(ge=0)
    zero_day_count: int = Field(ge=0)
    results: list[PredictionResult]
    batch_ai_summary: str | None = Field(default=None, description="AI-generated batch-level analysis")
    gemini_analysis: str | None = Field(default=None, description="Gemini AI explanation (optional)")


class GenerateRequest(BaseModel):
    num_normal: int = Field(default=30, ge=0, le=500, description="Number of normal traffic samples")
    num_known: int = Field(default=20, ge=0, le=500, description="Number of known attack samples")
    num_zero_day: int = Field(default=50, ge=0, le=500, description="Number of zero-day samples")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")


class GenerateResponse(BaseModel):
    total_samples: int
    num_normal: int
    num_known: int
    num_zero_day: int
    csv_filename: str
    preview: list[dict]
    message: str


class EvaluationSummary(BaseModel):
    total_samples: int
    normal_detected: int
    known_attacks_detected: int
    zero_day_detected: int
    results: list[PredictionResult]
    comparison: list[dict] | None = Field(default=None, description="Ground truth vs prediction comparison")
    batch_ai_summary: str | None = None
    gemini_analysis: str | None = None
