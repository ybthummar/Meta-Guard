from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv

# Search for .env explicitly at project root
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_dir: Path
    level1_model_path: Path
    level2_model_path: Path
    scaler_path: Path
    encoder_path: Path
    prototypes_path: Path
    edge_threshold: float
    open_set_distance_threshold: float | None
    open_set_margin_threshold: float | None
    open_set_threshold_multiplier: float


def _read_optional_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return float(raw_value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    model_dir_raw = os.getenv("MODEL_DIR")
    if model_dir_raw is None:
        model_dir = (project_root / "models").resolve()
    else:
        model_dir = Path(model_dir_raw)
        if not model_dir.is_absolute():
            model_dir = (project_root / model_dir).resolve()

    return Settings(
        project_root=project_root,
        model_dir=model_dir,
        level1_model_path=model_dir / os.getenv("LEVEL1_MODEL_FILE", "metaguard_level1_reptile.pt.zip"),
        level2_model_path=model_dir / os.getenv("LEVEL2_MODEL_FILE", "metaguard_level2_openset.pt.zip"),
        scaler_path=model_dir / os.getenv("SCALER_FILE", "metaguard_scaler.pkl"),
        encoder_path=model_dir / os.getenv("ENCODER_FILE", "metaguard_level2_encoder.pkl"),
        prototypes_path=model_dir / os.getenv("PROTOTYPES_FILE", "metaguard_prototypes.pt.zip"),
        edge_threshold=float(os.getenv("EDGE_THRESHOLD", "0.20")),
        open_set_distance_threshold=_read_optional_float("OPENSET_DISTANCE_THRESHOLD"),
        open_set_margin_threshold=_read_optional_float("OPENSET_MARGIN_THRESHOLD"),
        open_set_threshold_multiplier=float(os.getenv("OPENSET_THRESHOLD_MULTIPLIER", "1.10")),
    )
