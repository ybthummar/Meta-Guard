from __future__ import annotations

import json
import warnings
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is importable when the script is executed directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import get_settings
from backend.app.pipeline import MetaGuardPipeline


def evaluate_sample(pipeline: MetaGuardPipeline, raw_vector: np.ndarray) -> dict[str, object]:
    return pipeline.predict([raw_vector.astype(float).tolist()])[0]


def optimize_normal_sample(
    pipeline: MetaGuardPipeline,
    scaler: object,
    steps: int = 500,
    lr: float = 0.06,
) -> tuple[np.ndarray, dict[str, object]]:
    feature_dim = pipeline.expected_features
    edge_model = pipeline.edge_detector.model

    x = torch.zeros((1, feature_dim), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([0], dtype=torch.long)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = edge_model(x)
        loss = F.cross_entropy(logits, target) + 1e-3 * (x ** 2).mean()
        loss.backward()
        optimizer.step()

    scaled = x.detach().cpu().numpy()
    raw = scaler.inverse_transform(scaled)[0]
    result = evaluate_sample(pipeline, raw)
    return raw, result


def optimize_known_sample(
    pipeline: MetaGuardPipeline,
    scaler: object,
    steps: int = 900,
    lr: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]] | None:
    feature_dim = pipeline.expected_features
    edge_model = pipeline.edge_detector.model
    level2_model = pipeline.cloud_analyzer.model
    prototypes = pipeline.cloud_analyzer.prototypes

    num_classes = prototypes.shape[0]
    suspicious_target = torch.tensor([1], dtype=torch.long)

    candidate_classes = np.random.permutation(num_classes)[: min(16, num_classes)]
    for class_index in candidate_classes:
        prototype = prototypes[class_index : class_index + 1]
        class_target = torch.tensor([class_index], dtype=torch.long)

        for _ in range(3):
            x = torch.randn((1, feature_dim), dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()
                edge_logits = edge_model(x)
                embedding, level2_logits = level2_model.forward_with_embedding(x)

                distance = torch.norm(embedding - prototype, dim=1).mean()
                level2_ce = F.cross_entropy(level2_logits, class_target)
                level1_ce = F.cross_entropy(edge_logits, suspicious_target)

                loss = distance + 0.20 * level2_ce + 0.35 * level1_ce + 1e-4 * (x ** 2).mean()
                loss.backward()
                optimizer.step()

            scaled = x.detach().cpu().numpy()
            raw = scaler.inverse_transform(scaled)[0]
            result = evaluate_sample(pipeline, raw)

            if result["stage1"] == "suspicious" and result["stage2"] == "known attack":
                return raw, result

    return None


def optimize_unknown_sample(
    pipeline: MetaGuardPipeline,
    scaler: object,
    steps: int = 500,
    lr: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    feature_dim = pipeline.expected_features
    edge_model = pipeline.edge_detector.model
    level2_model = pipeline.cloud_analyzer.model
    prototypes = pipeline.cloud_analyzer.prototypes

    suspicious_target = torch.tensor([1], dtype=torch.long)

    x = torch.randn((1, feature_dim), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        edge_logits = edge_model(x)
        embedding, _ = level2_model.forward_with_embedding(x)

        min_distance = torch.cdist(embedding, prototypes).min(dim=1).values.mean()
        level1_ce = F.cross_entropy(edge_logits, suspicious_target)

        loss = -min_distance + 0.35 * level1_ce + 1e-4 * (x ** 2).mean()
        loss.backward()
        optimizer.step()

    scaled = x.detach().cpu().numpy()
    raw = scaler.inverse_transform(scaled)[0]
    result = evaluate_sample(pipeline, raw)
    return raw, result


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(7)
    torch.manual_seed(7)

    settings = get_settings()
    pipeline = MetaGuardPipeline.from_settings(settings)
    scaler = pipeline.scaler

    normal_raw, normal_result = optimize_normal_sample(pipeline, scaler)

    known_output = optimize_known_sample(pipeline, scaler)
    if known_output is None:
        raise RuntimeError("Unable to synthesize a known-attack sample with current model settings.")
    known_raw, known_result = known_output

    unknown_raw, unknown_result = optimize_unknown_sample(pipeline, scaler)

    payload = {
        "normal": normal_raw.astype(float).tolist(),
        "known_attack": known_raw.astype(float).tolist(),
        "unknown_attack": unknown_raw.astype(float).tolist(),
    }

    output_path = Path(__file__).resolve().parents[1] / "examples" / "sample_inputs.json"
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    print("Saved sample payloads to", output_path)
    print("normal:", normal_result)
    print("known_attack:", known_result)
    print("unknown_attack:", unknown_result)


if __name__ == "__main__":
    main()
