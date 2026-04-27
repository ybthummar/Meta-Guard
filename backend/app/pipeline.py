from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch

from .config import Settings
from .models import build_level1_from_state_dict, build_level2_from_state_dict


@dataclass
class EdgePrediction:
    is_suspicious: bool
    confidence: float


class EdgeDetector:
    """Runs the lightweight stage-1 model close to IoMT edge devices.

    Uses an explicit probability threshold on P(attack) — matching the
    notebook's ``max(lvl1_th, 0.20)`` decision rule.
    """

    def __init__(self, model: torch.nn.Module, threshold: float = 0.20) -> None:
        self.model = model
        self.threshold = threshold

    def predict(self, scaled_features: np.ndarray) -> list[EdgePrediction]:
        batch_tensor = torch.as_tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        results: list[EdgePrediction] = []
        for probs in probabilities:
            suspicious_prob = float(probs[1])
            is_suspicious = suspicious_prob >= self.threshold
            confidence = suspicious_prob if is_suspicious else float(probs[0])
            results.append(EdgePrediction(is_suspicious=is_suspicious, confidence=confidence))
        return results


class CloudOpenSetAnalyzer:
    """Runs stage-2 open-set analysis using cosine similarity in embedding space.

    Decision rule (matching the notebook exactly):
      zero-day  ←  cosine_distance > dist_threshold  AND  margin < margin_threshold
      known     ←  otherwise

    Where:
      cosine_distance = 1 - max_cosine_similarity_to_prototypes
      margin          = top1_similarity - top2_similarity
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prototypes: torch.Tensor,
        label_encoder: object | None,
        distance_threshold: float | None,
        margin_threshold: float | None,
        threshold_multiplier: float,
    ) -> None:
        self.model = model
        self.prototypes = prototypes.detach().cpu().float()
        self.label_encoder = label_encoder
        self.threshold_multiplier = threshold_multiplier

        self.distance_threshold = (
            distance_threshold
            if distance_threshold is not None
            else self._auto_distance_threshold(self.prototypes, threshold_multiplier)
        )
        self.margin_threshold = (
            margin_threshold
            if margin_threshold is not None
            else 0.50
        )

    @staticmethod
    def _auto_distance_threshold(prototypes: torch.Tensor, threshold_multiplier: float) -> float:
        """Fallback threshold derived from inter-prototype cosine distances."""
        if prototypes.ndim != 2 or prototypes.shape[0] < 2:
            return 1.0

        sim_matrix = prototypes @ prototypes.t()
        dist_matrix = 1.0 - sim_matrix
        diagonal_mask = torch.eye(dist_matrix.shape[0], dtype=torch.bool)
        dist_matrix = dist_matrix.masked_fill(diagonal_mask, float("inf"))

        nn_distances = dist_matrix.min(dim=1).values
        nn_mean = float(nn_distances.mean().item())
        nn_std = float(nn_distances.std().item()) if nn_distances.shape[0] > 1 else 0.0
        base_threshold = nn_mean + 2.0 * nn_std
        return max(base_threshold * threshold_multiplier, 1e-6)

    def _decode_label(self, class_index: int) -> str:
        if self.label_encoder is not None and hasattr(self.label_encoder, "inverse_transform"):
            try:
                decoded = self.label_encoder.inverse_transform(np.array([class_index]))
                return str(decoded[0])
            except Exception:
                pass
        return f"class_{class_index}"

    def analyze(self, scaled_features: np.ndarray) -> list[dict[str, object]]:
        if scaled_features.size == 0:
            return []

        batch_tensor = torch.as_tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            embedding, logits = self.model.forward_with_embedding(batch_tensor)

        embedding = embedding.detach().cpu()

        # Cosine similarity (both embedding and prototypes are L2-normalised)
        sim = embedding @ self.prototypes.t()

        k = min(2, sim.shape[1])
        top2_vals, top2_idx = torch.topk(sim, k=k, dim=1)
        top1_sim = top2_vals[:, 0]
        top2_sim = top2_vals[:, 1] if k > 1 else top2_vals[:, 0]
        nearest_indices = top2_idx[:, 0]

        distance = 1.0 - top1_sim
        margin = top1_sim - top2_sim

        # Dual-threshold open-set decision (matches notebook exactly)
        is_zero_day = (distance > self.distance_threshold) & (margin < self.margin_threshold)
        known_mask = ~is_zero_day

        outputs: list[dict[str, object]] = []
        for i in range(embedding.shape[0]):
            nearest_idx = int(nearest_indices[i].item())
            is_known = bool(known_mask[i].item())
            dist_val = float(distance[i].item())

            if is_known:
                confidence = max(0.0, min(1.0, float(top1_sim[i].item())))
            else:
                confidence = float(torch.sigmoid(distance[i] - self.distance_threshold).item())

            outputs.append(
                {
                    "stage2": "known attack" if is_known else "unknown attack",
                    "confidence": confidence,
                    "distance": dist_val,
                    "predicted_label": self._decode_label(nearest_idx) if is_known else None,
                    "closest_known_label": self._decode_label(nearest_idx),
                    "zero_day": not is_known,
                    "threshold": float(self.distance_threshold),
                }
            )
        return outputs


class MetaGuardPipeline:
    """End-to-end pipeline that orchestrates edge and cloud stages."""

    def __init__(
        self,
        scaler: object,
        edge_detector: EdgeDetector,
        cloud_analyzer: CloudOpenSetAnalyzer,
        label_encoder: object | None = None,
    ) -> None:
        self.scaler = scaler
        self.edge_detector = edge_detector
        self.cloud_analyzer = cloud_analyzer
        self.label_encoder = label_encoder
        self.expected_features = int(getattr(self.scaler, "n_features_in_", 45))

    @classmethod
    def from_settings(cls, settings: Settings) -> "MetaGuardPipeline":
        scaler = _load_pickle_artifact(settings.scaler_path)
        label_encoder = _load_pickle_artifact(settings.encoder_path)

        level1_state = _load_torch_artifact(settings.level1_model_path)
        level2_state = _load_torch_artifact(settings.level2_model_path)
        prototypes = _load_torch_artifact(settings.prototypes_path)

        if not isinstance(level1_state, dict):
            raise RuntimeError("Level-1 artifact does not contain a valid state_dict.")
        if not isinstance(level2_state, dict):
            raise RuntimeError("Level-2 artifact does not contain a valid state_dict.")
        if not isinstance(prototypes, torch.Tensor):
            raise RuntimeError("Prototype artifact must be a Tensor.")

        level1_model = build_level1_from_state_dict(level1_state)
        level2_model = build_level2_from_state_dict(level2_state)

        edge_detector = EdgeDetector(level1_model, threshold=settings.edge_threshold)
        cloud_analyzer = CloudOpenSetAnalyzer(
            model=level2_model,
            prototypes=prototypes,
            label_encoder=label_encoder,
            distance_threshold=settings.open_set_distance_threshold,
            margin_threshold=settings.open_set_margin_threshold,
            threshold_multiplier=settings.open_set_threshold_multiplier,
        )

        return cls(
            scaler=scaler,
            edge_detector=edge_detector,
            cloud_analyzer=cloud_analyzer,
            label_encoder=label_encoder,
        )

    def metadata(self) -> dict[str, object]:
        known_classes: list[str] = []
        if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            known_classes = [str(item) for item in self.label_encoder.classes_]

        return {
            "expected_features": self.expected_features,
            "known_class_count": len(known_classes),
            "known_classes": known_classes,
            "distance_threshold": float(self.cloud_analyzer.distance_threshold),
        }

    def preprocess_features(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if len(features) == 0:
            raise ValueError("Input is empty. Provide at least one sample.")

        try:
            feature_array = np.asarray(features, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Unable to parse features as numeric values: {exc}") from exc

        if feature_array.ndim != 2:
            raise ValueError("Input must be a 2D list-like structure: [samples][features].")

        if feature_array.shape[1] != self.expected_features:
            raise ValueError(
                f"Invalid feature length: expected {self.expected_features}, got {feature_array.shape[1]}."
            )

        try:
            scaled = self.scaler.transform(feature_array)
        except Exception as exc:
            raise RuntimeError(f"Scaler transform failed: {exc}") from exc

        return np.asarray(scaled, dtype=np.float32)

    def predict(self, features: Sequence[Sequence[float]]) -> list[dict[str, object]]:
        scaled_features = self.preprocess_features(features)
        edge_predictions = self.edge_detector.predict(scaled_features)

        results: list[dict[str, object] | None] = [None] * len(edge_predictions)
        suspicious_indices: list[int] = []

        for index, edge_prediction in enumerate(edge_predictions):
            if edge_prediction.is_suspicious:
                suspicious_indices.append(index)
                continue

            results[index] = {
                "stage1": "normal",
                "stage2": "not_applicable",
                "confidence": edge_prediction.confidence,
                "distance": None,
                "stage1_confidence": edge_prediction.confidence,
                "predicted_label": None,
                "closest_known_label": None,
                "zero_day": False,
                "threshold": None,
            }

        if suspicious_indices:
            suspicious_features = scaled_features[suspicious_indices]
            cloud_predictions = self.cloud_analyzer.analyze(suspicious_features)

            for local_row, global_index in enumerate(suspicious_indices):
                edge_prediction = edge_predictions[global_index]
                cloud_result = cloud_predictions[local_row]
                results[global_index] = {
                    "stage1": "suspicious",
                    "stage2": cloud_result["stage2"],
                    "confidence": float(cloud_result["confidence"]),
                    "distance": cloud_result["distance"],
                    "stage1_confidence": edge_prediction.confidence,
                    "predicted_label": cloud_result["predicted_label"],
                    "closest_known_label": cloud_result["closest_known_label"],
                    "zero_day": cloud_result["zero_day"],
                    "threshold": cloud_result["threshold"],
                }

        return [item for item in results if item is not None]


def _load_pickle_artifact(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")

    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def _load_torch_artifact(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
