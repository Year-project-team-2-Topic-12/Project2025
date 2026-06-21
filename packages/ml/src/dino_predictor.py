from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)


MURA_ANATOMIES = (
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST",
)


class SquarePadResize:
    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        w, h = image.size
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), color=(self.fill, self.fill, self.fill))
        canvas.paste(image, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


class MuraDinoClassifier:  # real base class is injected lazily in _build_mura_dino_classifier
    pass


def _build_mura_dino_classifier():
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class _MuraDinoClassifier(nn.Module):
        def __init__(
            self,
            model_name: str,
            bone_categories: list[str],
            dropout: float = 0.30,
            dino_model: nn.Module | None = None,
        ):
            super().__init__()
            self.model_name = model_name
            self.bone_categories = list(bone_categories)
            self.dino = dino_model if dino_model is not None else AutoModel.from_pretrained(model_name)
            hidden_size = int(self.dino.config.hidden_size)
            self.feature_dim = hidden_size * 2
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(0.20),
            )
            self.classifiers = nn.ModuleDict({bone: nn.Linear(128, 2) for bone in self.bone_categories})
            self.freeze_backbone()

        def freeze_backbone(self) -> None:
            for param in self.dino.parameters():
                param.requires_grad = False

        def transformer_layers(self):
            if hasattr(self.dino, "encoder") and hasattr(self.dino.encoder, "layer"):
                return list(self.dino.encoder.layer)
            raise AttributeError("Unsupported DINO backbone: cannot find dino.encoder.layer")

        def set_trainable_last_n_blocks(self, n_blocks: int) -> None:
            self.freeze_backbone()
            if n_blocks <= 0:
                return
            layers = self.transformer_layers()
            for layer in layers[-n_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

        def forward_features(self, x):
            outputs = self.dino(pixel_values=x)
            tokens = outputs.last_hidden_state
            cls_token = tokens[:, 0, :]
            patch_mean = tokens[:, 1:, :].mean(dim=1)
            return self.head(torch.cat([cls_token, patch_mean], dim=1))

        def forward(self, x, anatomies):
            feat = self.forward_features(x)
            logits = torch.empty((x.size(0), 2), dtype=feat.dtype, device=feat.device)
            anatomies = list(anatomies)
            for bone in sorted(set(anatomies)):
                idx = [i for i, value in enumerate(anatomies) if value == bone]
                idx_t = torch.tensor(idx, dtype=torch.long, device=feat.device)
                logits[idx_t] = self.classifiers[bone](feat[idx_t])
            return logits

    _MuraDinoClassifier.__name__ = "MuraDinoClassifier"
    _MuraDinoClassifier.__qualname__ = "MuraDinoClassifier"
    _MuraDinoClassifier.__module__ = "__main__"
    return _MuraDinoClassifier


MuraDinoClassifier = _build_mura_dino_classifier()


@dataclass(frozen=True)
class DinoPredictorConfig:
    tracking_uri: str
    model_name: str
    model_alias: str
    model_version: str
    model_uri: str
    image_size: int
    default_threshold: float
    threshold_override: float | None
    batch_size: int
    device: str

    @classmethod
    def from_env(cls) -> "DinoPredictorConfig":
        load_dotenv()
        threshold_override_raw = os.getenv("MURA_DINO_THRESHOLD", "").strip()
        threshold_override = float(threshold_override_raw) if threshold_override_raw else None
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"),
            model_name=os.getenv("MLFLOW_MODEL_NAME", "mura_dinov2_transformer"),
            model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "prd"),
            model_version=os.getenv("MLFLOW_MODEL_VERSION", "2"),
            model_uri=os.getenv("MLFLOW_MODEL_URI", ""),
            image_size=int(os.getenv("MURA_DINO_IMAGE_SIZE", "448")),
            default_threshold=float(os.getenv("MURA_DINO_DEFAULT_THRESHOLD", "0.5")),
            threshold_override=threshold_override,
            batch_size=int(os.getenv("MURA_DINO_BATCH_SIZE", "16")),
            device=os.getenv("MURA_DINO_DEVICE", "cpu"),
        )

    def resolved_model_uri(self) -> str:
        if self.model_uri:
            return self.model_uri
        if self.model_version:
            return f"models:/{self.model_name}/{self.model_version}"
        return f"models:/{self.model_name}@{self.model_alias}"


@dataclass
class DinoImagePrediction:
    filename: str | None
    anatomy: str
    probability: float
    prediction: int
    threshold: float
    original_size: tuple[int, int]
    original_image: Image.Image
    processed_image: Image.Image

    @property
    def confidence(self) -> float:
        return self.probability if self.prediction == 1 else 1.0 - self.probability


class DinoMlflowPredictor:
    def __init__(self, config: DinoPredictorConfig | None = None):
        self.config = config or DinoPredictorConfig.from_env()
        self.model_uri = self.config.resolved_model_uri()

        import mlflow
        import torch
        from torchvision import transforms

        mlflow.set_tracking_uri(self.config.tracking_uri)
        self.torch = torch
        self.device = self._resolve_device()
        self.pad_resize = SquarePadResize(self.config.image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = self._load_model()
        self.threshold = self._resolve_threshold()
        logger.info(
            "Loaded MURA DINO model %s on %s with threshold %.3f",
            self.model_uri,
            self.device,
            self.threshold,
        )

    def _resolve_device(self):
        requested = self.config.device.strip().lower()
        if requested == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            logger.warning("MURA_DINO_DEVICE=%s requested, but CUDA is unavailable; falling back to CPU", requested)
            return self.torch.device("cpu")
        return self.torch.device(requested)

    def _load_model(self):
        import mlflow.pytorch

        self._register_pickle_aliases()
        try:
            loaded_model = mlflow.pytorch.load_model(self.model_uri, map_location=self.device, weights_only=False)
        except TypeError:
            loaded_model = mlflow.pytorch.load_model(self.model_uri, map_location=self.device)
        model = self._rebuild_runtime_model(loaded_model)
        model.to(self.device)
        model.eval()
        return model

    def _register_pickle_aliases(self) -> None:
        main_module = sys.modules.get("__main__")
        if main_module is not None:
            setattr(main_module, "MuraDinoClassifier", MuraDinoClassifier)

    def _rebuild_runtime_model(self, loaded_model):
        # The MLflow artifact was pickled from a notebook. Rebuilding from the
        # state_dict avoids runtime crashes from stale pickled module state.
        dino_model = loaded_model.dino.__class__(loaded_model.dino.config)
        rebuilt_model = MuraDinoClassifier(
            getattr(loaded_model, "model_name", ""),
            list(loaded_model.bone_categories),
            dino_model=dino_model,
        )
        rebuilt_model.load_state_dict(loaded_model.state_dict())
        return rebuilt_model

    def _resolve_threshold(self) -> float:
        if self.config.threshold_override is not None:
            return self.config.threshold_override

        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.config.tracking_uri)
            if self.config.model_version:
                model_version = client.get_model_version(self.config.model_name, self.config.model_version)
            else:
                model_version = client.get_model_version_by_alias(self.config.model_name, self.config.model_alias)
            if not model_version.run_id:
                return self.config.default_threshold
            run = client.get_run(model_version.run_id)
            return float(run.data.metrics.get("final/best_threshold", self.config.default_threshold))
        except Exception as exc:
            logger.warning("Could not resolve MLflow threshold; using default %.3f: %s", self.config.default_threshold, exc)
            return self.config.default_threshold

    def known_anatomies(self) -> set[str]:
        if hasattr(self.model, "bone_categories"):
            return set(self.model.bone_categories)
        if hasattr(self.model, "classifiers"):
            return set(self.model.classifiers.keys())
        return set(MURA_ANATOMIES)

    def predict_images(
        self,
        image_payloads: Sequence[bytes],
        anatomies: Sequence[str],
        filenames: Sequence[str | None] | None = None,
    ) -> list[DinoImagePrediction]:
        if len(image_payloads) != len(anatomies):
            raise ValueError("Number of images and anatomy values must match")

        filenames = list(filenames or [None] * len(image_payloads))
        normalized_anatomies = [self._normalize_anatomy(value) for value in anatomies]
        self._validate_anatomies(normalized_anatomies)

        original_images: list[Image.Image] = []
        processed_images: list[Image.Image] = []
        tensors = []
        for payload in image_payloads:
            image = self._decode_image(payload)
            processed_image = self.pad_resize(image)
            tensor = self.normalize(self.to_tensor(processed_image))
            original_images.append(image)
            processed_images.append(processed_image)
            tensors.append(tensor)

        probabilities: list[float] = []
        for start in range(0, len(tensors), self.config.batch_size):
            batch_tensors = tensors[start : start + self.config.batch_size]
            batch_anatomies = normalized_anatomies[start : start + self.config.batch_size]
            batch = self.torch.stack(batch_tensors).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(batch, batch_anatomies)
                probs = self.torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
            probabilities.extend(float(value) for value in probs)

        return [
            DinoImagePrediction(
                filename=filename,
                anatomy=anatomy,
                probability=probability,
                prediction=int(probability >= self.threshold),
                threshold=self.threshold,
                original_size=original_image.size,
                original_image=original_image,
                processed_image=processed_image,
            )
            for filename, anatomy, probability, original_image, processed_image in zip(
                filenames,
                normalized_anatomies,
                probabilities,
                original_images,
                processed_images,
                strict=True,
            )
        ]

    def _decode_image(self, payload: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(payload))
            image.load()
        except Exception as exc:
            raise ValueError("Cannot decode image") from exc
        return image.convert("L")

    def _normalize_anatomy(self, anatomy: str) -> str:
        return anatomy.strip().upper()

    def _validate_anatomies(self, anatomies: Sequence[str]) -> None:
        allowed = self.known_anatomies()
        unknown = sorted(set(anatomies) - allowed)
        if unknown:
            raise ValueError(f"Unknown anatomy values for this model: {unknown}. Known values: {sorted(allowed)}")
