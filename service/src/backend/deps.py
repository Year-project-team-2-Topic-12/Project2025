from __future__ import annotations
from functools import lru_cache


@lru_cache(maxsize=1)
def get_hog_predictor() -> HogPredictor:
    from ml.hog_predictor import HogPredictor
    return HogPredictor()

@lru_cache(maxsize=1)
def get_inference_service() -> 'InferenceService':
    from .services.inference_service import InferenceService
    return InferenceService()
