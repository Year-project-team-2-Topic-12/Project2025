from typing import List, Optional, Union
from .base_schema import BaseSchema


class DebugImagePrediction(BaseSchema):
    filename: Optional[str] = None
    anatomy: str
    probability: float
    prediction: int
    confidence: float
    threshold: float
    processed_image: Optional[str] = None


class DebugPayload(BaseSchema):
    hog: Optional[List[float]] = None
    processed_image: Optional[str] = None
    hog_image: Optional[str] = None
    image_predictions: Optional[List[DebugImagePrediction]] = None


class PredictionResponse(BaseSchema):
    study_id: str
    anatomy: Optional[str] = None
    filenames: List[str]
    n_images: Optional[int] = None
    prediction: Optional[Union[str, int, float]] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    debug: Optional[DebugPayload] = None

class ForwardImageResponse(BaseSchema):
    filename: Optional[str] = None
    anatomy: Optional[str] = None
    prediction: Optional[Union[str, int, float]] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    image_base64: str
    debug: Optional[DebugPayload] = None
