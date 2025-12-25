from typing import Any, List, Optional, Union
from .base_schema import BaseSchema


class DebugPayload(BaseSchema):
    hog: List[float]
    processed_image: str
    hog_image: Optional[str] = None


class PredictionResponse(BaseSchema):
    study_id: str
    filenames: List[str]
    prediction: Optional[Union[str, int, float]] = None
    confidence: Optional[float] = None
    debug: Optional[DebugPayload] = None

class ForwardImageResponse(BaseSchema):
    filename: Optional[str] = None
    prediction: Optional[Union[str, int, float]] = None
    confidence: Optional[float] = None
    image_base64: str
    debug: Optional[DebugPayload] = None
