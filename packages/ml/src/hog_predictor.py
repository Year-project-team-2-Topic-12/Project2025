from .data import load_model_pipeline
from .env import SELECTED_MODEL
from sklearn.pipeline import Pipeline
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HogPredictor:
    def __init__(self, model_name=SELECTED_MODEL):
        self.model_name = model_name
        logger.info("Loading model pipeline: %s", self.model_name)
        self.model: Pipeline = load_model_pipeline(self.model_name)
        logger.info("Model loaded successfully.")

    def predict(self, images: np.ndarray) -> np.ndarray:
        logger.debug("Predicting using the model on the provided image")
        return self.model.predict(images)

    def predict_with_confidence(self, image: np.ndarray, is_multiple=False) -> tuple[np.ndarray, np.ndarray]:
        images = np.array([image]) if not is_multiple else image
        prediction = self.predict(images)
        confidence = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(images)
            if isinstance(proba, np.ndarray) and proba.size > 0:
                logger.debug("prediction datatype %s", prediction.dtype)
                confidence = proba[np.arange(len(prediction)), prediction]
                logger.debug("confidence %s", confidence)
                return prediction, confidence
        else:
            raise NotImplementedError("Model does not support probability estimates.")

