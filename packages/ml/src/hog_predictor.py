from .data import load_model_pipeline
from .env import SELECTED_MODEL
from sklearn.pipeline import Pipeline
import numpy as np

class HogPredictor:
    def __init__(self, model_name=SELECTED_MODEL):
        self.model_name = model_name
        print(f"Loading model pipeline: {self.model_name}")
        self.model: Pipeline = load_model_pipeline(self.model_name)
        print("Model loaded successfully.", self.model)

    def predict(self, images: np.ndarray) -> np.ndarray:
        print(f"Predicting using the model on the provided image")
        return self.model.predict(images)

    def predict_with_confidence(self, image: np.ndarray, is_multiple=False) -> tuple[np.ndarray, float | np.ndarray | None]:
        images = np.array([image]) if not is_multiple else image
        prediction = self.predict(images)
        confidence = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(images)
            if isinstance(proba, np.ndarray) and proba.size > 0:
                confidence = proba[np.arange(len(images)), images]

                confidence = float(np.max(proba))

        return prediction, confidence
