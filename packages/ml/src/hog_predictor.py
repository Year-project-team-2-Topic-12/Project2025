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

    def predict(self, image: np.ndarray):
        print(f"Predicting using the model on the provided image")
        return self.model.predict([image])

    def predict_with_confidence(self, image: np.ndarray) -> tuple[object, float | None]:
        prediction = self.predict(image)
        prediction_value = prediction.item() if isinstance(prediction, np.ndarray) and prediction.size == 1 else prediction
        confidence = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([image])
            if isinstance(proba, np.ndarray) and proba.size > 0:
                confidence = float(np.max(proba))

        return prediction_value, confidence
