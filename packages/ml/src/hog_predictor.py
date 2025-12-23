from .data import load_model_pipeline
from .env import SELECTED_MODEL
import numpy as np

class HogPredictor:
    def __init__(self, model_name=SELECTED_MODEL):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        # Placeholder for model loading logic
        print(f"Loading model from {self.model_name}")
        return load_model_pipeline(self.model_name)

    def predict(self, image: np.ndarray):
        print(f"Predicting using the model on the provided image")
        return self.model.predict([image])
