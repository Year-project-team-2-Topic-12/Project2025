from .data import load_model_pipeline
from .env import SELECTED_MODEL
import numpy as np

class HogPredictor:
    def __init__(self, model_name=SELECTED_MODEL):
        self.model_name = model_name
        print(f"Loading model pipeline: {self.model_name}")
        self.model = load_model_pipeline(self.model_name)
        print("Model loaded successfully.", self.model)

    def predict(self, image: np.ndarray):
        print(f"Predicting using the model on the provided image")
        return self.model.predict([image])
