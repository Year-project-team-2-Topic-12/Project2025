from ..deps import get_hog_predictor
from ml.hog_predictor import HogPredictor
from ml.hog import compute_files_hog
from fastapi import Depends, UploadFile
from ml.preprocessing import resize_with_padding_cv2, enhance_brightness_cv2
import cv2
import numpy as np



class InferenceService:
    def __init__(self, as_gray=True, predictor: HogPredictor=Depends(get_hog_predictor)):
        self.as_gray = as_gray
        self.predictor = predictor

    def predict(self, input_data: UploadFile):

        image = self._read_upload_cv2_gray(input_data)
        processed_data = self._preprocess(image)
        predictions = self.predictor.predict(processed_data)
        results = self._postprocess(predictions)
        
        return results
    
    def _read_upload_cv2_gray(self, upload: UploadFile) -> np.ndarray:
        data = upload.file.read()           # bytes
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Cannot decode image")

        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = enhance_brightness_cv2(image)
        image = resize_with_padding_cv2(image)
        print("Preprocessed image shape:", image.shape)
        hog_vector = compute_files_hog([image])
        print("HOG vector shape:", hog_vector.shape)
        return hog_vector

    def _postprocess(self, predictions):
        return str(predictions)