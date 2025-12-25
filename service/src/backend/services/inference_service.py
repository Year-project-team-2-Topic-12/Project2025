import base64
from typing import Any

from ..deps import get_hog_predictor
from ..schemas.inference_schema import ForwardImageResponse, DebugPayload, PredictionResponse
from ml.hog_predictor import HogPredictor
from ml.preprocessing import resize_with_padding_cv2, enhance_brightness_cv2
from ml.hog import compute_images_hog, compute_hog_with_visualization
from fastapi import Depends, UploadFile
import cv2
import numpy as np



class InferenceService:
    def __init__(self, as_gray=True, predictor: HogPredictor=Depends(get_hog_predictor)):
        self.as_gray = as_gray
        self.predictor = predictor

    def predict_single(self, upload: UploadFile, debug: bool = False) -> ForwardImageResponse:
        image = self._read_upload_cv2_gray(upload)
        hog_vector, processed_image, hog_image = self._preprocess_study([image], return_image=True)
        prediction_value, confidence = self._predict_with_confidence(hog_vector)

        if processed_image is None:
            raise ValueError("Processed image not available")

        response: ForwardImageResponse = ForwardImageResponse(
            filename=upload.filename,
            prediction=prediction_value,
            confidence=confidence,
            image_base64=self._encode_image_base64(processed_image),
        )

        if debug:
            response.debug = DebugPayload(
                hog=hog_vector.tolist(),
                processed_image=self._encode_image_base64(processed_image),
                hog_image=self._encode_hog_image_base64(hog_image) if hog_image is not None else None,
            )

        return response

    def predict_studies(
        self,
        input_data: list[UploadFile],
        *,
        study_ids: list[str],
        debug: bool = False,
    ) -> list[PredictionResponse]:
        studies: dict[str, list[UploadFile]] = {}
        study_order: list[str] = []
        for upload, study_id in zip(input_data, study_ids, strict=True):
            if study_id not in studies:
                studies[study_id] = []
                study_order.append(study_id)
            studies[study_id].append(upload)

        results: list[PredictionResponse] = []
        for study_id in study_order:
            uploads = studies[study_id]
            images: list[np.ndarray] = []
            filenames: list[str] = []
            for upload in uploads:
                images.append(self._read_upload_cv2_gray(upload))
                if upload.filename:
                    filenames.append(upload.filename)

            hog_vector, processed_image, hog_image = self._preprocess_study(images, return_image=debug)
            prediction_value, confidence = self._predict_with_confidence(hog_vector)
            results.append(
                self._postprocess(
                    prediction_value,
                    confidence=confidence,
                    study_id=study_id,
                    filenames=filenames,
                    debug=debug,
                    hog_vector=hog_vector,
                    processed_image=processed_image,
                    hog_image=hog_image,
                )
            )

        return results
    
    def _read_upload_cv2_gray(self, upload: UploadFile) -> np.ndarray:
        data = upload.file.read()           # bytes
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Cannot decode image")

        return img

    def _preprocess(
        self,
        image: np.ndarray,
        *,
        return_image: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        enhanced = enhance_brightness_cv2(image)
        resized = resize_with_padding_cv2(enhanced)
        if return_image:
            hog_vector, hog_image = compute_hog_with_visualization(resized)
        else:
            hog_vector = compute_images_hog([resized])
            hog_image = None
        if return_image:
            return hog_vector, resized, hog_image
        return hog_vector, None, None

    def _preprocess_study(
        self,
        images: list[np.ndarray],
        *,
        return_image: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        processed_images: list[np.ndarray] = []
        for image in images:
            enhanced = enhance_brightness_cv2(image)
            resized = resize_with_padding_cv2(enhanced)
            processed_images.append(resized)

        hog_vector = compute_images_hog(processed_images)
        if return_image:
            hog_vector_first, hog_image = compute_hog_with_visualization(processed_images, is_multiple=True)
            return hog_vector, processed_images[0], hog_image
        return hog_vector, None, None

    def _encode_image_base64(self, image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Cannot encode image to PNG")
        b64 = base64.b64encode(buffer).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _encode_hog_image_base64(self, hog_image: np.ndarray) -> str:
        hog_u8 = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
        return self._encode_image_base64(hog_u8)

    def _predict_with_confidence(self, hog_vector: np.ndarray) -> tuple[Any, float | None]:
        predictions = self.predictor.predict(hog_vector)
        prediction_value = self._normalize_prediction(predictions)
        print("Результаты предсказания:")
        print(predictions)
        print(type(predictions))
        confidence = None
        if hasattr(self.predictor.model, "predict_proba"):
            proba = self.predictor.model.predict_proba([hog_vector])
            print("Вероятности предсказания:")
            print(proba)
        return prediction_value, 0

    def _normalize_prediction(self, predictions: Any) -> Any:
        if isinstance(predictions, np.ndarray):
            return predictions.item() if predictions.size == 1 else predictions.tolist()
        if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
            return predictions[0]
        return predictions

    def _postprocess(
        self,
        predictions: Any,
        *,
        confidence: float | None,
        study_id: str,
        filenames: list[str],
        debug: bool,
        hog_vector: np.ndarray,
        processed_image: np.ndarray | None,
        hog_image: np.ndarray | None,
    ) -> PredictionResponse:
        prediction_value = self._normalize_prediction(predictions)

        response: PredictionResponse = PredictionResponse(
            study_id=study_id,
            filenames=filenames,
            prediction=prediction_value,
            confidence=confidence,
        )

        if debug:
            if processed_image is None:
                raise ValueError("Debug image requested but not available")
            response.debug = DebugPayload(
                hog=hog_vector.tolist(),
                processed_image=self._encode_image_base64(processed_image),
                hog_image=self._encode_hog_image_base64(hog_image) if hog_image is not None else None,
            )
        return response
