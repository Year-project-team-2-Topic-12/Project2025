import base64
from typing import Any

from ..deps import get_hog_predictor, get_hog_predictor_multiple
from ..schemas.inference_schema import ForwardImageResponse, DebugPayload, PredictionResponse
from ml.hog_predictor import HogPredictor
from ml.preprocessing import resize_with_padding_cv2, enhance_brightness_cv2
from ml.hog import compute_hog_with_visualization
from fastapi import UploadFile
import cv2
import numpy as np
from .request_logging_service import RequestLoggingService
import numpy as np
import time

def measure_inference_time_decorator(func):
    def wrapper(self: 'InferenceService', *args, **kwargs):
            start_time = time.time()
            status = 'Успех!'
            result = None
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                status = f'Ошибка {e}!'
                raise e
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                result_value = None
                if result is not None:
                    if hasattr(result, "prediction"):
                        result_value = str(getattr(result, "prediction"))
                    else:
                        result_value = "Нет предсказания"
                self.request_logging_service.add_inference_request(
                    input_meta=f"Вызов {func.__name__}",
                    image_width=self.stat_image_width,
                    image_height=self.stat_image_height,
                    duration=duration_ms,
                    result=result_value,
                    status=status,
                )

            return result
    return wrapper

class InferenceService:
    def __init__(self, predictor_multiple: HogPredictor, predictor_single: HogPredictor, as_gray=True, request_logging_service: RequestLoggingService = None):
        self.as_gray = as_gray
        self.predictor_multiple = predictor_multiple
        self.predictor_single = predictor_single
        self.request_logging_service = request_logging_service

        # для сервисов истории и статистики запросов
        self.stat_image_height = None
        self.stat_image_width = None

    @measure_inference_time_decorator
    def predict_single(self, upload: UploadFile, debug: bool = False) -> ForwardImageResponse:
        image = self._read_upload_cv2_gray(upload)
        hog_vector, processed_image, hog_image = self._preprocess_study([image], return_image=True)
        prediction_value, confidence = self.predictor_single.predict_with_confidence(hog_vector, is_multiple=False)

        if processed_image is None:
            raise ValueError("Processed image not available")

        response: ForwardImageResponse = ForwardImageResponse(
            filename=upload.filename,
            prediction=prediction_value.item(),
            confidence=confidence.item(),
            image_base64=self._encode_image_base64(image),
        )

        self.stat_image_height, self.stat_image_width = image.shape[:2]
        print(f"Image shape for stats: {self.stat_image_height}x{self.stat_image_width}")

        if debug:
            response.debug = DebugPayload(
                hog=hog_vector.tolist(),
                processed_image=self._encode_image_base64(processed_image),
                hog_image=self._encode_hog_image_base64(hog_image) if hog_image is not None else None,
            )

        return response

    @measure_inference_time_decorator
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
        hog_vectors_studies: list[np.ndarray] = []
        processed_images_studies: list[np.ndarray | None] = []
        hog_images_studies: list[np.ndarray | None] = []
        results: list[PredictionResponse] = []
        image_sizes: list[tuple[int, int]] = []
        for study_id in study_order:
            uploads = studies[study_id]
            images: list[np.ndarray] = []
            filenames: list[str] = []
            for upload in uploads:
                images.append(self._read_upload_cv2_gray(upload))
                image_sizes.append(images[-1].shape[:2])
                if upload.filename:
                    filenames.append(upload.filename)

            hog_vector, processed_image, hog_image = self._preprocess_study(images, return_image=debug)
            hog_vectors_studies.append(hog_vector)
            processed_images_studies.append(processed_image)
            hog_images_studies.append(hog_image)
        avg_height, avg_width = int(np.mean([size[0] for size in image_sizes])), int(np.mean([size[1] for size in image_sizes]))
        print(f"Average processed image size across studies: {avg_height}x{avg_width}")
        self.stat_image_height = avg_height
        self.stat_image_width = avg_width

        prediction, confidence = self.predictor_multiple.predict_with_confidence(np.array(hog_vectors_studies), is_multiple=True)
        for idx, study_id in enumerate(study_order):
            hog_vector = hog_vectors_studies[idx]
            processed_image = processed_images_studies[idx]
            print("len processed images studies", np.array(processed_images_studies).shape)
            print("hog vector shape", np.array(hog_vectors_studies).shape)
            hog_image = hog_images_studies[idx]
            filenames = [upload.filename for upload in studies[study_id] if upload.filename]
            prediction_value = prediction[idx]
            confidence_value = confidence[idx]
            results.append(
                self._postprocess(
                    prediction_value,
                    confidence=confidence_value,
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
        data = upload.file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Cannot decode image")

        return img

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
            print("Processed image shape:", resized.shape)

        hog_vector, hog_image = compute_hog_with_visualization(np.array(processed_images))
        if return_image:
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

    def _normalize_prediction(self, predictions: np.ndarray) -> Any:
        return predictions.item() if predictions.size == 1 else predictions.tolist()

    def _postprocess(
        self,
        predictions: np.ndarray,
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
