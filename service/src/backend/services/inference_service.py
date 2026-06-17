import base64
import logging
import time
from collections import OrderedDict
from functools import wraps
from io import BytesIO

import numpy as np
from fastapi import UploadFile
from PIL import Image

from ml.dino_predictor import DinoImagePrediction, DinoMlflowPredictor

from ..schemas.inference_schema import DebugImagePrediction, DebugPayload, ForwardImageResponse, PredictionResponse
from .request_logging_service import RequestLoggingService

logger = logging.getLogger(__name__)


def measure_inference_time_decorator(func):
    @wraps(func)
    def wrapper(self: "InferenceService", *args, **kwargs):
        start_time = time.time()
        status = "Успех!"
        result = None
        try:
            result = func(self, *args, **kwargs)
        except Exception as exc:
            status = f"Ошибка {exc}!"
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            result_value = self._prediction_log_value(result)
            input_meta = self._prediction_log_meta(func.__name__, result)
            if self.request_logging_service is not None:
                self.request_logging_service.add_inference_request(
                    input_meta=input_meta,
                    image_width=self.stat_image_width,
                    image_height=self.stat_image_height,
                    duration=duration_ms,
                    result=result_value,
                    status=status,
                )
        return result

    return wrapper


class InferenceService:
    def __init__(
        self,
        predictor: DinoMlflowPredictor,
        request_logging_service: RequestLoggingService | None = None,
    ):
        self.predictor = predictor
        self.request_logging_service = request_logging_service

        # для сервисов истории и статистики запросов
        self.stat_image_height = None
        self.stat_image_width = None

    @measure_inference_time_decorator
    def predict_single(self, upload: UploadFile, *, anatomy: str, debug: bool = False) -> ForwardImageResponse:
        payload = self._read_upload_bytes(upload)
        image_prediction = self.predictor.predict_images([payload], [anatomy], [upload.filename])[0]

        width, height = image_prediction.original_size
        self.stat_image_height = height
        self.stat_image_width = width
        logger.debug("Image shape for stats: %sx%s", self.stat_image_height, self.stat_image_width)

        response = ForwardImageResponse(
            filename=upload.filename,
            anatomy=image_prediction.anatomy,
            prediction=image_prediction.prediction,
            probability=image_prediction.probability,
            confidence=image_prediction.confidence,
            threshold=image_prediction.threshold,
            image_base64=self._encode_pil_image_base64(image_prediction.original_image),
        )

        if debug:
            response.debug = self._build_debug_payload([image_prediction])

        return response

    @measure_inference_time_decorator
    def predict_studies(
        self,
        input_data: list[UploadFile],
        *,
        study_ids: list[str],
        anatomies: list[str],
        debug: bool = False,
    ) -> list[PredictionResponse]:
        if len(input_data) != len(study_ids) or len(input_data) != len(anatomies):
            raise ValueError("Images, study IDs and anatomies counts must match")

        payloads = [self._read_upload_bytes(upload) for upload in input_data]
        filenames = [upload.filename for upload in input_data]
        image_predictions = self.predictor.predict_images(payloads, anatomies, filenames)
        self._set_average_stat_size(image_predictions)

        studies: OrderedDict[str, list[DinoImagePrediction]] = OrderedDict()
        for study_id, image_prediction in zip(study_ids, image_predictions, strict=True):
            studies.setdefault(study_id, []).append(image_prediction)

        results: list[PredictionResponse] = []
        for study_id, predictions in studies.items():
            study_anatomies = {prediction.anatomy for prediction in predictions}
            if len(study_anatomies) != 1:
                raise ValueError(f"Study {study_id} contains multiple anatomy values: {sorted(study_anatomies)}")

            probability = float(np.mean([prediction.probability for prediction in predictions]))
            threshold = predictions[0].threshold
            prediction_value = int(probability >= threshold)
            confidence = probability if prediction_value == 1 else 1.0 - probability
            response = PredictionResponse(
                study_id=study_id,
                anatomy=predictions[0].anatomy,
                filenames=[prediction.filename for prediction in predictions if prediction.filename],
                n_images=len(predictions),
                prediction=prediction_value,
                probability=probability,
                confidence=confidence,
                threshold=threshold,
            )

            if debug:
                response.debug = self._build_debug_payload(predictions)

            results.append(response)

        return results

    def _read_upload_bytes(self, upload: UploadFile) -> bytes:
        data = upload.file.read()
        if not data:
            raise ValueError("Uploaded image is empty")
        return data

    def _set_average_stat_size(self, predictions: list[DinoImagePrediction]) -> None:
        if not predictions:
            self.stat_image_height = None
            self.stat_image_width = None
            return
        heights = [prediction.original_size[1] for prediction in predictions]
        widths = [prediction.original_size[0] for prediction in predictions]
        self.stat_image_height = int(np.mean(heights))
        self.stat_image_width = int(np.mean(widths))
        logger.debug("Average uploaded image size: %sx%s", self.stat_image_height, self.stat_image_width)

    def _build_debug_payload(self, predictions: list[DinoImagePrediction]) -> DebugPayload:
        first_prediction = predictions[0]
        return DebugPayload(
            processed_image=self._encode_pil_image_base64(first_prediction.processed_image),
            image_predictions=[
                DebugImagePrediction(
                    filename=prediction.filename,
                    anatomy=prediction.anatomy,
                    probability=prediction.probability,
                    prediction=prediction.prediction,
                    confidence=prediction.confidence,
                    threshold=prediction.threshold,
                    processed_image=self._encode_pil_image_base64(prediction.processed_image),
                )
                for prediction in predictions
            ],
        )

    def _encode_pil_image_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _prediction_log_value(self, result) -> str | None:
        if result is None:
            return None
        if isinstance(result, list):
            return "; ".join(self._format_prediction_log_item(item) for item in result) or "Нет предсказания"
        prediction = self._get_log_field(result, "prediction")
        if prediction is not None:
            probability = self._get_log_field(result, "probability")
            if probability is not None:
                return f"{prediction} (p={float(probability):.4f})"
            return str(prediction)
        return "Нет предсказания"

    def _prediction_log_meta(self, func_name: str, result) -> str:
        if isinstance(result, list):
            studies_count = len(result)
            images_count = sum(int(self._get_log_field(item, "n_images") or 0) for item in result)
            return f"Вызов {func_name}: studies={studies_count}, images={images_count}"
        return f"Вызов {func_name}"

    def _format_prediction_log_item(self, item) -> str:
        study_id = self._get_log_field(item, "study_id")
        prediction = self._get_log_field(item, "prediction")
        probability = self._get_log_field(item, "probability")
        anatomy = self._get_log_field(item, "anatomy")
        n_images = self._get_log_field(item, "n_images")

        parts = []
        if study_id is not None:
            parts.append(f"{study_id}")
        if prediction is not None:
            parts.append(f"pred={prediction}")
        if probability is not None:
            parts.append(f"p={float(probability):.4f}")
        if anatomy is not None:
            parts.append(f"anatomy={anatomy}")
        if n_images is not None:
            parts.append(f"images={n_images}")

        return " ".join(parts) if parts else "Нет предсказания"

    def _get_log_field(self, item, field: str):
        if isinstance(item, dict):
            return item.get(field)
        return getattr(item, field, None)
