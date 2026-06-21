import traceback

from fastapi import APIRouter, HTTPException, File, Header, Request
from fastapi import UploadFile, Depends
from typing import Callable
import logging

from ..services.inference_service import InferenceService
from ..deps import get_inference_service
from ..schemas.inference_schema import ForwardImageResponse, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)
MURA_ANATOMIES = {
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST",
}

def validate_image_upload(upload: UploadFile) -> None:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="bad request")


def validate_content_type(request: Request, expected_prefix: str = "multipart/form-data") -> None:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith(expected_prefix):
        raise HTTPException(status_code=400, detail="bad request")

def parse_anatomy_header(anatomy: str | None) -> str:
    if not anatomy or not anatomy.strip():
        raise HTTPException(status_code=400, detail="X-Anatomy header is required")
    normalized = anatomy.strip().upper()
    if normalized not in MURA_ANATOMIES:
        raise HTTPException(status_code=400, detail=f"Unknown anatomy: {normalized}")
    return normalized

def parse_anatomies_header(
    images_count: int,
    anatomies: str | None,
    anatomy: str | None,
) -> list[str]:
    if anatomies and anatomies.strip():
        parsed = [item.strip().upper() for item in anatomies.split(",") if item.strip()]
    elif anatomy and anatomy.strip():
        parsed = [anatomy.strip().upper()] * images_count
    else:
        raise HTTPException(status_code=400, detail="X-Anatomies or X-Anatomy header is required")

    if len(parsed) != images_count:
        raise HTTPException(status_code=400, detail="bad request")

    unknown = sorted(set(parsed) - MURA_ANATOMIES)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown anatomy: {', '.join(unknown)}")
    return parsed

def print_error_fulltraceback(e: Exception) -> None:

    traceback.print_exc()

def run_prediction(executable: Callable):
    try:
        return executable()
    except Exception as e:
        logger.exception("Ошибка при выполнении предсказания: %s", e)
        print_error_fulltraceback(e)
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")

@router.post("/forward", response_model=ForwardImageResponse)
def forward(
    request: Request,
    image: UploadFile = File(..., alias="image"), 
    anatomy: str | None = Header(None, alias="X-Anatomy"),
    debug: bool = Header(False, alias="X-Debug"),
    service: InferenceService = Depends(get_inference_service),
):
    validate_content_type(request)
    validate_image_upload(image)
    parsed_anatomy = parse_anatomy_header(anatomy)
    return run_prediction(lambda: service.predict_single(image, anatomy=parsed_anatomy, debug=debug))

@router.post("/forwardMultiple", response_model=list[PredictionResponse])
def forward_multiple(
    request: Request,
    images: list[UploadFile] = File(..., alias="images"), 
    study_ids: str = Header(..., alias="X-Study-Ids"),
    anatomies: str | None = Header(None, alias="X-Anatomies"),
    anatomy: str | None = Header(None, alias="X-Anatomy"),
    debug: bool = Header(False, alias="X-Debug"),
    service: InferenceService = Depends(get_inference_service),
):
    validate_content_type(request)

    parsed_study_ids = [item.strip() for item in study_ids.split(",") if item.strip()]

    if len(images) != len(parsed_study_ids):
        raise HTTPException(status_code=400, detail="bad request")
    for file in images:
        validate_image_upload(file)
    parsed_anatomies = parse_anatomies_header(len(images), anatomies, anatomy)
    return run_prediction(lambda: service.predict_studies(images, study_ids=parsed_study_ids, anatomies=parsed_anatomies, debug=debug))
