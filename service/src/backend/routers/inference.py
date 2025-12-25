import traceback

from fastapi import APIRouter, HTTPException, File, Header, Request
from fastapi import UploadFile, Depends
from typing import Callable

from ..services.inference_service import InferenceService
from ..deps import get_inference_service
from ..schemas.inference_schema import ForwardImageResponse, PredictionResponse

router = APIRouter()

def validate_image_upload(upload: UploadFile) -> None:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="bad request")


def validate_content_type(request: Request, expected_prefix: str = "multipart/form-data") -> None:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith(expected_prefix):
        raise HTTPException(status_code=400, detail="bad request")

def print_error_fulltraceback(e: Exception) -> None:

    traceback.print_exc()

def run_prediction(executable: Callable):
    try:
        return executable()
    except Exception as e:
        print(f"Ошибка при выполнении предсказания: {e}")
        print_error_fulltraceback(e)
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")

@router.post("/forward", response_model=ForwardImageResponse)
async def forward(
    request: Request,
    image: UploadFile = File(..., alias="image"), 
    debug: bool = Header(False, alias="X-Debug"),
    service: InferenceService = Depends(get_inference_service),
):
    validate_content_type(request)
    validate_image_upload(image)
    return run_prediction(lambda: service.predict_single(image, debug=debug))

@router.post("/forwardMultiple", response_model=list[PredictionResponse])
def forward_multiple(
    request: Request,
    images: list[UploadFile] = File(..., alias="images"), 
    study_ids: str = Header(..., alias="X-Study-Ids"),
    debug: bool = Header(False, alias="X-Debug"),
    service: InferenceService = Depends(get_inference_service),
):
    validate_content_type(request)

    parsed_study_ids = [item.strip() for item in study_ids.split(",") if item.strip()]

    if len(images) != len(parsed_study_ids):
        raise HTTPException(status_code=400, detail="bad request")
    for file in images:
        validate_image_upload(file)
    return run_prediction(lambda: service.predict_studies(images, study_ids=parsed_study_ids, debug=debug))
