from fastapi import APIRouter, HTTPException
from backend.database import engine
from fastapi import UploadFile, Depends

from ..services.inference_service import InferenceService

router = APIRouter()

@router.post("/forward")
def forward(file: UploadFile, service: InferenceService = Depends(InferenceService)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type - model only accepts images")
    return service.predict(file)
