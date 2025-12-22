from fastapi import APIRouter, HTTPException
from backend.database import engine
from fastapi import UploadFile

router = APIRouter()

@router.post("/forward")
def forward(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type - model only accepts images")
    return {"filename": file.filename}
