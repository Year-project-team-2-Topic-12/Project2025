from fastapi import APIRouter, HTTPException, Depends
from backend.database import engine
from backend import crud
from sqlalchemy.orm import Session as OrmSession
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import UploadFile

security = HTTPBearer()
router = APIRouter()

@router.post("/forward")
def forward(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type - model only accepts images")
    return {"filename": file.filename}
