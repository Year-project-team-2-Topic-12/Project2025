from fastapi import APIRouter, HTTPException, Depends
from backend.database import engine
from backend import crud
from sqlalchemy.orm import Session as OrmSession
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()
router = APIRouter()

@router.get("/history")
def read_history():
    with OrmSession(engine) as session:
        logs = crud.get_all_logs(session)
    return [
        {
            "id": log.id,
            "timestamp": log.timestamp,
            "input_meta" : log.input_meta,
            "duration": log.duration,
            "result": log.result,
            "status": log.status,
        }
        for log in logs
    ]

@router.delete("/history")
def delete_history(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "secret_token":
        raise HTTPException(status_code=403, detail="Unauthorized")
    with OrmSession(engine) as session:
        crud.delete_all_logs(session)
    return {"message": "История удалена"}