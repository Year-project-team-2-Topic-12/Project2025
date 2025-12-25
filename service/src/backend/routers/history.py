from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.deps import get_auth_service, get_request_logging_service
from backend.schemas.request_logging_schema import DeleteHistoryResponse, RequestLogEntry
from backend.services.auth_service import AuthService
from backend.services.request_logging_service import RequestLoggingService

security = HTTPBearer()
router = APIRouter()


@router.get("/history", response_model=list[RequestLogEntry])
def read_history(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    logging_service: RequestLoggingService = Depends(get_request_logging_service),
):
    _ = credentials
    _ = auth_service
    return logging_service.get_history()


@router.delete("/history", response_model=DeleteHistoryResponse)
def delete_history(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    logging_service: RequestLoggingService = Depends(get_request_logging_service),
):
    auth_service.require_admin(credentials)
    logging_service.delete_history()
    return {"message": "История удалена"}
