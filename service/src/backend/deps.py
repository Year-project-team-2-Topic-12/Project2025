from __future__ import annotations
from functools import lru_cache
from fastapi import Depends
from sqlalchemy.orm import Session

from backend.db.connection import get_session
from backend.db.repositories.user_repository import UserRepository
from backend.db.repositories.request_log_repository import RequestLogRepository
from backend.services.auth_service import AuthService
from backend.services.request_logging_service import RequestLoggingService
from backend.services.stats_service import StatsService


@lru_cache(maxsize=1)
def get_dino_predictor() -> 'DinoMlflowPredictor':
    from ml.dino_predictor import DinoMlflowPredictor
    return DinoMlflowPredictor()

def get_user_repository(session: Session = Depends(get_session)) -> UserRepository:
    return UserRepository(session)

def get_request_log_repository(
    session: Session = Depends(get_session),
) -> RequestLogRepository:
    return RequestLogRepository(session)

def get_auth_service(
    repo: UserRepository = Depends(get_user_repository),
) -> AuthService:
    return AuthService(repo)

def get_request_logging_service(
    repo: RequestLogRepository = Depends(get_request_log_repository),
) -> RequestLoggingService:
    return RequestLoggingService(repo)


def get_stats_service(
    repo: RequestLogRepository = Depends(get_request_log_repository),
) -> StatsService:
    return StatsService(repo)

def get_inference_service(
        dino_predictor = Depends(get_dino_predictor),
        request_logging_service = Depends(get_request_logging_service)
    ) -> 'InferenceService':
    from .services.inference_service import InferenceService
    return InferenceService(predictor=dino_predictor, request_logging_service=request_logging_service)
