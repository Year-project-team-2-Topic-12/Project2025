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
def get_hog_predictor() -> 'HogPredictor':
    from ml.hog_predictor import HogPredictor
    return HogPredictor('hog_pca_poly_logreg_pics')

@lru_cache(maxsize=1)
def get_hog_predictor_multiple() -> 'HogPredictor':
    from ml.hog_predictor import HogPredictor
    return HogPredictor()

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
        hog_predictor_multiple = Depends(get_hog_predictor_multiple),
        hog_predictor_single = Depends(get_hog_predictor),
        request_logging_service = Depends(get_request_logging_service)
    ) -> 'InferenceService':
    from .services.inference_service import InferenceService
    return InferenceService(predictor_multiple=hog_predictor_multiple, predictor_single=hog_predictor_single, request_logging_service=request_logging_service)
