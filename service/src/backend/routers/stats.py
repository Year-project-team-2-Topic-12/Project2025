from fastapi import APIRouter, Depends

from backend.deps import get_stats_service
from backend.schemas.stats_schema import StatsResponse
from backend.services.stats_service import StatsService

router = APIRouter()

@router.get("/stats", response_model=StatsResponse)
def get_stats(
    stats_service: StatsService = Depends(get_stats_service),
):
    return stats_service.get_stats()
