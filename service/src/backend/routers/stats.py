from fastapi import APIRouter, Depends
import numpy as np

from backend.deps import get_request_logging_service
from backend.schemas.stats_schema import StatsResponse
from backend.services.request_logging_service import RequestLoggingService

router = APIRouter()

@router.get("/stats", response_model=StatsResponse)
def get_stats(
    logging_service: RequestLoggingService = Depends(get_request_logging_service),
):
    logs = logging_service.get_successful_logs_for_stats()
    
    if not logs:
        return {
            "processing_time": {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0},
            "image_stats": {"mean_width": 0, "mean_height": 0, "count": 0}
        }

    durations = [log.duration for log in logs if log.duration is not None]
    if durations:
        mean = float(np.mean(durations))
        p50 = float(np.percentile(durations, 50))
        p95 = float(np.percentile(durations, 95))
        p99 = float(np.percentile(durations, 99))
    else:
        mean = p50 = p95 = p99 = 0

    widths = [log.image_width for log in logs if log.image_width is not None]
    heights = [log.image_height for log in logs if log.image_height is not None]
    
    if widths and heights:
        mean_width = float(np.mean(widths))
        mean_height = float(np.mean(heights))
        count = len(widths)
    else:
        mean_width = mean_height = 0
        count = 0

    return {
        "processing_time": {
            "mean_ms": mean,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99
        },
        "image_stats": {
            "mean_width": mean_width,
            "mean_height": mean_height,
            "count": count
        }
    }
