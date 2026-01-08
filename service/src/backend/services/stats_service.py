import numpy as np

from backend.db.repositories.request_log_repository import RequestLogRepository
from backend.schemas.stats_schema import StatsResponse, ProcessingTimeStats, ImageStats


class StatsService:
    def __init__(self, repo: RequestLogRepository):
        self.repo = repo

    def get_stats(self) -> StatsResponse:
        logs = self.repo.get_successful_logs()

        if not logs:
            return StatsResponse(
                processing_time=ProcessingTimeStats(mean_ms=0, p50_ms=0, p95_ms=0, p99_ms=0),
                image_stats=ImageStats(mean_width=0, mean_height=0, count=0),
            )

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

        return StatsResponse(
            processing_time=ProcessingTimeStats(
                mean_ms=mean,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
            ),
            image_stats=ImageStats(
                mean_width=mean_width,
                mean_height=mean_height,
                count=count,
            ),
        )
