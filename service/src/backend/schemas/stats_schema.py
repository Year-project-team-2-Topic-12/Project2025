from backend.schemas.base_schema import BaseSchema


class ProcessingTimeStats(BaseSchema):
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class ImageStats(BaseSchema):
    mean_width: float
    mean_height: float
    count: int


class StatsResponse(BaseSchema):
    processing_time: ProcessingTimeStats
    image_stats: ImageStats
