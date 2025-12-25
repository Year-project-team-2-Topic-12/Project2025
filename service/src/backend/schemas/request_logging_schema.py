from datetime import datetime

from backend.schemas.base_schema import BaseSchema


class RequestLogEntry(BaseSchema):
    timestamp: datetime | None = None
    input_meta: str | None = None
    duration: float | None = None
    result: str | None = None
    status: int | str | None = None
    image_width: int | None = None
    image_height: int | None = None


class DeleteHistoryResponse(BaseSchema):
    message: str
