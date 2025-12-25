from datetime import datetime

from backend.schemas.base_schema import BaseSchema


class RequestLogEntry(BaseSchema):
    id: int
    timestamp: datetime
    input_meta: str | None = None
    duration: float | None = None
    result: str | None = None
    status: int | None = None


class DeleteHistoryResponse(BaseSchema):
    message: str
