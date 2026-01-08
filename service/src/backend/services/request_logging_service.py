from ..db.repositories.request_log_repository import RequestLogRepository
from ..schemas.request_logging_schema import RequestLogEntry
from ..db.models import RequestLog

class RequestLoggingService:
    def __init__(self, repo: RequestLogRepository):
        self.repo = repo

    def get_history(self):
        logs = self.repo.get_all_logs()
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp,
                "input_meta": log.input_meta,
                "duration": log.duration,
                "result": log.result,
                "status": log.status,
            }
            for log in logs
        ]

    def delete_history(self) -> None:
        self.repo.delete_all_logs()

    def add_log_entry(self, log_data: RequestLogEntry) -> RequestLog:
        log_data_dict = log_data.model_dump(exclude_unset=True)
        request_status = log_data_dict.get("status")
        if request_status is not None:
            log_data_dict["status"] = str(request_status)
        return self.repo.add_log(log_data_dict)

    def add_inference_request(
        self,
        input_meta: str = "Предсказание модели",
        image_width: int | None = None,
        image_height: int | None = None,
        duration: float | None = None,
        result: str | None = None,
        status: int | str | None = None,
    ) -> RequestLog:
        log_data = RequestLogEntry(
            input_meta=input_meta,
            image_width=image_width,
            image_height=image_height,
            duration=duration,
            result=result,
            status=status
        )
        return self.add_log_entry(log_data)

    def get_successful_logs_for_stats(self):
        return self.repo.get_successful_logs()

    def seed_test_log(self) -> None:
        self.repo.add_test_log()
