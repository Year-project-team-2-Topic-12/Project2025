from backend.db.repositories.request_log_repository import RequestLogRepository


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

    def add_log_entry(self, log_data: dict):
        return self.repo.add_log(log_data)

    def get_successful_logs_for_stats(self):
        return self.repo.get_successful_logs()

    def seed_test_log(self) -> None:
        self.repo.add_test_log()
