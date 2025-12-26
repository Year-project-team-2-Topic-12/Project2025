from datetime import datetime, timezone

from sqlalchemy import select, func, or_
from sqlalchemy.orm import Session

from backend.db.models import RequestLog

class RequestLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_all_logs(self):
        result = self.session.execute(select(RequestLog))
        return result.scalars().all()

    def delete_all_logs(self) -> None:
        self.session.execute(RequestLog.__table__.delete())
        self.session.commit()

    def add_log(self, log_data: dict | RequestLog) -> RequestLog:
        log = log_data if isinstance(log_data, RequestLog) else RequestLog(**log_data)
        self.session.add(log)
        self.session.commit()
        self.session.refresh(log)
        return log

    def add_test_log(self) -> None:
        new_log = RequestLog(
            timestamp=datetime.now(timezone.utc),
            input_meta="тестовые данные",
            image_width=2048,
            image_height=1536,
            duration=0.1,
            result="успех",
            status=200,
        )
        self.session.add(new_log)
        self.session.commit()

    def get_successful_logs(self):
        result_field = func.lower(RequestLog.result)
        no_error = or_(RequestLog.result.is_(None), ~result_field.contains("ошибка"))
        return self.session.query(RequestLog).filter(no_error).all()
