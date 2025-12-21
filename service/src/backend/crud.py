from sqlalchemy.orm import Session
from backend.models import RequestLog
from sqlalchemy import select
from datetime import datetime, timezone

def get_all_logs(session: Session):
    result = session.execute(select(RequestLog))
    return result.scalars().all()

def delete_all_logs(session: Session):
    session.execute(RequestLog.__table__.delete())
    session.commit()

def add_log(session: Session, log_data: dict):
    log = RequestLog(**log_data)
    session.add(log)
    session.commit()
    session.refresh(log)
    return log

def add_test_log(session):
    new_log = RequestLog(
        timestamp=datetime.now(timezone.utc),
        input_meta="тестовые данные",
        image_width=2048,
        image_height=1536,
        duration=0.1,
        result="успех",
        status=200
    )
    session.add(new_log)
    session.commit()

def get_successful_logs(session: Session):
    return session.query(RequestLog).filter(RequestLog.status == 200).all()
