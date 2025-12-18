from sqlalchemy.orm import Session
from models import RequestLog
from sqlalchemy import select

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
        input_meta="тестовые данные",
        duration=0.1,
        result="успех",
        status=200
    )
    session.add(new_log)
    session.commit()

def get_user_by_username(session: Session, username: str):
    """Поиск пользователя по имени для проверки пароля"""
    return session.query(User).filter(User.username == username).first()

def create_user(session: Session, username: str, password_hash: str, role: str = "user"):
    """Создание нового пользователя (для скрипта create_admin.py)"""
    db_user = User(username=username, hashed_password=password_hash, role=role)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user
