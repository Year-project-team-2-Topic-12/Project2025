from sqlalchemy.orm import Session
from backend.models import RequestLog, User
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

# Функции для пользователей 

def get_user_by_username(session: Session, username: str):
    """
    Ищет пользователя в базе по логину.
    Нужен при входе в систему (login) и при проверке токена.
    """
    return session.query(User).filter(User.username == username).first()

def create_user(session: Session, username: str, password_hash: str, role: str = "user"):
    """
    Создает нового пользователя.
    Нужен для скрипта инициализации (чтобы создать первого админа).
    """
    new_user = User(username=username, hashed_password=password_hash, role=role)
    session.add(new_user)   # Добавляем в сессию
    session.commit()        # Сохраняем в файл БД
    session.refresh(new_user) # Получаем ID созданного юзера
    return new_user
