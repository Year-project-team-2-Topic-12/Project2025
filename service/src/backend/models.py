from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

# создаем базу для моделей
Base = declarative_base()

# создаем класс, который позволит выводить логи в таблице

class RequestLog(Base):
    __tablename__ = 'request_logs'

    id = Column(Integer, primary_key=True, index=True) # уникальный id для каждой записи
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc)) # дата и время запроса
    input_meta = Column(String) # что пришло в запросе
    image_width = Column(Integer, nullable=True)     # ширина изображения
    image_height = Column(Integer, nullable=True)    # высота изображения
    duration = Column(Float) # сколько времени заняло
    result = Column(String) #
    status = Column(Integer) # HTTP код ответа

class User(Base):
    __tablename__ = "users"  # Имя таблицы в базе данных 

    id = Column(Integer, primary_key=True, index=True) # Уникальный номер (1, 2, 3...)
    username = Column(String, unique=True, index=True) # Логин. unique=True не даст создать двух юзеров с одним именем.
    hashed_password = Column(String) # Мы не храним пароль, мы храним его "хеш" (набор символов $2b$12$eX...)
    role = Column(String, default="user") # Роль 'user' (обычный) или 'admin' (суперпользователь)