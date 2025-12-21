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