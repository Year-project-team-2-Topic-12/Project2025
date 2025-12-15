from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# создаем базу для моделей
Base = declarative_base()

# создаем класс, который позволит выводить логи в таблице

class RequestLog(Base):
    __tablename__ = 'request_logs'

    id = Column(Integer, primary_key=True, index=True) # уникальный id для каждой записи
    timestamp = Column(DateTime, default=datetime.utcnow) # дата и время запроса
    input_meta = Column(String) # что пришло в запросе
    duration = Column(Float) # сколько времени заняло
    result = Column(String) #
    status = Column(Integer) # HTTP код ответа