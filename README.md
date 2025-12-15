Реализовано:

- Сохранение каждого запроса к API (например, /forward) в базу данных
- Возможность просматривать историю запросов через GET `/history`
- Возможность удалить всю историю через DELETE `/history` (с авторизацией)
- Механизм Alembic-митаций для управления структурой БД

Структура:
persistence/
├── main.py # точка входа в приложение
├── models.py # модель таблицы RequestLog
├── crud.py # логика сохранения и чтения логов
├── database.py # подключение к SQLite через SQLAlchemy
├── routers/
│ └── history.py # маршруты /history (GET и DELETE)
├── alembic/ # миграции alembic
│ └── versions/ # автосозданные версии
├── request_logs.db # SQLite база данных (создается автоматически)
├── requirements.txt # зависимости
├── alembic.ini # настройки Alembic


Тестирование:
* Создать виртуальное окружение и установить requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

* Применить миграции Alembic, создаст файл базы данных history.db и таблицу request_logs
alembic upgrade head

* Запуск приложения (URL http://127.0.0.1:8000/docs)
uvicorn main:app --reload

GET /history возвращает список сохраненных логов, при запуске создается тестовый лог для проверки 

DELETE /history удаляет все записи из базы, требуется авторизация (верхний правый угол, пароль для тестов secret_token)

Alembic:
При внесении изменений в models.py позволяет обновить структуру базы данных одной командой 
* Создание новой миграции 
alembic revision --autogenerate -m "change message"
* Применение новой миграции
alembic upgrade head


Логи сохраняются в history.db



Пример использования в POST /forward:

from fastapi import APIRouter, UploadFile, HTTPException
from sqlalchemy.orm import Session
from database import engine
from crud import add_log
import time

router = APIRouter()

@router.post("/forward")
async def forward(image: UploadFile):
    start = time.time()

    # запуск модели
    result = # какой-то результат
    duration = round(time.time() - start, 3)

    # Подготовка данных для логирования
    log_data = {
        "input_meta": f"filename: {image.filename}", #информация о изображении
        "duration": duration,  # сколько времени заняла обработка
        "result": result, # результат
        "status_code": 200 # HTTP код ответа
    }

    # Логируем в базу
    with Session(engine) as session:
        add_log(session, log_data)

    return {"result": result}
