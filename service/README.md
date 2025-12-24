**Реализовано:**

- Сохранение каждого запроса к API (например, /forward) в базу данных
- Возможность просматривать историю запросов через GET `/history`
- Возможность удалить всю историю через DELETE `/history` (с авторизацией)
- Механизм Alembic-митаций для управления структурой БД
- статистика по успешным запросам (среднее время, квантили, размеры изображений) через GET `/stats`
-Безопасный вход в систему с выдачей временных токенов доступа.
-Использование алгоритма bcrypt (библиотека passlib) для безопасного хранения данных пользователей.
-Распределения ролей на Admin(польное права) и User(может выполнить инференс и просматривать историю запросов).
-Middleware - защита на уровне приложения, который проверяет токены до того, как запрос попадет в роутеры.

**Структура:**
```text
backend/
├── main.py              # Точка входа: подключение роутеров и Middleware
├── models.py            # SQLAlchemy модели (RequestLog и User)
├── crud.py              # Логика взаимодействия с БД (логи и пользователи)
├── database.py          # Настройка подключения к SQLite
├── security.py          # Логика JWT, хеширование и AuthMiddleware
├── routers/
│   ├── auth.py          # Маршрут /auth/login (вход в систему)
│   ├── history.py       # Маршруты /history (просмотр и удаление)
│   └── stats.py         # Маршруты /stats (статистика)
├── alembic/             # Папка миграций БД
├── history.db           # SQLite база данных
├── pyproject.toml       # Зависимости проекта (включая passlib и jose)
├── alembic.ini          # Конфигурация Alembic
└── bootstrap.sh         # Скрипт автоматизации разработки
```


**Тестирование:**
* Используйте скрипт автоматизации для установки всех зависимостей (включая новые библиотеки для безопасности):
```text
./bootstrap.sh
# Выберите пункт 4 (Install ALL deps)
```

* Применить миграции Alembic, создаст файл базы данных 
./bootstrap.sh
# Выберите пункт 7 (Alembic revision --autogenerate), назовите "add_users"
# Выберите пункт 6 (Alembic upgrade head)
```
*Инициализация пользователей
Запустите скрипт для создания первого администратора и тестового пользователя:

```text
python init_admin.py
Admin: admin / admin

User: user / user
```

* Запуск приложения (URL http://127.0.0.1:8000/docs)
```text
./bootstrap.sh
# Выберите пункт 5 (Run backend)
```

Логи сохраняются в history.db

**GET /history** возвращает список сохраненных логов, при запуске создается тестовый лог для проверки  
**DELETE /history** удаляет все записи из базы, требуется авторизация (верхний правый угол, пароль для тестов secret_token)
**GET /stats** возвращает статистику по успешным запросам (status=200): среднее время обработки, квантили (50%, 95%, 99%) и средние размеры изображений


**Alembic:**
При внесении изменений в models.py позволяет обновить структуру базы данных одной командой 
* Создание новой миграции
```text
alembic revision --autogenerate -m "change message"
```
* Применение новой миграции
```text
alembic upgrade head
```


Пример использования в POST /forward:
```text
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
        "image_width": 2048,    # ширина изображения (для /stats)
        "image_height": 1536,   # высота изображения (для /stats)
        "duration": duration,  # сколько времени заняла обработка
        "result": result, # результат
        "status_code": 200 # HTTP код ответа
    }

    # Логируем в базу
    with Session(engine) as session:
        add_log(session, log_data)

    return {"result": result}
```
*Как работает авторизация 

В проекте реализован AuthMiddleware (находится в security.py), который перехватывает каждый входящий запрос:

Открытые пути: Пути /auth/login, /docs и /openapi.json открыты для всех.

Проверка токена: Для всех остальных путей проверяется заголовок Authorization: Bearer <token>.

Контроль ролей: Если пользователь пытается выполнить DELETE /history, Middleware проверяет, установлена ли у него роль admin в базе данных.

Если роль user, возвращается ошибка 403 Forbidden.

*Пример использования безопасности

1. Получение токена: Отправьте POST запрос на /auth/login с вашим логином и паролем. В ответ вы получите access_token.

2. Авторизация в Swagger: Нажмите кнопку Authorize в правом верхнем углу страницы /docs и вставьте полученный токен.

3. Защищенный запрос: Теперь при вызове /history сервер будет знать, кто вы, и разрешит (или запретит) действие в зависимости от вашей роли.
