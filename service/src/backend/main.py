from fastapi import FastAPI
from backend.routers import history, inference, stats,  auth
from backend import security

app = FastAPI(
    title="MURA Classifier API",
    description="API для хранения и просмотра истории запросов",
    version="1.0.0"
)

# Подключения Middleware
app.add_middleware(security.AuthMiddleware)

app.include_router(auth.router)
app.include_router(history.router)
app.include_router(stats.router)
app.include_router(inference.router)

# для тестовой записи логов при запуске
from backend.database import engine
from sqlalchemy.orm import Session as OrmSession
from backend import crud

with OrmSession(engine) as session:
    crud.add_test_log(session)