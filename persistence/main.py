from fastapi import FastAPI
from routers import history, stats

app = FastAPI(
    title="MURA Classifier API",
    description="API для хранения и просмотра истории запросов",
    version="1.0.0"
)

app.include_router(history.router)
app.include_router(stats.router)


# для тестовой записи логов при запуске
from database import engine
from sqlalchemy.orm import Session as OrmSession
import crud

with OrmSession(engine) as session:
    crud.add_test_log(session)