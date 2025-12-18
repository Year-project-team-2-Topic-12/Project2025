from fastapi import FastAPI
from sqlalchemy.orm import Session
from database import engine
import crud
from routers import auth, history

app = FastAPI(
    title="MURA Classifier API",
    description="ML-сервис с авторизацией (JWT) и историей запросов.",
    version="2.0.0"
)

# --- ПОДКЛЮЧЕНИЕ РОУТЕРОВ ---
# 1. Роутер авторизации (/auth/login)
app.include_router(auth.router)

# 2. Роутер истории (/history)
app.include_router(history.router)


# --- СОБЫТИЯ ЗАПУСКА ---
@app.on_event("startup")
def startup_event():
    with Session(engine) as session:
        crud.add_test_log(session)
        print("--- Сервер запущен. Тестовый лог создан. ---")
