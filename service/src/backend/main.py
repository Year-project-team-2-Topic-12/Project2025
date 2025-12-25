from fastapi import FastAPI
from backend.routers import history, inference, stats, auth
from backend.db.connection import get_session
from backend.db.repositories.user_repository import UserRepository
from backend.services.auth_service import AuthMiddleware, AuthService
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Запуск приложения...")
    session_gen = get_session()
    session = next(session_gen)
    try:
        auth_service = AuthService(UserRepository(session))
        auth_service.ensure_admin()
    finally:
        session_gen.close()
    yield
    print("Завершение приложения...")

app = FastAPI(
    title="MURA Classifier API",
    description="API для хранения и просмотра истории запросов",
    version="1.0.0",
    servers=[{"url": ""}],
    lifespan=lifespan,
)

# Подключения Middleware
app.add_middleware(AuthMiddleware)

app.include_router(auth.router)
app.include_router(history.router)
app.include_router(stats.router)
app.include_router(inference.router)
