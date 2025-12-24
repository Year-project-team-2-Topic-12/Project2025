import time
from datetime import datetime, timedelta
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from backend.database import engine
from backend import crud

# Настройки
SECRET_KEY = "SUPER_SECRET_KEY_CHANGE_ME"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain, hashed):
    """Сравнивает пароль с хешем"""
    return pwd_context.verify(plain, hashed)

def get_hash(password):
    """Создает хеш пароля"""
    return pwd_context.hash(password)

def create_token(username: str):
    """Генерирует токен"""
    data = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# Middleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        
        # 1. Открытые пути
        OPEN_PATHS = [
            "/docs", "/openapi.json",
            "/auth/login",  # Вход открыт для всех
            "/favicon.ico"
        ]
        
        if request.url.path in OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        # 2. Проверка токена
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return Response("Unauthorized: No token provided", status_code=401)
        
        try:
            token_pure = token.split(" ")[1]
            payload = jwt.decode(token_pure, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except JWTError:
            return Response("Unauthorized: Invalid token", status_code=401)

        # 3. Проверка прав в БД
        with Session(engine) as db:
            user = crud.get_user_by_username(db, username)
            if not user:
                return Response("User not found", status_code=401)
            
            # ЗАЩИТА DELETE /history (Только Админ)
            if request.url.path == "/history" and request.method == "DELETE":
                if user.role != "admin":
                    return Response("Forbidden: Admins only", status_code=403)

        return await call_next(request)