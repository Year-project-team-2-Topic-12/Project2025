from datetime import datetime, timedelta

from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware

from backend.db.connection import get_session
from backend.db.repositories.user_repository import UserRepository
from ml import env

# Auth settings
SECRET_KEY = env.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_token(username: str) -> str:
    data = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")
    return username


class AuthService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def login(self, username: str, password: str) -> dict:
        user = self.repo.get_user_by_username(username)
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Неверный логин или пароль")

        token = create_token(user.username)
        return {"access_token": token, "token_type": "bearer"}

    def _get_user_from_token(self, token: str):
        username = decode_token(token)
        user = self.repo.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    def require_admin(self, credentials: HTTPAuthorizationCredentials) -> None:
        token = credentials.credentials
        user = self._get_user_from_token(token)
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Forbidden: Admins only")

    def register(self, username: str, password: str, authorization: str | None) -> dict:
        if not username.strip() or not password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: No token provided")

        token_pure = authorization.split(" ")[1]
        user = self._get_user_from_token(token_pure)
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Forbidden: Admins only")

        if self.repo.get_user_by_username(username):
            raise HTTPException(status_code=409, detail="User already exists")

        pwd_hash = get_hash(password)
        new_user = self.repo.create_user(username, pwd_hash, role="user")
        return {"username": new_user.username, "role": new_user.role}

    def ensure_admin(self) -> None:
        if self.repo.get_user_by_username("admin"):
            return

        password = env.ADMIN_PASSWORD or "admin"
        pwd_hash = get_hash(password)
        self.repo.create_user("admin", pwd_hash, role="admin")

    def list_users(self) -> list[dict]:
        users = self.repo.list_users()
        return [{"username": user.username, "role": user.role} for user in users]

    def delete_user(self, username: str) -> None:
        if username == "admin":
            raise HTTPException(status_code=400, detail="Cannot delete admin user")
        deleted = self.repo.delete_user(username)
        if not deleted:
            raise HTTPException(status_code=404, detail="User not found")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        open_paths = [
            "/docs",
            "/openapi.json",
            "/auth/login",
            "/favicon.ico",
        ]

        if request.url.path in open_paths or request.method == "OPTIONS":
            return await call_next(request)

        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return Response("Unauthorized: No token provided", status_code=401)

        token_pure = token.split(" ")[1]
        try:
            username = decode_token(token_pure)
        except HTTPException:
            return Response("Unauthorized: Invalid token", status_code=401)

        session_gen = get_session()
        session = next(session_gen)
        try:
            repo = UserRepository(session)
            user = repo.get_user_by_username(username)
            if not user:
                return Response("User not found", status_code=401)

            if request.url.path == "/history" and request.method == "DELETE":
                if user.role != "admin":
                    return Response("Forbidden: Admins only", status_code=403)
        finally:
            session_gen.close()

        return await call_next(request)
