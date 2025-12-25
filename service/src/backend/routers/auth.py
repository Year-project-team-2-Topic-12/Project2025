from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import engine
from backend import crud
# Импортируем логику из соседнего файла security
from backend import security 

router = APIRouter(prefix="/auth", tags=["Auth"])

class RegisterRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as db:
        user = crud.get_user_by_username(db, form_data.username)
        
        # Проверяем пароль через функцию в security.py
        if not user or not security.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Неверный логин или пароль")
        
        # Создаем токен через функцию в security.py
        token = security.create_token(user.username)
        
        return {"access_token": token, "token_type": "bearer"}


@router.post("/register")
def register(payload: RegisterRequest, authorization: str | None = Header(default=None)):
    if not payload.username.strip() or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: No token provided")

    token_pure = authorization.split(" ")[1]
    try:
        payload_data = jwt.decode(token_pure, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        username = payload_data.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")

    with Session(engine) as db:
        if not username:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")
        user = crud.get_user_by_username(db, username)
        if not user or user.role != "admin":
            raise HTTPException(status_code=403, detail="Forbidden: Admins only")

        if crud.get_user_by_username(db, payload.username):
            raise HTTPException(status_code=409, detail="User already exists")

        pwd_hash = security.get_hash(payload.password)
        new_user = crud.create_user(db, payload.username, pwd_hash, role="user")
        return {"username": new_user.username, "role": new_user.role}
