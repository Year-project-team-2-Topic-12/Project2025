from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import engine
from backend import crud
# Импортируем логику из соседнего файла security
from backend import security 

router = APIRouter(prefix="/auth", tags=["Auth"])

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