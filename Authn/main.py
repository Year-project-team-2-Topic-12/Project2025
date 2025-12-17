# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm # Стандартная форма для логина
from security import create_jwt_token, get_current_user
from db import get_user, verify_password

app = FastAPI()

# Обрати внимание: вместо User модели мы используем form_data: OAuth2PasswordRequestForm
# Это позволяет Swagger UI отправлять username/password правильно.
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    
    # 1. Проверяем существование пользователя
    # 2. Проверяем хеш пароля
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Если всё ок — выдаем токен
    access_token = create_jwt_token({"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


# Пример защищенного маршрута
@app.get("/users/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    user = get_user(current_user)
    return user