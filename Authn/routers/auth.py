from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import engine
import security, crud

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as db:
        user = crud.get_user_by_username(db, form_data.username)
        if not user or not security.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Неверный логин или пароль")
        
        token = security.create_access_token(data={"sub": user.username})
        return {"access_token": token, "token_type": "bearer"}
