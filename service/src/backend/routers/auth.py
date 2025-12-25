from fastapi import APIRouter, Depends, Header
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from backend.deps import get_auth_service
from backend.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])

class RegisterRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(get_auth_service),
):
    return service.login(form_data.username, form_data.password)


@router.post("/register")
def register(
    payload: RegisterRequest,
    authorization: str | None = Header(default=None),
    service: AuthService = Depends(get_auth_service),
):
    return service.register(payload.username, payload.password, authorization)
