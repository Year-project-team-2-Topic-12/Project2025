from fastapi import APIRouter, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from backend.deps import get_auth_service
from backend.schemas.auth_schema import DeleteUserResponse, RegisterRequest, RegisterResponse, TokenResponse, UserResponse
from backend.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(get_auth_service),
):
    return service.login(form_data.username, form_data.password)


@router.post("/register", response_model=RegisterResponse)
def register(
    payload: RegisterRequest,
    authorization: str | None = Header(default=None),
    service: AuthService = Depends(get_auth_service),
):
    return service.register(payload.username, payload.password, authorization)


@router.get("/users", response_model=list[UserResponse])
def list_users(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    service: AuthService = Depends(get_auth_service),
):
    service.require_admin(credentials)
    return service.list_users()


@router.delete("/users/{username}", response_model=DeleteUserResponse)
def delete_user(
    username: str,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    service: AuthService = Depends(get_auth_service),
):
    service.require_admin(credentials)
    service.delete_user(username)
    return {"message": "Пользователь удалён"}
