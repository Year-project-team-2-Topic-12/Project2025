from backend.schemas.base_schema import BaseSchema


class TokenResponse(BaseSchema):
    access_token: str
    token_type: str


class RegisterRequest(BaseSchema):
    username: str
    password: str


class RegisterResponse(BaseSchema):
    username: str
    role: str


class UserResponse(BaseSchema):
    username: str
    role: str


class DeleteUserResponse(BaseSchema):
    message: str
