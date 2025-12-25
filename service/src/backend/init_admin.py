from backend.db.connection import get_session
from backend.db.repositories.user_repository import UserRepository
from backend.services.auth_service import AuthService


def main() -> None:
    print("Проверяем администратора...")
    session_gen = get_session()
    session = next(session_gen)
    try:
        auth_service = AuthService(UserRepository(session))
        auth_service.ensure_admin()
    finally:
        session_gen.close()
    print("Проверка завершена")


if __name__ == "__main__":
    main()
