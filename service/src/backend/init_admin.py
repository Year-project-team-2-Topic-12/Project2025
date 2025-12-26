from backend.db.connection import get_session
from backend.db.repositories.user_repository import UserRepository
from backend.services.auth_service import AuthService
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Проверяем, что пользователь admin создан...")
    session_gen = get_session()
    session = next(session_gen)
    try:
        auth_service = AuthService(UserRepository(session))
        auth_service.ensure_admin()
    finally:
        session_gen.close()
    logger.info("Проверка завершена")


if __name__ == "__main__":
    main()
