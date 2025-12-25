from backend.database import engine
from sqlalchemy.orm import Session
from backend import crud
from backend import security
from ml import env


def ensure_admin():
    print("Проверяем администратора...")
    with Session(engine) as db:
        if crud.get_user_by_username(db, "admin"):
            print("Админ уже существует")
            return

        password = env.ADMIN_PASSWORD
        print(f"Создаём админа с паролем из переменных окружения: {password}")
        pwd_hash = security.get_hash(password)
        crud.create_user(db, "admin", pwd_hash, role="admin")
        print("Админ создан! (admin / заданный пароль)")
