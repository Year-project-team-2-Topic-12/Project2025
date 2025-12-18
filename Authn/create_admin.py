from sqlalchemy.orm import Session
from database import engine
import crud
import security

def init_db():
    """
    Создает начальных пользователей в базе данных.
    Запускать этот скрипт нужно один раз, после применения миграций.
    """
    # Создаем сессию подключения к БД
    with Session(engine) as session:
        
        if crud.get_user_by_username(session, "admin"):
            print("⚠️  Пользователь 'admin' уже существует. Пропускаем.")
        else:
            print("⚙️  Создаем администратора...")
            # Хешируем пароль
            admin_hash = security.get_password_hash("adminpass")
            # Сохраняем в БД
            crud.create_user(session, "admin", admin_hash, role="admin")
            print("✅ Admin создан (Login: admin / Pass: adminpass)")

        # 2. Проверяем, есть ли обычный пользователь 
            print("⚙️  Создаем пользователя...")
            user_hash = security.get_password_hash("user123")
            crud.create_user(session, "User", user_hash, role="user")
            print("✅ User создан (Login: User / Pass: user123)")

if __name__ == "__main__":
    print("--- Начало инициализации БД ---")
    try:
        init_db()
        print("--- Готово! Теперь можно запускать uvicorn ---")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Подсказка: Вы применили миграции (alembic upgrade head)? Таблица 'users' должна существовать.")
