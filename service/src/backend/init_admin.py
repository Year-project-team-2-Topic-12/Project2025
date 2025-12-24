from backend.database import engine
from sqlalchemy.orm import Session
from backend import crud
from backend import security 

def init():
    print("Создаем администратора...")
    with Session(engine) as db:
        if crud.get_user_by_username(db, "admin"):
            print("Админ уже существует")
        else:
            # Берем хешер из security
            pwd_hash = security.get_hash("admin") 
            crud.create_user(db, "admin", pwd_hash, role="admin")
            print("Админ создан! (admin / admin)")

if __name__ == "__main__":
    init()