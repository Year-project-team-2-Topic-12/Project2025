from passlib.context import CryptContext

# Настройка хеширования
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Функция для проверки пароля
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Функция для генерации хеша (нужна, чтобы создать админа)
def get_password_hash(password):
    return pwd_context.hash(password)

# Симуляция БД. Пароль "adminpass" захеширован
USERS_DATA = [
    {
        "username": "admin", 
        "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    },
    {                            
        "username": "Sunnatilla19",
        "password": "$2b$12$XNRgNStayicmUAAW6uIBlOeo99hJpmb1YNyoqtUMJgCOuLFgLry2G"
    }
]

def get_user(username: str):
    for user in USERS_DATA:
        if user.get("username") == username:
            return user
    return None