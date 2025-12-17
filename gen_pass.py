from passlib.context import CryptContext

# Настройки как в вашем проекте
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Генерация хеша
print(pwd_context.hash("Ваше_пароль"))

# Проверка пароля