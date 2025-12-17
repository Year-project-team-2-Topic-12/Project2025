 1. Установка библиотек
```bash
pip install fastapi uvicorn PyJWT passlib[bcrypt] bcrypt==3.2.0 python-multipart

или воспользоваться requirements.txt

2. Запуск сервера Bash
python -m uvicorn main:app --reload

3. Работа с паролями и хешированием реализовано в файле gen_pass.py. надо вести свой парол и запустить скрипт он вам ответ сгенерируйт хеш и его нада с логинам добавить в файл db.py.

4. Тестирование через Swagger: Откройте в браузере: http://127.0.0.1:8000/docs Нажмите зеленую кнопку Authorize справа сверху.Введите данные одного из пользователей .Нажмите Authorize, затем Close. Теперь все защищенные роуты будут работать под вашим токеном.

5. Как защитить свой код

Чтобы ваш эндпоинт (например, /forward или /history) был доступен только после входа, добавьте зависимость get_current_user.
Пример:

from fastapi import Depends
from security import get_current_user # Импорт функции
@app.post("/forward")   
async def function(data: dict, user: str = Depends(get_current_user)):
    # Если токен невалиден, FastAPI сам вернет ошибку 401.
    # Если всё ок, в переменной 'user' будет имя авторизованного пользователя.
    return {"status": "success", "user": user}

