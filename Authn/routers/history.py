rom fastapi import APIRouter, HTTPException, Depends
from database import engine
import crud
import security  # Импортируем модуль безопасности
from sqlalchemy.orm import Session as OrmSession
from models import User # Импортируем модель пользователя для типизации

router = APIRouter(tags=["History"])

@router.get("/history")
def read_history(current_user: User = Depends(security.get_current_user)):
    """
    Просмотр истории. Доступен любому авторизованному пользователю.
    Сохраняем формат вывода, который написала коллега.
    """
    with OrmSession(engine) as session:
        logs = crud.get_all_logs(session)
    
    # Сохраняем оригинальную логику формирования списка
    return [
        {
            "id": log.id,
            "timestamp": log.timestamp,
            "input_meta": log.input_meta,
            "duration": log.duration,
            "result": log.result,
            "status": log.status,
        }
        for log in logs
    ]

@router.delete("/history")
def delete_history(admin_user: User = Depends(security.check_admin_privilege)):
    """
    Удаление истории. 
    Вместо ручной проверки 'secret_token', теперь работает твоя 
    проверка check_admin_privilege, которая лезет в БД и проверяет роль.
    """
    with OrmSession(engine) as session:
        crud.delete_all_logs(session)
        
    return {"message": f"История удалена администратором: {admin_user.username}"}
