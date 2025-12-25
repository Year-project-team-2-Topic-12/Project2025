from sqlalchemy.orm import Session

from backend.db.models import User


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_user_by_username(self, username: str):
        return self.session.query(User).filter(User.username == username).first()

    def create_user(self, username: str, password_hash: str, role: str = "user") -> User:
        new_user = User(username=username, hashed_password=password_hash, role=role)
        self.session.add(new_user)
        self.session.commit()
        self.session.refresh(new_user)
        return new_user
