from pydantic import BaseModel

class BaseSchema(BaseModel):
    model_config = {
        "validate_assignment": True,
    }