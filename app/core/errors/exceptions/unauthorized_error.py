from app.core.errors.base.error_base import ErrorBase
from fastapi import status

class UnauthorizedError(ErrorBase):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message=message, status=status.HTTP_401_UNAUTHORIZED)