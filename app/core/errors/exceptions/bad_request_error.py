from app.core.errors.base.error_base import ErrorBase
from fastapi import status

class BadRequestError(ErrorBase):
    def __init__(self, message: str = "Bad Request"):
        super().__init__(message=message, status=status.HTTP_400_BAD_REQUEST)