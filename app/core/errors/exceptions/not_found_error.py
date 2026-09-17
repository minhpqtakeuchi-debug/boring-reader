from app.core.errors.base.error_base import ErrorBase
from fastapi import status

class NotFoundError(ErrorBase):
    def __init__(self, message: str = "Not Found"):
        super().__init__(message=message, status=status.HTTP_404_NOT_FOUND)