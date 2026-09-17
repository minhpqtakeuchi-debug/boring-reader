from app.core.errors.base.error_base import ErrorBase
from fastapi import status

class InternalServerError(ErrorBase):
    def __init__(self, message: str = "Internal Server Error"):
        super().__init__(message=message, status=status.HTTP_500_INTERNAL_SERVER_ERROR)