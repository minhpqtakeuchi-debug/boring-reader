from app.core.errors.base.error_base import ErrorBase
from fastapi import status

class ForbiddenError(ErrorBase):
    def __init__(self, message: str = "Forbiden"):
        super().__init__(message=message, status=status.HTTP_403_FORBIDDEN)