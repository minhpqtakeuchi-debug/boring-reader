from fastapi import HTTPException, status

class ErrorBase(Exception):
    def __init__(self, message: str, status: status):
        self.message = message
        self.status = status
        super().__init__(self.message)

    def to_http_exception(self) -> HTTPException:
        return HTTPException(status_code=self.status, detail=self.message)