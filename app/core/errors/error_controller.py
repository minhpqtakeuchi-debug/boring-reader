from app.core.errors.base.error_base import ErrorBase
from app.core.errors.exceptions.internal_server_error import InternalServerError
from app.shared.dto.wrapers.standard_response import StandardResponse

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAIError
import psycopg
import logging

logger = logging.getLogger(__name__)

class ErrorController:
    @staticmethod
    def handle_exception(e: Exception):
        """Return standardized error response instead of plain HTTPException.
        Resets DB session + engine on DB errors, no retry.
        """
        message = str(e)
        details = {}
        logger.error(f"Error: {message}")

        # --- Database errors ---
        if isinstance(e, (SQLAlchemyError, psycopg.Error)):
            error = InternalServerError("Database error occurred")
            details = {"DB": message}

        # --- OpenAI errors ---
        elif isinstance(e, OpenAIError):
            error = InternalServerError("OpenAI service error occurred")
            details = {"OpenAI": message}

        elif isinstance(e, ErrorBase):
            error = e
        else:
            error = InternalServerError(message)

        return JSONResponse(
            status_code=error.status,
            content=StandardResponse(
                code=error.status,
                message=error.message,
                results=details
            ).model_dump()
        )
    
    @staticmethod
    def handle_http_exception(e: StarletteHTTPException):
        """Return standardized error response instead of plain HTTPException."""
        message = str(e)
        logger.error(f"Error: {message}")

        return JSONResponse(
            status_code=e.status_code,
            content=StandardResponse(
                code=e.status_code,
                message=f"API {e.detail}",
                results={}
            ).model_dump()
        )

    @staticmethod
    def handle_validation_error(e: RequestValidationError):
        """Return standardized error response for validation errors."""
        logger.error(f"Validation error: {e.errors()}")

        # Collect all validation errors
        error_details = [
            {
                "loc": err.get("loc", []),
                "input": err.get("input", ""),
                "msg": err.get("msg", ""),
                "type": err.get("type", "")
            }
            for err in e.errors()
        ]

        return JSONResponse(
            status_code=422,
            content=StandardResponse(
                code=422,
                message="Validation error",
                results={"errors": error_details}
            ).model_dump()
        )
