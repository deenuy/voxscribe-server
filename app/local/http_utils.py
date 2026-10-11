"""Small helpers so every endpoint turns a bad request into the same clean 4xx."""

from fastapi import Request
from pydantic import BaseModel, ValidationError

from .errors import InvalidRequestError


async def json_object(request: Request) -> dict:
    """The request body as a JSON object, or a 400 that says what was wrong (never a 500)."""
    try:
        body = await request.json()
    except ValueError:
        raise InvalidRequestError("body: not valid JSON") from None
    if not isinstance(body, dict):
        raise InvalidRequestError("body: must be a JSON object")
    return body


def parse_model(model: type[BaseModel], data: dict):
    """Validate `data` against a pydantic model, turning a failure into a 400 with the first problem."""
    try:
        return model(**data)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise InvalidRequestError(f"{where}: {first.get('msg', 'invalid value')}".strip(": ")) from None
