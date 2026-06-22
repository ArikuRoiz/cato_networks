from __future__ import annotations

from uuid import UUID, uuid4


def str_to_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        return uuid4()
