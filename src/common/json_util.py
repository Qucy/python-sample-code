"""
JSON helpers for converting Python structures to JSON strings suitable for storage.

Primary use case: insert list-of-dicts into PostgreSQL JSONB columns using %s placeholders.

Example (psycopg2 style):
    cursor.execute(
        "INSERT INTO my_table (data) VALUES (%s::jsonb)",
        (to_json_array([{"a": 1}, {"b": 2}]),)
    )
"""

import json
from typing import Any
from datetime import date, datetime
from decimal import Decimal


def _json_default(o: Any):
    """Best-effort conversion for non-JSON-native types."""
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        # Convert Decimal to float for JSON
        return float(o)
    # Pydantic/dataclass-like objects
    if hasattr(o, "dict") and callable(getattr(o, "dict")):
        try:
            return o.dict()
        except Exception:
            pass
    if hasattr(o, "__dict__"):
        try:
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    # Fallback to string representation
    return str(o)


def to_json_array(arr: Any, *, ensure_ascii: bool = False, sort_keys: bool = False) -> str:
    """
    Convert a Python list (e.g. [{...}, {...}]) into a JSON array string.

    - Accepts list, tuple, or set and normalizes to list.
    - Uses a default converter for common non-serializable types (datetime, Decimal, objects).
    - Returns a JSON string suitable for inserting into a JSONB column.

    Raises ValueError if input is not list-like.
    """
    if arr is None:
        return "[]"
    if isinstance(arr, (tuple, set)):
        arr = list(arr)
    if not isinstance(arr, list):
        raise ValueError("to_json_array expects a list/tuple/set")
    return json.dumps(arr, ensure_ascii=ensure_ascii, sort_keys=sort_keys, default=_json_default)