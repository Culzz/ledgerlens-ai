"""Model output parsing and the answer contract.

Shared by the evaluation harness and, later, by the serving path, so a
prediction is scored on exactly what the product would have returned.
"""

from .answers import (
    FORMAT_FALLBACK,
    FORMAT_JSON,
    FORMAT_MARKER,
    FORMAT_UNPARSED,
    ParsedAnswer,
    extract_json_object,
    last_number_in,
    normalise_text,
    parse_answer,
    to_decimal,
)

__all__ = [
    "FORMAT_FALLBACK",
    "FORMAT_JSON",
    "FORMAT_MARKER",
    "FORMAT_UNPARSED",
    "ParsedAnswer",
    "extract_json_object",
    "last_number_in",
    "normalise_text",
    "parse_answer",
    "to_decimal",
]
