"""Model output parsing and the answer contract.

Shared by the evaluation harness and, later, by the serving path, so a
prediction is scored on exactly what the product would have returned.
"""

from .answers import (
    FORMAT_FALLBACK,
    FORMAT_JSON,
    FORMAT_MARKER,
    FORMAT_UNEXECUTED,
    FORMAT_UNPARSED,
    ParsedAnswer,
    extract_json_object,
    last_number_in,
    normalise_text,
    parse_answer,
    to_decimal,
)
from .orchestrator import (
    PROMPT_VERSION,
    Citation,
    Document,
    GroundedAnswer,
    Orchestrator,
    Retrieval,
    build_program_messages,
    label_for,
)

__all__ = [
    "FORMAT_FALLBACK",
    "FORMAT_JSON",
    "FORMAT_MARKER",
    "FORMAT_UNEXECUTED",
    "FORMAT_UNPARSED",
    "PROMPT_VERSION",
    "Citation",
    "Document",
    "GroundedAnswer",
    "Orchestrator",
    "ParsedAnswer",
    "Retrieval",
    "build_program_messages",
    "extract_json_object",
    "label_for",
    "last_number_in",
    "normalise_text",
    "parse_answer",
    "to_decimal",
]
