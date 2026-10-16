"""Deterministic calculation engine for FinQA's reasoning DSL.

FinQA asks questions whose answers are produced by a small program, e.g.

    subtract(5829, 5735)
    divide(subtract(150, 120), 120)
    subtract(150, 120), divide(#0, 120), multiply(#1, 100)

A program is a comma-separated list of steps. Each step is a call whose
arguments are numbers, ``#n`` references to the result of step n, or nested
calls. Running a gold program should reproduce the reference answer exactly.

Why this exists separately from the LLM: the language model plans and explains,
but it never supplies a number. This is the only thing that does arithmetic.

Decimal, not float
------------------
Financial figures do not tolerate binary floating point. ``0.1 + 0.2`` is not
``0.3`` in float, and a percentage change computed that way can be off in the
last digits, which is exactly what a benchmark tolerance check cares about.
Everything here uses ``Decimal``.

What is verified
----------------
Everything below was checked against the real dev split before being relied on:

* the arithmetic operations and the step/reference encoding (``#0`` is step 0)
* ``exe_ans`` is the executed result of ``program``
* ``const_100`` is the literal 100, and ``const_m1`` is -1 (an ``m`` prefix
  means minus). 159 dev records use these.
* ``greater`` returns ``"yes"``/``"no"``
* table operations name their row by label, not by index: ``table_average(settlements, none)``
  is "every value in the row labelled settlements", and the second argument is
  ``none`` when there is no column restriction
* cells carry footnote markers: ``-13 ( 13 )`` is -13
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

GREATER_TRUE = "yes"
GREATER_FALSE = "no"

MAX_STEPS = 64


class CalculationError(Exception):
    """Base class for anything that stops a program from producing an answer."""


class ProgramSyntaxError(CalculationError):
    """The program text could not be parsed."""


class UnsupportedOperation(CalculationError):
    """The program used an operation this engine does not implement."""


class UnresolvedReference(CalculationError):
    """A ``#n`` reference pointed at a step that does not exist."""


class InvalidArgument(CalculationError):
    """An argument was the wrong kind of thing."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUMBER = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)%?")
_REF = re.compile(r"#[0-9]+")


@dataclass(frozen=True)
class _Token:
    kind: str  # name | number | ref | comma | lparen | rparen
    text: str


def _tokenize(program: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(program)
    while i < n:
        ch = program[i]
        if ch.isspace():
            i += 1
            continue
        if ch == ",":
            tokens.append(_Token("comma", ch))
            i += 1
            continue
        if ch == "(":
            tokens.append(_Token("lparen", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Token("rparen", ch))
            i += 1
            continue
        if ch == "#":
            m = _REF.match(program, i)
            if not m:
                raise ProgramSyntaxError(f"bad reference at position {i}: {program[i:i + 8]!r}")
            tokens.append(_Token("ref", m.group(0)))
            i = m.end()
            continue
        m = _NAME.match(program, i)
        if m:
            word = m.group(0)
            # FinQA writes named constants as const_<value>: const_100 is 100.
            # A leading `m` means minus, so const_m1 is -1. Treating these as
            # function calls was the single biggest source of failures against
            # the real split.
            if word.startswith("const_"):
                body = word[len("const_") :]
                negative = body.startswith("m")
                digits = body[1:] if negative else body
                if not digits or not re.fullmatch(r"\d+(?:\.\d+)?", digits):
                    raise ProgramSyntaxError(
                        f"malformed constant {word!r} in {program!r}"
                    )
                tokens.append(_Token("number", ("-" if negative else "") + digits))
            else:
                tokens.append(_Token("name", word))
            i = m.end()
            continue
        m = _NUMBER.match(program, i)
        if m:
            tokens.append(_Token("number", m.group(0)))
            i = m.end()
            continue
        raise ProgramSyntaxError(f"unexpected character {ch!r} at position {i} in {program!r}")
    return tokens


@dataclass(frozen=True)
class Literal:
    """A number or a ``#n`` reference."""

    text: str

    @property
    def is_reference(self) -> bool:
        return self.text.startswith("#")

    @property
    def reference_index(self) -> int:
        return int(self.text[1:])


@dataclass(frozen=True)
class Call:
    """A function call: a name and its arguments."""

    name: str
    args: tuple[Any, ...]  # Literal or Call


class _Parser:
    def __init__(self, tokens: Sequence[_Token], program: str) -> None:
        self._tokens = tokens
        self._pos = 0
        self._program = program

    def _peek(self) -> _Token | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> _Token:
        tok = self._peek()
        if tok is None:
            raise ProgramSyntaxError(f"unexpected end of program in {self._program!r}")
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> _Token:
        tok = self._next()
        if tok.kind != kind:
            raise ProgramSyntaxError(
                f"expected {kind} but found {tok.text!r} in {self._program!r}"
            )
        return tok

    def parse_expression(self) -> Any:
        tok = self._next()
        if tok.kind in ("number", "ref"):
            return Literal(tok.text)
        if tok.kind != "name":
            raise ProgramSyntaxError(
                f"expected a name, number or reference but found {tok.text!r} "
                f"in {self._program!r}"
            )
        if tok.text in _TABLE_OPS:
            # Table operations take labels, not expressions, and a label can be
            # several words ("operating margin"). Parse their arguments as raw
            # text rather than as nested calls.
            return Call(tok.text, self._parse_table_arguments())
        self._expect("lparen")
        args: list[Any] = []
        if self._peek() is not None and self._peek().kind != "rparen":
            args.append(self.parse_expression())
            while self._peek() is not None and self._peek().kind == "comma":
                self._next()
                args.append(self.parse_expression())
        self._expect("rparen")
        return Call(tok.text, tuple(args))

    def _parse_table_arguments(self) -> tuple[Literal, ...]:
        """Read a table operation's arguments as labels.

        The arguments are row and column labels, so they can be several words
        ("operating margin") or the keyword ``none``. They are not expressions,
        and treating them as expressions is what breaks on the real data.
        """
        self._expect("lparen")
        args: list[Literal] = []
        current: list[str] = []

        while True:
            tok = self._peek()
            if tok is None:
                raise ProgramSyntaxError(
                    f"unterminated table operation in {self._program!r}"
                )
            if tok.kind == "rparen":
                self._next()
                args.append(Literal(" ".join(current).strip()))
                break
            if tok.kind == "comma":
                self._next()
                args.append(Literal(" ".join(current).strip()))
                current = []
                continue
            if tok.kind == "lparen":
                raise ProgramSyntaxError(
                    f"unexpected '(' in table arguments in {self._program!r}"
                )
            self._next()
            current.append(tok.text)

        if any(not a.text for a in args):
            raise ProgramSyntaxError(
                f"empty table argument in {self._program!r}"
            )
        return tuple(args)

    def parse_program(self) -> list[Call]:
        steps = [self.parse_expression()]
        while self._peek() is not None and self._peek().kind == "comma":
            self._next()
            steps.append(self.parse_expression())
        if self._peek() is not None:
            raise ProgramSyntaxError(
                f"trailing {self._peek().text!r} after program in {self._program!r}"
            )
        # A bare number is not a step. Every step in a FinQA program is a call,
        # and accepting a constant here would let a malformed program through as
        # if it had produced an answer.
        for step in steps:
            if not isinstance(step, Call):
                raise ProgramSyntaxError(
                    f"every step must be a function call, found {step.text!r} "
                    f"in {self._program!r}"
                )
        return steps


def parse_program(program: str) -> list[Call]:
    """Parse a program string into its steps."""
    if not program or not program.strip():
        raise ProgramSyntaxError("empty program")
    steps = _Parser(_tokenize(program), program).parse_program()
    if len(steps) > MAX_STEPS:
        raise ProgramSyntaxError(f"program has {len(steps)} steps, limit is {MAX_STEPS}")
    return steps


# --------------------------------------------------------------------------
# Table selection
# --------------------------------------------------------------------------

Table = Sequence[Sequence[str]]

# A resolver turns a table operation's literal arguments into the cell values it
# should operate over.
TableResolver = Callable[[Table, Sequence[Literal]], list[Decimal]]


def _normalise_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _find_row(table: Table, label: str) -> int | None:
    for index, row in enumerate(table):
        if row and _normalise_label(row[0]) == label:
            return index
    return None


def _find_column(table: Table, spec: str) -> int | None:
    if not table:
        return None
    wanted = _normalise_label(spec)
    for index, cell in enumerate(table[0]):
        if _normalise_label(cell) == wanted:
            return index
    return None


def default_table_resolver(table: Table, args: Sequence[Literal]) -> list[Decimal]:
    """Pick the cells a table operation works over.

    FinQA names the row by its label, not its index. Real examples:

        table_average(settlements, none)
        table_max(operating margin, none)
        table_min(segment operating income, none)
        table_sum(total obligations, none)

    The second argument is the string ``none`` when the whole row is wanted, or
    a column label to restrict to one column. Labels are matched
    case-insensitively with whitespace collapsed.

    Cells that are not numbers are skipped rather than treated as zero, because
    these rows mix figures with text notes (``( in millions )`` and similar). If
    nothing in the selection parses, that is an error, not an empty result.
    """
    if len(args) == 1:
        row_label, column_spec = args[0], Literal("none")
    elif len(args) == 2:
        row_label, column_spec = args
    else:
        raise InvalidArgument(
            "a table operation takes a row label and an optional column, got "
            f"{len(args)} argument(s)"
        )

    if not table:
        raise InvalidArgument("table operation used but no table was provided")

    if not all(isinstance(a, Literal) for a in (row_label, column_spec)):
        raise InvalidArgument(
            "a table operation takes literal labels, not nested calls"
        )

    label = _normalise_label(row_label.text)
    row_index = _find_row(table, label)
    if row_index is None:
        raise InvalidArgument(f"no row labelled {row_label.text!r} in the table")

    row = table[row_index]
    spec = column_spec.text

    if spec == "none":
        cells = list(row[1:])
    else:
        column = _find_column(table, spec)
        if column is None:
            raise InvalidArgument(f"no column labelled {spec!r} in the table")
        if column >= len(row):
            raise InvalidArgument(f"row {row_index} has no column {column}")
        cells = [row[column]]

    values: list[Decimal] = []
    for cell in cells:
        try:
            values.append(_parse_number(cell))
        except InvalidArgument:
            continue

    if not values:
        where = f"row {label!r}" if spec == "none" else f"row {label!r} column {spec!r}"
        raise InvalidArgument(f"no numeric values in {where}")
    return values


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"

# A trailing "( 13 )" is a footnote marker. Chained at the end so it only fires
# when something precedes it: on its own, "(1,234)" is a negative number.
_FOOTNOTE = re.compile(r"^(.*?)\s*\(\s*[^()]*\s*\)\s*$")


def _parse_number(text: str) -> Decimal:
    """Parse a financial figure.

    Handles the shapes that actually turn up in these tables: ``1,234``,
    ``$1,234``, ``(1,234)`` for negatives, ``-13 ( 13 )`` where the parenthesised
    part is a footnote, ``100% ( 100 % )``, ``12.5%``, ``.455``, a bare dash
    meaning zero, and the various unicode dashes that arrive from PDF extraction.

    A percent sign means the value is a fraction, so ``12.5%`` is 0.125. That
    applies to program literals too. Confirmed against the real dev split:
    ``multiply(25%, 5283.3)`` answers 1320.825, and the operating-margin row of
    AON/2014/page_47.pdf-1 averages to 0.08867 rather than 8.867.
    """
    if text is None:
        raise InvalidArgument("cannot parse None as a number")

    raw = str(text).strip()
    if raw == "":
        raise InvalidArgument("empty cell cannot be parsed as a number")

    footnote = _FOOTNOTE.match(raw)
    if footnote and footnote.group(1).strip():
        raw = footnote.group(1).strip()

    for dash in _DASHES:
        raw = raw.replace(dash, "-")

    raw = raw.replace("\u00a0", "").replace(" ", "")

    # A lone dash is how these tables usually write zero.
    if raw in ("-", "--"):
        return Decimal(0)

    negative = False

    # Parenthesised negatives, e.g. (1,234)
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1]

    raw = raw.removeprefix("+")

    if raw.endswith("%"):
        percent = True
        raw = raw[:-1]
    else:
        percent = False

    raw = raw.replace("$", "").replace(",", "")

    if raw in ("", "-", "--"):
        raise InvalidArgument(f"cannot parse {text!r} as a number")

    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise InvalidArgument(f"cannot parse {text!r} as a number") from exc

    if percent:
        value = value / Decimal(100)

    return -value if negative else value


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """One executed step, kept so the arithmetic can be shown later."""

    index: int
    operation: str
    arguments: tuple[str, ...]
    result: Decimal | str

    def render(self) -> str:
        args = ", ".join(self.arguments)
        return f"#{self.index} = {self.operation}({args}) = {self.result}"


@dataclass(frozen=True)
class CalculationTrace:
    """The full record of how an answer was computed."""

    program: str
    steps: tuple[Step, ...]
    result: Decimal | str
    table_resolver: str = "default"

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.result, Decimal)

    def render(self) -> str:
        lines = [f"program: {self.program}"]
        lines.extend("  " + s.render() for s in self.steps)
        lines.append(f"result:  {self.result}")
        return "\n".join(lines)


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(1 if value else 0)
    if isinstance(value, (int,)):
        return Decimal(value)
    if isinstance(value, float):
        # str() first: Decimal(0.1) carries the binary error, Decimal("0.1") does not.
        return Decimal(str(value))
    return _parse_number(value)


def _require_numeric(value: Any, operation: str) -> Decimal:
    if isinstance(value, str):
        raise InvalidArgument(
            f"{operation} needs numbers but step produced {value!r}"
        )
    return _decimal(value)


def _apply_numeric(op: str, args: Sequence[Decimal]) -> Decimal:
    if op == "add":
        return args[0] + args[1]
    if op == "subtract":
        return args[0] - args[1]
    if op == "multiply":
        return args[0] * args[1]
    if op == "divide":
        if args[1] == 0:
            raise CalculationError(f"division by zero: {args[0]} / {args[1]}")
        return args[0] / args[1]
    if op == "exp":
        base, power = args
        if power == power.to_integral_value():
            return base ** int(power)
        # Fractional powers have no exact decimal form; go through float and
        # note it, rather than pretending to be exact.
        return Decimal(str(float(base) ** float(power)))
    raise UnsupportedOperation(op)


_NUMERIC_OPS = {"add", "subtract", "multiply", "divide", "exp"}
_TABLE_OPS = {"table_max", "table_min", "table_sum", "table_average"}


def execute_program(
    program: str,
    *,
    table: Table | None = None,
    table_resolver: TableResolver | None = None,
) -> CalculationTrace:
    """Run a program and return its result plus a step-by-step trace.

    Raises ``CalculationError`` (or a subclass) on anything it cannot execute.
    Silently returning a plausible number on a failure would defeat the point.
    """
    steps = parse_program(program)
    results: list[Decimal | str] = []
    executed: list[Step] = []
    resolver = table_resolver or default_table_resolver

    def evaluate(node: Any) -> Decimal | str:
        if isinstance(node, Literal):
            if node.is_reference:
                idx = node.reference_index
                if idx >= len(results):
                    raise UnresolvedReference(
                        f"#{idx} referenced before it exists "
                        f"(only {len(results)} step(s) so far)"
                    )
                return results[idx]
            return _parse_number(node.text)

        # Call
        op = node.name
        if op == "greater":
            args = [_require_numeric(evaluate(a), op) for a in node.args]
            if len(args) != 2:
                raise InvalidArgument(f"greater takes 2 arguments, got {len(args)}")
            return GREATER_TRUE if args[0] > args[1] else GREATER_FALSE

        if op in _TABLE_OPS:
            literals = []
            for a in node.args:
                if not isinstance(a, Literal):
                    raise InvalidArgument(
                        f"{op} takes literal cell references, got a nested call"
                    )
                literals.append(a)
            values = resolver(table, literals) if table else resolver((), literals)
            if not values:
                raise InvalidArgument(f"{op} selected no cells")
            if op == "table_max":
                return max(values)
            if op == "table_min":
                return min(values)
            if op == "table_sum":
                return sum(values, Decimal(0))
            return sum(values, Decimal(0)) / Decimal(len(values))

        if op in _NUMERIC_OPS:
            args = [_require_numeric(evaluate(a), op) for a in node.args]
            if len(args) != 2:
                raise InvalidArgument(f"{op} takes 2 arguments, got {len(args)}")
            return _apply_numeric(op, args)

        raise UnsupportedOperation(f"unknown operation {op!r}")

    for i, step in enumerate(steps):
        value = evaluate(step)
        results.append(value)
        rendered_args = tuple(_render_arg(a) for a in step.args)
        executed.append(Step(i, step.name, rendered_args, value))

    return CalculationTrace(
        program=program,
        steps=tuple(executed),
        result=results[-1],
        table_resolver="default" if table_resolver is None else getattr(
            table_resolver, "__name__", "custom"
        ),
    )


def _render_arg(node: Any) -> str:
    if isinstance(node, Literal):
        return node.text
    inner = ", ".join(_render_arg(a) for a in node.args)
    return f"{node.name}({inner})"


# --------------------------------------------------------------------------
# Comparing answers
# --------------------------------------------------------------------------

def answers_match(
    predicted: Any,
    expected: Any,
    *,
    tolerance: Decimal | str | float = Decimal("1e-3"),
) -> bool:
    """Do two answers agree, allowing for rounding?

    Numbers compare within ``tolerance``. Strings compare case-insensitively
    after stripping. Strings and numbers never match each other, because
    "yes" and 1 are not the same answer.
    """
    if predicted is None or expected is None:
        return False

    pred_str = isinstance(predicted, str) and not _looks_numeric(predicted)
    exp_str = isinstance(expected, str) and not _looks_numeric(expected)
    if pred_str or exp_str:
        return str(predicted).strip().lower() == str(expected).strip().lower()

    try:
        p = _decimal(predicted)
        e = _decimal(expected)
    except InvalidArgument:
        return False

    tol = _decimal(tolerance)
    return abs(p - e) <= tol


def _looks_numeric(value: str) -> bool:
    try:
        _parse_number(value)
    except InvalidArgument:
        return False
    return True
