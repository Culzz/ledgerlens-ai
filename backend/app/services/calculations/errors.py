"""What the calculation service raises when a program cannot produce an answer.

These live apart from the engine so that a module which only needs to *raise* one
- the financial primitives, for instance - does not have to import the thing that
executes programs. The engine imports them from here and re-exports them, so
``from app.services.calculations import InvalidArgument`` is unchanged.

Every one of them means the same kind of thing: the program is not going to give
an answer, and the message is what the caller gets to work with. On the model path
that message goes back to the model verbatim, which is why they name the specific
problem rather than saying "invalid program".
"""


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
