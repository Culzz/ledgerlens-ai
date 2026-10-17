"""The financial formulas a program can name.

``divide(637, 5)`` is arithmetic and the engine already does it. These are the
things an analyst has a name for, and they are here for two reasons. A program can
say what it means instead of writing the same rearrangement of ``exp`` and
``divide`` every time, and the audit trail can name the quantity, which is the
difference between a reader who can check an answer and a reader who has to trust
it.

Two rules run through all of them.

**The operands are the caller's.** Nothing here goes looking for "revenue" or
"EBITDA" in the report. Each formula computes the relationship it was handed. One
that went and found its own inputs would be guessing on the reader's behalf, and
by the time the guess reached the answer card it would look like a fact.

**A rate comes back as a fraction.** ``cagr(100, 121, 2)`` answers ``0.1``, not
``10``. That is the convention the rest of the engine already uses - FinQA stores a
percentage as a fraction - and a formula that quietly broke it would answer wrong
by exactly a hundred while looking reasonable.

Each one is checked against a hand calculation in ``test_finance.py``, including
the cases where it has to refuse, because a formula with no test is a formula
nobody has run.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from .errors import InvalidArgument

_ONE = Decimal(1)
_ZERO = Decimal(0)


def real_power(base: Decimal, exponent: Decimal) -> Decimal:
    """``base ** exponent`` where the exponent need not be a whole number.

    A whole exponent stays exact, because Decimal can do that. A fractional one has
    no exact decimal form and goes through float - the reason a compound growth rate
    is reported to a few places rather than claimed exact.

    A negative base with a fractional exponent is refused rather than passed to
    float. Float answers that with a complex number, and the failure is worse than
    an error: ``Decimal(str(...))`` on a complex raises ``InvalidOperation``, which
    is not a ``CalculationError``, so it would escape the engine's contract and the
    repair loop that depends on it. The engine's own ``exp`` calls this for that
    reason.
    """
    if exponent == exponent.to_integral_value():
        return base ** int(exponent)
    if base <= 0:
        raise InvalidArgument(f"a fractional power needs a positive base, got {base}")
    return Decimal(str(float(base) ** float(exponent)))


def _cagr(args: Sequence[Decimal]) -> Decimal:
    start, end, periods = args
    if start <= 0:
        raise InvalidArgument(f"cagr needs a positive starting figure, got {start}")
    if end <= 0:
        raise InvalidArgument(f"cagr needs a positive ending figure, got {end}")
    if periods <= 0:
        raise InvalidArgument(f"cagr needs a positive number of periods, got {periods}")
    return real_power(end / start, _ONE / periods) - _ONE


def _present_value(args: Sequence[Decimal]) -> Decimal:
    """The value today of the same cash flow arriving every period.

    A level annuity, not a forecast: one amount, repeated for ``periods``, at
    ``rate``. A growing or uneven forecast goes through ``dcf`` or an explicit sum
    of these, and either way the program says which it meant.
    """
    cash_flow, rate, periods = args
    if periods <= 0:
        raise InvalidArgument(f"present_value needs a positive number of periods, got {periods}")
    if rate == 0:
        # At a zero rate this is just the sum, and the factor below would divide
        # by the rate to find that out.
        return cash_flow * periods
    if rate <= -_ONE:
        raise InvalidArgument(f"a discount rate of -100% or less has no present value, got {rate}")
    factor = (_ONE - real_power(_ONE + rate, -periods)) / rate
    return cash_flow * factor


def _perpetuity(args: Sequence[Decimal]) -> Decimal:
    """A cash flow that carries on for ever, growing at ``growth``.

    The Gordon growth shape, and usually the terminal value of a ``dcf`` rather
    than an answer on its own. It needs the discount rate above the growth rate;
    below that the sum does not converge and there is no number to give.
    """
    cash_flow, rate, growth = args
    if rate <= growth:
        raise InvalidArgument(
            "a perpetuity needs the discount rate above the growth rate, "
            f"got a rate of {rate} and growth of {growth}"
        )
    return cash_flow * (_ONE + growth) / (rate - growth)


def _dcf(args: Sequence[Decimal]) -> Decimal:
    """A discounted cash flow, in the one shape a flat program can write.

    A single cash flow growing at ``growth`` for ``periods`` years, discounted at
    ``rate``, followed by a terminal value that grows at ``terminal_growth`` for
    ever. That is a real DCF and a deliberately narrow one: a forecast whose years
    differ from each other cannot be written here, because a program step holds one
    number and there is nowhere to put a series. Summing ``present_value`` calls is
    how to do that, and the trail will then show each year rather than one figure.

    The two growth rates are separate arguments because that is the assumption a
    DCF actually makes - the years you can see, and the years you cannot. With a
    single rate they are the same, the terminal value continues exactly what the
    forecast stops, and the answer collapses to ``perpetuity`` at every horizon.
    That is true rather than a flaw, it is tested, and it is the reason this
    function has five arguments instead of four.
    """
    cash_flow, rate, growth, terminal_growth, periods = args
    if periods != periods.to_integral_value():
        raise InvalidArgument(f"dcf discounts a whole number of periods, got {periods}")
    if periods <= 0:
        raise InvalidArgument(f"dcf needs a positive number of periods, got {periods}")
    if rate <= terminal_growth:
        raise InvalidArgument(
            "the terminal value needs the discount rate above the terminal growth "
            f"rate, got a rate of {rate} and terminal growth of {terminal_growth}"
        )

    years = int(periods)
    discounted = _ZERO
    for year in range(1, years + 1):
        grown = cash_flow * real_power(_ONE + growth, Decimal(year))
        discounted += grown / real_power(_ONE + rate, Decimal(year))

    final_year = cash_flow * real_power(_ONE + growth, Decimal(years))
    terminal = final_year * (_ONE + terminal_growth) / (rate - terminal_growth)
    return discounted + terminal / real_power(_ONE + rate, Decimal(years))


def _ebitda(args: Sequence[Decimal]) -> Decimal:
    """Earnings before interest, tax, depreciation and amortisation.

    The three add-backs are the amounts as a statement reports them, without their
    sign, because adding an expense back means adding its size. A negative one is
    refused rather than absorbed: a program that passes an interest figure of
    ``-20`` means something - it read a statement that prints expenses negative -
    and quietly flipping it would hide a disagreement about signs instead of
    surfacing it.
    """
    net_income, interest, taxes, depreciation = args
    for label, value in (
        ("interest", interest),
        ("tax", taxes),
        ("depreciation and amortisation", depreciation),
    ):
        if value < 0:
            raise InvalidArgument(
                f"ebitda adds back an expense, so {label} has to be the amount as "
                f"reported without its sign - got {value}. Pass {abs(value)}, or "
                "write the sum with add and subtract if the statement's own signs "
                "are the ones you want."
            )
    return net_income + interest + taxes + depreciation


def _quick_ratio(args: Sequence[Decimal]) -> Decimal:
    """Current assets less inventory, over current liabilities.

    The subtraction is the whole point of the ratio: inventory is the part of
    current assets that cannot be turned into cash at short notice, so naming the
    ratio is clearer than writing the difference and hoping a reader recognises it.
    """
    current_assets, inventory, current_liabilities = args
    if current_liabilities == 0:
        raise InvalidArgument("quick_ratio divides by current liabilities, which are zero")
    return (current_assets - inventory) / current_liabilities


def _debt_to_equity(args: Sequence[Decimal]) -> Decimal:
    """Total debt over total equity.

    A name for a division, which sounds like padding until the direction is the
    question: debt over equity and equity over debt are both written as a ratio by
    someone, and only one of them is what a reader will assume this means.
    """
    total_debt, total_equity = args
    if total_equity == 0:
        raise InvalidArgument("debt_to_equity divides by equity, which is zero")
    return total_debt / total_equity


@dataclass(frozen=True)
class Primitive:
    """A named formula and the arguments its definition takes.

    The argument names are not decoration: they carry the arity, and they are what
    the error says when a program gets the count wrong. "cagr takes 3 arguments
    (start, end, periods), got 2" is a message a model can act on; "invalid
    argument" is not.
    """

    name: str
    arguments: tuple[str, ...]
    compute: Callable[[Sequence[Decimal]], Decimal]

    @property
    def arity(self) -> int:
        return len(self.arguments)


FINANCE_PRIMITIVES: dict[str, Primitive] = {
    primitive.name: primitive
    for primitive in (
        Primitive("cagr", ("start", "end", "periods"), _cagr),
        Primitive("present_value", ("cash_flow", "rate", "periods"), _present_value),
        Primitive("perpetuity", ("cash_flow", "rate", "growth"), _perpetuity),
        Primitive("dcf", ("cash_flow", "rate", "growth", "terminal_growth", "periods"), _dcf),
        Primitive("ebitda", ("net_income", "interest", "taxes", "depreciation"), _ebitda),
        Primitive(
            "quick_ratio",
            ("current_assets", "inventory", "current_liabilities"),
            _quick_ratio,
        ),
        Primitive("debt_to_equity", ("total_debt", "total_equity"), _debt_to_equity),
    )
}


def apply_finance(name: str, args: Sequence[Decimal]) -> Decimal:
    """Run a named formula, checking the argument count before anything else."""
    primitive = FINANCE_PRIMITIVES[name]
    if len(args) != primitive.arity:
        raise InvalidArgument(
            f"{name} takes {primitive.arity} arguments "
            f"({', '.join(primitive.arguments)}), got {len(args)}"
        )
    return primitive.compute(args)
