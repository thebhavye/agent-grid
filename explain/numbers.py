"""Shared numeric guard for optional explanation renderers.

Numbers are compared at six decimal places, including numbers in identifiers
and timestamps. This is a numeric consistency check, not a truth verifier.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import json
import re

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numbers_supported(text, facts):
    """Reject non-text, empty output and numeric claims absent from facts."""
    if not isinstance(text, str) or not text.strip():
        return False

    def normalized(value):
        with localcontext() as ctx:
            ctx.prec = 40
            return Decimal(value).quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN)

    try:
        allowed = {normalized(n) for n in _NUMBER.findall(json.dumps(facts, default=str))}
        return all(normalized(n) in allowed for n in _NUMBER.findall(text))
    except (InvalidOperation, ValueError, OverflowError):
        return False
