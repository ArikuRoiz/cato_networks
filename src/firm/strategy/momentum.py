"""Momentum signal computation — pure function, no IO.

N-day price-return momentum over a window of ``Bar`` objects.
"""

from __future__ import annotations

from firm.domain.entities import Bar


def compute_momentum(bars: list[Bar], n_days: int = 5) -> float:
    """Return the N-day simple price return as a float in (-inf, +inf).

    Uses ``bars[-1].close`` as the current price and ``bars[-n_days].close``
    as the reference price *n_days* back.

    Parameters
    ----------
    bars:
        Chronologically ordered list of ``Bar`` objects.  Must contain at
        least ``n_days + 1`` entries.
    n_days:
        Lookback length in bars.  Defaults to 5 (one trading week).

    Returns
    -------
    float
        ``(close[-1] - close[-n_days]) / close[-n_days]`` — positive for
        upward momentum, negative for downward.

    Raises
    ------
    ValueError
        When ``len(bars) < n_days + 1``, i.e. not enough history to compute
        the return over *n_days*.
    """
    if len(bars) < n_days + 1:
        raise ValueError(
            f"compute_momentum requires at least {n_days + 1} bars "
            f"(n_days={n_days}); got {len(bars)}."
        )
    reference_close = float(bars[-n_days].close)
    if reference_close == 0.0:
        return 0.0
    return float((bars[-1].close - bars[-n_days].close) / bars[-n_days].close)
