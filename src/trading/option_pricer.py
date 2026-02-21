"""Black-Scholes put option pricing for mock broker."""

import math
from typing import Optional


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution (approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def price_put(
    spot: float,
    strike: float,
    days_to_expiry: float,
    vol: float = 0.25,
    r: float = 0.035,
) -> float:
    """Calculate European put option price via Black-Scholes.

    Args:
        spot: Current underlying price.
        strike: Option strike price.
        days_to_expiry: Days until expiration.
        vol: Annualized volatility (default 0.25 = 25%).
        r: Risk-free rate (default 0.035 = 3.5%).

    Returns:
        Put option price (same currency unit as spot/strike).
    """
    if days_to_expiry <= 0:
        # Expired: intrinsic value only
        return max(strike - spot, 0.0)

    T = days_to_expiry / 365.0
    sqrt_T = math.sqrt(T)

    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    put_price = (
        strike * math.exp(-r * T) * _norm_cdf(-d2)
        - spot * _norm_cdf(-d1)
    )

    return max(put_price, 0.0)
