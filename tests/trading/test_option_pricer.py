"""Tests for Black-Scholes put option pricer."""

from src.trading.option_pricer import price_put


class TestPricePut:
    def test_atm_put_positive(self):
        """ATM put with time value should be positive."""
        p = price_put(spot=50000, strike=50000, days_to_expiry=30)
        assert p > 0

    def test_deep_itm_put(self):
        """Deep ITM put (strike >> spot) should be roughly intrinsic."""
        p = price_put(spot=40000, strike=50000, days_to_expiry=30)
        intrinsic = 50000 - 40000
        assert p >= intrinsic * 0.9  # at least ~90% of intrinsic

    def test_deep_otm_put(self):
        """Deep OTM put (strike << spot) should be near zero."""
        p = price_put(spot=60000, strike=50000, days_to_expiry=30)
        assert p < 500  # very small

    def test_expired_itm(self):
        """Expired ITM put = intrinsic value."""
        p = price_put(spot=48000, strike=50000, days_to_expiry=0)
        assert p == 2000.0

    def test_expired_otm(self):
        """Expired OTM put = 0."""
        p = price_put(spot=52000, strike=50000, days_to_expiry=0)
        assert p == 0.0

    def test_negative_days_treated_as_expired(self):
        p = price_put(spot=48000, strike=50000, days_to_expiry=-5)
        assert p == 2000.0

    def test_higher_vol_higher_price(self):
        """Higher volatility -> higher option price."""
        low = price_put(spot=50000, strike=50000, days_to_expiry=30, vol=0.15)
        high = price_put(spot=50000, strike=50000, days_to_expiry=30, vol=0.40)
        assert high > low

    def test_longer_expiry_higher_price(self):
        """More time to expiry -> higher option price."""
        short = price_put(spot=50000, strike=50000, days_to_expiry=7)
        long = price_put(spot=50000, strike=50000, days_to_expiry=60)
        assert long > short

    def test_non_negative(self):
        """Put price should never be negative."""
        p = price_put(spot=100000, strike=50000, days_to_expiry=1)
        assert p >= 0.0
