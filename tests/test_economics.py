"""Tests for health-economic calculations."""

from src.economics import estimate_lwbs_revenue_gain, estimate_triage_labor_savings


def test_estimate_lwbs_revenue_gain():
    """LWBS revenue gain should reflect avoided incomplete visits."""
    result = estimate_lwbs_revenue_gain(100000, 0.05, 0.04, 250)
    assert round(result, 2) == 250000


def test_estimate_triage_labor_savings():
    """Labor savings should convert saved minutes to hourly labor value."""
    result = estimate_triage_labor_savings(60, 10, 5, 60)
    assert result == 300
