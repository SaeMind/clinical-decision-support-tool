"""Health-economic impact model for ED triage operations."""

from __future__ import annotations


def estimate_lwbs_revenue_gain(
    annual_ed_visits: int,
    baseline_lwbs_rate: float,
    target_lwbs_rate: float,
    net_revenue_per_completed_visit: float,
) -> float:
    """Estimate annual revenue retained from lower LWBS rate.

    Args:
        annual_ed_visits: Annual emergency department visit volume.
        baseline_lwbs_rate: Baseline left-without-being-seen rate.
        target_lwbs_rate: Target left-without-being-seen rate.
        net_revenue_per_completed_visit: Net revenue per completed ED visit.

    Returns:
        Estimated annual retained revenue.
    """
    avoided_lwbs_visits = annual_ed_visits * max(baseline_lwbs_rate - target_lwbs_rate, 0)
    return avoided_lwbs_visits * net_revenue_per_completed_visit


def estimate_triage_labor_savings(
    annual_ed_visits: int,
    baseline_triage_minutes: float,
    target_triage_minutes: float,
    loaded_nurse_hourly_rate: float,
) -> float:
    """Estimate annual labor capacity value from reduced triage time.

    Args:
        annual_ed_visits: Annual emergency department visit volume.
        baseline_triage_minutes: Baseline triage duration in minutes.
        target_triage_minutes: Target triage duration in minutes.
        loaded_nurse_hourly_rate: Fully loaded hourly nurse labor cost.

    Returns:
        Estimated annual labor capacity value.
    """
    minutes_saved = annual_ed_visits * max(baseline_triage_minutes - target_triage_minutes, 0)
    return (minutes_saved / 60) * loaded_nurse_hourly_rate


def economic_impact(settings: dict) -> dict[str, float]:
    """Calculate a basic ED economic impact model.

    Args:
        settings: Loaded settings dictionary containing economics assumptions.

    Returns:
        Impact dictionary.
    """
    economics = settings["economics"]
    lwbs_gain = estimate_lwbs_revenue_gain(
        annual_ed_visits=economics["annual_ed_visits"],
        baseline_lwbs_rate=economics["baseline_lwbs_rate"],
        target_lwbs_rate=economics["target_lwbs_rate"],
        net_revenue_per_completed_visit=economics["net_revenue_per_completed_visit"],
    )
    labor_value = estimate_triage_labor_savings(
        annual_ed_visits=economics["annual_ed_visits"],
        baseline_triage_minutes=economics["baseline_triage_minutes"],
        target_triage_minutes=economics["target_triage_minutes"],
        loaded_nurse_hourly_rate=economics["loaded_nurse_hourly_rate"],
    )
    return {
        "estimated_lwbs_revenue_gain": round(lwbs_gain, 2),
        "estimated_labor_capacity_value": round(labor_value, 2),
        "estimated_total_annual_value": round(lwbs_gain + labor_value, 2),
    }
