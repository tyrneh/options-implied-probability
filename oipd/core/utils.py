import numpy as np

__all__ = [
    "resolve_risk_free_rate",
]


def resolve_risk_free_rate(
    risk_free_rate: float, risk_free_rate_mode: str, time_years: float
) -> float:
    """Return the continuous-compounded risk-free rate for pricing.

    Args:
        risk_free_rate: Input risk-free rate quoted by the user.
        risk_free_rate_mode: Quoting convention, either ``"annualized"`` for
            simple annualized rates or ``"continuous"`` for continuous rates.
        time_years: Time to expiry in years for the instrument being priced.

    Returns:
        Continuous-compounded rate consistent with ``exp(-r * T)``.

    Raises:
        ValueError: If ``risk_free_rate_mode`` is unrecognized.
    """
    rate = float(risk_free_rate)
    if risk_free_rate_mode == "continuous":
        return rate
    if risk_free_rate_mode == "annualized":
        if time_years <= 0:
            return rate
        return float(np.log1p(rate * time_years) / time_years)
    raise ValueError(
        "risk_free_rate_mode must be 'annualized' or 'continuous', "
        f"got {risk_free_rate_mode!r}"
    )
