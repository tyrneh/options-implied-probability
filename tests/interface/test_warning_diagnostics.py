"""Interface tests for warning diagnostics containers."""

import warnings

import numpy as np
import pandas as pd
import pytest


def _assert_empty_warning_diagnostics(interface_object):
    """Assert that an interface object exposes empty warning diagnostics.

    Args:
        interface_object: Public OIPD interface object with warning diagnostics.
    """
    diagnostics = interface_object.warning_diagnostics

    assert diagnostics.events == []
    assert diagnostics.summary.total_events == 0
    assert diagnostics.summary.by_category == {}
    assert diagnostics.summary.by_event_type == {}
    assert diagnostics.summary.by_severity == {}
    assert diagnostics.summary.worst_severity is None


def _assert_json_like_detail_value(value):
    """Assert that a warning detail value is JSON-like and summarized.

    Args:
        value: Detail value to inspect.
    """
    assert not isinstance(value, (np.ndarray, pd.DataFrame, pd.Series))
    if isinstance(value, dict):
        for nested_value in value.values():
            _assert_json_like_detail_value(nested_value)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for nested_value in value:
            _assert_json_like_detail_value(nested_value)
        return
    assert value is None or isinstance(value, (str, int, float, bool))


class _FakeDiagnostics:
    """Small diagnostics object for interface warning tests.

    Args:
        min_g: Optional SVI butterfly diagnostic value.
        butterfly_warning: Optional SVI butterfly warning value.
    """

    def __init__(
        self,
        *,
        min_g: float | None = None,
        butterfly_warning: float | None = None,
    ) -> None:
        """Initialize fake SVI diagnostics.

        Args:
            min_g: Optional minimum butterfly diagnostic value.
            butterfly_warning: Optional warning-worthy butterfly diagnostic value.
        """
        self.min_g = min_g
        self.butterfly_warning = butterfly_warning
        self.status = "warning" if butterfly_warning is not None else "success"


def _fake_vol_curve(strikes):
    """Return constant implied volatility values for fake fit tests.

    Args:
        strikes: Strike values supplied by the public interface.

    Returns:
        np.ndarray: Constant implied-volatility values.
    """
    strike_array = np.asarray(strikes, dtype=float)
    return np.full_like(strike_array, 0.25, dtype=float)


def _fit_metadata(
    *,
    mid_price_filled=0,
    staleness_report=None,
    diagnostics=None,
    expiry="2025-02-01",
):
    """Build minimal volatility fit metadata for warning diagnostics tests.

    Args:
        mid_price_filled: Missing mid price fill count.
        staleness_report: Optional stale quote filtering report.
        diagnostics: Optional fake SVI diagnostics object.
        expiry: Expiry value for metadata.

    Returns:
        dict: Minimal fit metadata.
    """
    return {
        "at_money_vol": 0.25,
        "forward_price": 100.0,
        "pricing_engine": "black76",
        "method": "svi",
        "mid_price_filled": mid_price_filled,
        "staleness_report": staleness_report or {},
        "diagnostics": diagnostics or _FakeDiagnostics(min_g=0.01),
        "expiry": pd.Timestamp(expiry),
        "time_to_expiry_years": 31.0 / 365.0,
    }


def _cdf_repair_metadata(
    *,
    expiry="2025-02-01",
    severity="material",
    repair_applied=True,
    time_to_expiry_years=None,
    interpolated=None,
    interpolated_from_expiries=None,
):
    """Build minimal probability metadata for CDF repair diagnostics tests.

    Args:
        expiry: Expiry value to include in diagnostic details.
        severity: CDF monotonicity severity label.
        repair_applied: Whether the CDF repair flag should be set.
        time_to_expiry_years: Optional year-fraction maturity lineage.
        interpolated: Optional interpolation lineage flag.
        interpolated_from_expiries: Optional source-expiry lineage.

    Returns:
        dict: Minimal materialization metadata with CDF diagnostics.
    """
    metadata = {
        "expiry": pd.Timestamp(expiry),
        "cdf_violation_policy": "warn",
        "cdf_monotonicity_repair_applied": repair_applied,
        "cdf_monotonicity_repair_tolerance": 5e-6,
        "cdf_total_negative_variation_tolerance": 1e-4,
        "cdf_monotonicity_severity": severity,
        "raw_cdf_negative_step_count": 1,
        "raw_cdf_max_negative_step": -5e-5,
        "raw_cdf_total_negative_variation": 5e-5,
        "raw_cdf_worst_step_strike": 101.0,
    }
    if time_to_expiry_years is not None:
        metadata["time_to_expiry_years"] = time_to_expiry_years
    if interpolated is not None:
        metadata["interpolated"] = interpolated
    if interpolated_from_expiries is not None:
        metadata["interpolated_from_expiries"] = interpolated_from_expiries
    return metadata


def _fake_probability_curve(resolved_market, metadata):
    """Build an array-backed ProbCurve for surface diagnostics tests.

    Args:
        resolved_market: Resolved market snapshot for the fake curve.
        metadata: Probability metadata to attach to the fake snapshot.

    Returns:
        ProbCurve: Materialized probability curve with small valid arrays.
    """
    from oipd.interface.probability import ProbCurve

    prices = np.array([90.0, 100.0, 110.0], dtype=float)
    pdf_values = np.array([0.01, 0.02, 0.01], dtype=float)
    cdf_values = np.array([0.1, 0.5, 0.9], dtype=float)
    return ProbCurve.from_arrays(
        resolved_market=resolved_market,
        metadata=metadata,
        prices=prices,
        pdf_values=pdf_values,
        cdf_values=cdf_values,
    )


def _fake_interpolator(*args, **kwargs):
    """Return a minimal interpolator for surface fit tests.

    Args:
        *args: Ignored positional arguments.
        **kwargs: Ignored keyword arguments.

    Returns:
        object: Object placeholder accepted by the public surface interface.
    """
    return object()


def _fake_surface_model(expiry_reports, market_inputs):
    """Build a minimal fitted surface with warning reports.

    Args:
        expiry_reports: Warning reports to attach to the surface model.
        market_inputs: Market inputs for resolving slice market snapshots.

    Returns:
        DiscreteSurface: Minimal fitted surface model.
    """
    from oipd.market_inputs import resolve_market
    from oipd.pipelines.vol_surface.models import DiscreteSurface

    expiries = [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-04-01")]
    slices = {
        expiry: {
            "curve": _fake_vol_curve,
            "metadata": _fit_metadata(expiry=expiry),
        }
        for expiry in expiries
    }
    resolved_markets = {expiry: resolve_market(market_inputs) for expiry in expiries}
    slice_chains = {expiry: pd.DataFrame({"expiry": [expiry]}) for expiry in expiries}
    return DiscreteSurface(
        slices,
        resolved_markets,
        slice_chains,
        fit_warning_reports=expiry_reports,
    )


def test_new_vol_curve_has_empty_warning_diagnostics():
    """New VolCurve objects expose empty diagnostics before fitting."""
    from oipd import VolCurve

    _assert_empty_warning_diagnostics(VolCurve())


def test_new_vol_surface_has_empty_warning_diagnostics():
    """New VolSurface objects expose empty diagnostics before fitting."""
    from oipd import VolSurface

    _assert_empty_warning_diagnostics(VolSurface())


def test_lazy_prob_curve_has_empty_warning_diagnostics_before_materialization(
    fitted_vol_curve,
):
    """Lazy ProbCurve objects expose empty diagnostics before probability access."""
    prob_curve = fitted_vol_curve.implied_distribution()

    _assert_empty_warning_diagnostics(prob_curve)


def test_lazy_prob_surface_has_empty_warning_diagnostics_before_materialization(
    fitted_vol_surface,
):
    """Lazy ProbSurface objects expose empty diagnostics before probability access."""
    prob_surface = fitted_vol_surface.implied_distribution()

    _assert_empty_warning_diagnostics(prob_surface)


def test_lazy_prob_surface_slice_has_empty_warning_diagnostics_before_materialization(
    fitted_vol_surface,
):
    """Lazy ProbSurface slices expose empty diagnostics before probability access."""
    prob_surface = fitted_vol_surface.implied_distribution()
    prob_curve = prob_surface.slice(prob_surface.expiries[0])

    _assert_empty_warning_diagnostics(prob_curve)


def test_prob_curve_from_arrays_has_empty_warning_diagnostics(fitted_vol_curve):
    """Array-backed ProbCurve objects expose empty diagnostics at construction."""
    from oipd.interface.probability import ProbCurve

    prob_curve = ProbCurve.from_arrays(
        resolved_market=fitted_vol_curve.resolved_market,
        metadata={"expiry": "2025-02-01"},
        prices=np.array([90.0, 100.0, 110.0]),
        pdf_values=np.array([0.01, 0.02, 0.01]),
        cdf_values=np.array([0.0, 0.5, 1.0]),
    )

    _assert_empty_warning_diagnostics(prob_curve)


def test_fitted_vol_surface_pillar_slice_has_empty_warning_diagnostics(
    fitted_vol_surface,
):
    """Exact fitted VolSurface slices expose empty diagnostics."""
    pillar_expiry = fitted_vol_surface.expiries[0]
    vol_curve = fitted_vol_surface.slice(pillar_expiry)

    _assert_empty_warning_diagnostics(vol_curve)


def test_fitted_vol_surface_interpolated_slice_has_empty_warning_diagnostics(
    fitted_vol_surface,
):
    """Interpolated VolSurface slices expose empty diagnostics."""
    first_expiry, second_expiry = fitted_vol_surface.expiries[:2]
    interpolated_expiry = first_expiry + (second_expiry - first_expiry) / 2
    vol_curve = fitted_vol_surface.slice(interpolated_expiry)

    _assert_empty_warning_diagnostics(vol_curve)


def test_prob_curve_materialization_records_cdf_repair_and_summary_warning(
    fitted_vol_curve,
    monkeypatch,
):
    """ProbCurve materialization records CDF repair and emits one summary warning."""
    import oipd.interface.probability as probability_interface
    from oipd.pipelines.probability.models import DistributionSnapshot
    from oipd.warnings import ModelRiskWarning

    def fake_materializer(*args, **kwargs):
        """Return a repaired CDF snapshot without running numerical pipelines."""
        return DistributionSnapshot(
            prices=np.array([90.0, 100.0, 110.0]),
            pdf_values=np.array([0.01, 0.02, 0.01]),
            cdf_values=np.array([0.1, 0.5, 0.9]),
            metadata=_cdf_repair_metadata(severity="material"),
        )

    monkeypatch.setattr(
        probability_interface,
        "materialize_distribution_from_definition",
        fake_materializer,
    )

    prob_curve = fitted_vol_curve.implied_distribution()
    with pytest.warns(ModelRiskWarning) as recorded_warnings:
        _ = prob_curve.prices

    assert len(recorded_warnings) == 1
    assert ".warning_diagnostics.events" in str(recorded_warnings[0].message)
    events = prob_curve.warning_diagnostics.events
    assert len(events) == 1
    assert events[0].event_type == "cdf_repair"
    assert events[0].category == "model_risk"
    assert events[0].severity == "warning"
    assert events[0].details["cdf_monotonicity_severity"] == "material"
    for detail_value in events[0].details.values():
        _assert_json_like_detail_value(detail_value)


def test_prob_curve_cdf_repair_uses_severe_metadata_severity(
    fitted_vol_curve,
    monkeypatch,
):
    """ProbCurve maps severe CDF repair metadata to severe event severity."""
    import oipd.interface.probability as probability_interface
    from oipd.pipelines.probability.models import DistributionSnapshot
    from oipd.warnings import ModelRiskWarning

    def fake_materializer(*args, **kwargs):
        """Return a snapshot with severe CDF repair metadata."""
        return DistributionSnapshot(
            prices=np.array([90.0, 100.0, 110.0]),
            pdf_values=np.array([0.01, 0.02, 0.01]),
            cdf_values=np.array([0.1, 0.5, 0.9]),
            metadata=_cdf_repair_metadata(severity="severe"),
        )

    monkeypatch.setattr(
        probability_interface,
        "materialize_distribution_from_definition",
        fake_materializer,
    )

    prob_curve = fitted_vol_curve.implied_distribution()
    with pytest.warns(ModelRiskWarning):
        _ = prob_curve.cdf_values

    events = prob_curve.warning_diagnostics.events
    assert len(events) == 1
    assert events[0].event_type == "cdf_repair"
    assert events[0].severity == "severe"
    assert prob_curve.warning_diagnostics.summary.worst_severity == "severe"


def test_prob_curve_from_chain_retains_hidden_vol_fit_warning_diagnostics(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """ProbCurve.from_chain keeps diagnostics from its hidden VolCurve.fit."""
    import oipd.interface.probability as probability_interface
    from oipd import ProbCurve
    from oipd.pipelines.probability.models import DistributionSnapshot
    from oipd.warnings import DataQualityWarning, ModelRiskWarning

    fallback_chain = single_expiry_chain.copy()
    fallback_chain["bid"] = np.nan
    fallback_chain["ask"] = np.nan

    with pytest.warns(DataQualityWarning):
        prob_curve = ProbCurve.from_chain(fallback_chain, market_inputs)

    inherited_events = prob_curve.warning_diagnostics.events
    assert "VolCurve.fit" in {event.scope for event in inherited_events}
    assert "price_fallback" in {event.event_type for event in inherited_events}

    def fake_materializer(*args, **kwargs):
        """Return a repaired CDF snapshot for later probability materialization."""
        return DistributionSnapshot(
            prices=np.array([90.0, 100.0, 110.0]),
            pdf_values=np.array([0.01, 0.02, 0.01]),
            cdf_values=np.array([0.1, 0.5, 0.9]),
            metadata=_cdf_repair_metadata(),
        )

    monkeypatch.setattr(
        probability_interface,
        "materialize_distribution_from_definition",
        fake_materializer,
    )

    with pytest.warns(ModelRiskWarning):
        _ = prob_curve.prices

    events = prob_curve.warning_diagnostics.events
    assert "VolCurve.fit" in {event.scope for event in events}
    assert "ProbCurve.materialize" in {event.scope for event in events}
    assert {"price_fallback", "cdf_repair"}.issubset(
        {event.event_type for event in events}
    )


def test_prob_curve_warning_as_error_does_not_commit_snapshot(
    fitted_vol_curve,
    monkeypatch,
):
    """ProbCurve retries materialization after summary warning-as-error."""
    import oipd.interface.probability as probability_interface
    from oipd.pipelines.probability.models import DistributionSnapshot
    from oipd.warnings import ModelRiskWarning

    materialization_count = 0

    def fake_materializer(*args, **kwargs):
        """Return a repaired snapshot and count materialization attempts."""
        nonlocal materialization_count
        materialization_count += 1
        return DistributionSnapshot(
            prices=np.array([90.0, 100.0, 110.0]),
            pdf_values=np.array([0.01, 0.02, 0.01]),
            cdf_values=np.array([0.1, 0.5, 0.9]),
            metadata=_cdf_repair_metadata(),
        )

    monkeypatch.setattr(
        probability_interface,
        "materialize_distribution_from_definition",
        fake_materializer,
    )

    prob_curve = fitted_vol_curve.implied_distribution()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelRiskWarning)
        with pytest.raises(ModelRiskWarning):
            _ = prob_curve.prices

    assert prob_curve._native_snapshot is None
    assert prob_curve.warning_diagnostics.events == []
    assert materialization_count == 1

    with pytest.warns(ModelRiskWarning):
        _ = prob_curve.prices
    assert materialization_count == 2
    assert prob_curve._native_snapshot is not None


def test_prob_surface_from_chain_retains_hidden_vol_fit_warning_diagnostics(
    multi_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """ProbSurface.from_chain keeps diagnostics from its hidden VolSurface.fit."""
    import oipd.interface.probability as probability_interface
    from oipd import ProbSurface
    from oipd.pipelines.probability.models import DistributionSnapshot
    from oipd.warnings import DataQualityWarning, ModelRiskWarning

    fallback_chain = multi_expiry_chain.copy()
    fallback_chain["bid"] = np.nan
    fallback_chain["ask"] = np.nan

    with pytest.warns(DataQualityWarning):
        prob_surface = ProbSurface.from_chain(fallback_chain, market_inputs)

    inherited_events = prob_surface.warning_diagnostics.events
    assert "VolSurface.fit" in {event.scope for event in inherited_events}
    assert "price_fallback" in {event.event_type for event in inherited_events}

    def fake_materializer(definition, *args, **kwargs):
        """Return a repaired CDF snapshot for a later surface-owned query."""
        expiry = definition.vol_metadata.get("expiry", prob_surface.expiries[0])
        return DistributionSnapshot(
            prices=np.array([90.0, 100.0, 110.0]),
            pdf_values=np.array([0.01, 0.02, 0.01]),
            cdf_values=np.array([0.1, 0.5, 0.9]),
            metadata=_cdf_repair_metadata(expiry=expiry),
        )

    monkeypatch.setattr(
        probability_interface,
        "materialize_distribution_from_definition",
        fake_materializer,
    )

    with pytest.warns(ModelRiskWarning):
        _ = prob_surface.quantile(0.5, t=prob_surface.expiries[0])

    events = prob_surface.warning_diagnostics.events
    assert "VolSurface.fit" in {event.scope for event in events}
    assert "ProbSurface.query" in {event.scope for event in events}
    assert {"price_fallback", "cdf_repair"}.issubset(
        {event.event_type for event in events}
    )


def test_prob_surface_plot_fan_aggregates_cdf_repair_summaries(
    fitted_vol_surface,
    monkeypatch,
):
    """ProbSurface.plot_fan emits one model-risk summary for repaired slices."""
    import oipd.interface.probability as probability_interface
    from oipd.warnings import ModelRiskWarning

    prob_surface = fitted_vol_surface.implied_distribution()
    expiries = prob_surface.expiries[:2]
    for expiry in expiries:
        cache_key = int(pd.Timestamp(expiry).value)
        prob_surface._internal_curve_cache[cache_key] = _fake_probability_curve(
            prob_surface.slice(expiry).resolved_market,
            _cdf_repair_metadata(expiry=expiry),
        )
    unrelated_expiry = pd.Timestamp("2025-06-01")
    prob_surface._internal_curve_cache[int(unrelated_expiry.value)] = (
        _fake_probability_curve(
            prob_surface.slice(expiries[0]).resolved_market,
            _cdf_repair_metadata(expiry=unrelated_expiry),
        )
    )

    def fake_build_fan_quantile_summary_frame(surface):
        """Return a minimal fan frame while leaving preloaded cache in place."""
        return pd.DataFrame({"expiry": list(expiries), "is_pillar": [True, True]})

    monkeypatch.setattr(
        probability_interface,
        "build_fan_quantile_summary_frame",
        fake_build_fan_quantile_summary_frame,
    )
    monkeypatch.setattr(
        probability_interface,
        "plot_probability_summary",
        lambda *args, **kwargs: "figure",
    )

    with pytest.warns(ModelRiskWarning) as recorded_warnings:
        result = prob_surface.plot_fan()

    assert result == "figure"
    assert len(recorded_warnings) == 1
    assert ".warning_diagnostics.events" in str(recorded_warnings[0].message)
    events = prob_surface.warning_diagnostics.events
    assert [event.event_type for event in events] == ["cdf_repair", "cdf_repair"]
    assert {event.details["is_pillar"] for event in events} == {True}
    assert {event.details["expiry"] for event in events} == {
        pd.Timestamp(expiry).isoformat() for expiry in expiries
    }


def test_prob_surface_plot_fan_records_skipped_expiry_workflow_warning(
    fitted_vol_surface,
    monkeypatch,
):
    """ProbSurface.plot_fan records fan skip reports as workflow diagnostics."""
    import oipd.interface.probability as probability_interface
    from oipd.warnings import WorkflowWarning

    prob_surface = fitted_vol_surface.implied_distribution()
    skipped_expiry = pd.Timestamp(prob_surface.expiries[0]) + pd.Timedelta(days=1)
    summary_frame = pd.DataFrame(
        {"expiry": [prob_surface.expiries[0]], "is_pillar": [True]}
    )
    summary_frame.attrs["fan_skip_report"] = {
        "skipped_count": 1,
        "reason_summary": "1 x CalculationError: synthetic CDF failure",
        "skipped_expiries": [skipped_expiry.isoformat()],
    }

    monkeypatch.setattr(
        probability_interface,
        "build_fan_quantile_summary_frame",
        lambda surface: summary_frame,
    )
    monkeypatch.setattr(
        probability_interface,
        "plot_probability_summary",
        lambda *args, **kwargs: "figure",
    )

    with pytest.warns(WorkflowWarning) as recorded_warnings:
        result = prob_surface.plot_fan()

    assert result == "figure"
    assert len(recorded_warnings) == 1
    events = prob_surface.warning_diagnostics.events
    assert len(events) == 1
    assert events[0].event_type == "skipped_expiry"
    assert events[0].details["skipped_count"] == 1
    assert skipped_expiry.isoformat() in events[0].details["skipped_expiries"]


def test_prob_surface_plot_fan_propagates_strict_cdf_failure(
    fitted_vol_surface,
    monkeypatch,
):
    """Strict CDF materialization failures are not downgraded to fan skips."""
    from oipd.core.errors import CalculationError, InvalidInputError
    from oipd.interface.probability import ProbSurface

    prob_surface = ProbSurface(
        vol_surface=fitted_vol_surface,
        cdf_violation_policy="raise",
    )
    first_expiry = min(prob_surface.expiries)
    strict_failure_expiry = first_expiry + pd.Timedelta(days=1)
    valid_curve = _fake_probability_curve(
        prob_surface.slice(first_expiry).resolved_market,
        {"expiry": first_expiry},
    )

    def fake_internal_slice(expiry):
        """Return one valid slice, then raise a strict CDF materialization error."""
        if pd.Timestamp(expiry) == strict_failure_expiry:
            try:
                raise InvalidInputError("strict direct CDF monotonicity violation")
            except InvalidInputError as exc:
                raise CalculationError("Failed to compute CDF") from exc
        return valid_curve

    monkeypatch.setattr(prob_surface, "_internal_slice", fake_internal_slice)

    with pytest.raises(CalculationError, match="Failed to compute CDF"):
        prob_surface.plot_fan()

    assert prob_surface.warning_diagnostics.events == []


def test_prob_surface_density_results_replaces_stale_cdf_repair_events(
    fitted_vol_surface,
    monkeypatch,
):
    """Repeated surface exports replace stale CDF repair diagnostics."""
    import oipd.interface.probability as probability_interface
    from oipd.warnings import ModelRiskWarning

    prob_surface = fitted_vol_surface.implied_distribution()
    expiry = prob_surface.expiries[0]
    cache_key = int(pd.Timestamp(expiry).value)
    resolved_market = prob_surface.slice(expiry).resolved_market
    unrelated_expiry = pd.Timestamp("2025-06-01")
    prob_surface._internal_curve_cache[int(unrelated_expiry.value)] = (
        _fake_probability_curve(
            resolved_market,
            _cdf_repair_metadata(expiry=unrelated_expiry),
        )
    )
    materialization_count = 0

    def fake_build_surface_density_results_frame(surface, *args, **kwargs):
        """Populate cache with repaired metadata once, then clean metadata."""
        nonlocal materialization_count
        materialization_count += 1
        if materialization_count == 1:
            metadata = _cdf_repair_metadata(expiry=expiry)
        else:
            metadata = _cdf_repair_metadata(expiry=expiry, repair_applied=False)
        surface._internal_curve_cache[cache_key] = _fake_probability_curve(
            resolved_market,
            metadata,
        )
        return pd.DataFrame(
            {"expiry": [expiry], "price": [100.0], "pdf": [0.1], "cdf": [0.5]}
        )

    monkeypatch.setattr(
        probability_interface,
        "build_surface_density_results_frame",
        fake_build_surface_density_results_frame,
    )

    with pytest.warns(ModelRiskWarning):
        prob_surface.density_results()
    events = prob_surface.warning_diagnostics.events
    assert [event.event_type for event in events] == ["cdf_repair"]
    assert events[0].details["expiry"] == pd.Timestamp(expiry).isoformat()

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        prob_surface.density_results()

    assert recorded_warnings == []
    assert prob_surface.warning_diagnostics.events == []


def test_prob_surface_query_replaces_stale_cdf_repair_events(
    fitted_vol_surface,
):
    """Repeated surface queries replace stale CDF repair diagnostics."""
    from oipd.warnings import ModelRiskWarning

    prob_surface = fitted_vol_surface.implied_distribution()
    expiry = prob_surface.expiries[0]
    cache_key = int(pd.Timestamp(expiry).value)
    resolved_market = prob_surface.slice(expiry).resolved_market
    prob_surface._internal_curve_cache[cache_key] = _fake_probability_curve(
        resolved_market,
        _cdf_repair_metadata(expiry=expiry),
    )

    with pytest.warns(ModelRiskWarning):
        prob_surface.pdf(100.0, t=expiry)
    assert [event.event_type for event in prob_surface.warning_diagnostics.events] == [
        "cdf_repair"
    ]

    prob_surface._internal_curve_cache[cache_key] = _fake_probability_curve(
        resolved_market,
        _cdf_repair_metadata(expiry=expiry, repair_applied=False),
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        prob_surface.pdf(100.0, t=expiry)

    assert recorded_warnings == []
    assert prob_surface.warning_diagnostics.events == []


def test_public_prob_surface_slice_warns_while_surface_query_aggregates(
    fitted_vol_surface,
    monkeypatch,
):
    """Public slices emit curve warnings; surface queries aggregate warnings."""
    import oipd.interface.probability as probability_interface
    from oipd.pipelines.probability.models import DistributionSnapshot
    from oipd.warnings import ModelRiskWarning

    def fake_materializer(definition, *args, **kwargs):
        """Return repaired metadata for any probability definition."""
        expiry = definition.vol_metadata.get("expiry", "2025-02-01")
        return DistributionSnapshot(
            prices=np.array([90.0, 100.0, 110.0]),
            pdf_values=np.array([0.01, 0.02, 0.01]),
            cdf_values=np.array([0.1, 0.5, 0.9]),
            metadata=_cdf_repair_metadata(expiry=expiry),
        )

    monkeypatch.setattr(
        probability_interface,
        "materialize_distribution_from_definition",
        fake_materializer,
    )

    public_surface = fitted_vol_surface.implied_distribution()
    public_curve = public_surface.slice(public_surface.expiries[0])
    with pytest.warns(ModelRiskWarning, match="ProbCurve\\.materialize"):
        _ = public_curve.quantile(0.5)
    assert public_curve.warning_diagnostics.events[0].event_type == "cdf_repair"

    internal_surface = fitted_vol_surface.implied_distribution()
    with pytest.warns(ModelRiskWarning, match="ProbSurface\\.quantile"):
        _ = internal_surface.quantile(0.5, t=internal_surface.expiries[0])
    assert internal_surface.warning_diagnostics.events[0].event_type == "cdf_repair"


def test_cdf_repair_details_include_surface_lineage_fields(
    fitted_vol_surface,
    monkeypatch,
):
    """CDF repair diagnostics retain compact surface lineage details."""
    import oipd.interface.probability as probability_interface
    from oipd.warnings import ModelRiskWarning

    prob_surface = fitted_vol_surface.implied_distribution()
    expiry = prob_surface.expiries[0]
    cache_key = int(pd.Timestamp(expiry).value)
    metadata = _cdf_repair_metadata(
        expiry=expiry,
        time_to_expiry_years=0.25,
        interpolated=True,
        interpolated_from_expiries=[
            pd.Timestamp("2025-02-01"),
            pd.Timestamp("2025-04-01"),
        ],
    )
    prob_surface._internal_curve_cache[cache_key] = _fake_probability_curve(
        prob_surface.slice(expiry).resolved_market,
        metadata,
    )

    monkeypatch.setattr(
        probability_interface,
        "build_surface_density_results_frame",
        lambda surface, *args, **kwargs: pd.DataFrame(
            {"expiry": [expiry], "price": [100.0], "pdf": [0.1], "cdf": [0.5]}
        ),
    )

    with pytest.warns(ModelRiskWarning):
        prob_surface.density_results()

    details = prob_surface.warning_diagnostics.events[0].details
    assert details["time_to_expiry_years"] == pytest.approx(0.25)
    assert details["interpolated"] is True
    assert details["interpolated_from_expiries"] == (
        "2025-02-01T00:00:00",
        "2025-04-01T00:00:00",
    )


def test_warning_diagnostics_has_no_public_mutation_methods():
    """WarningDiagnostics exposes read APIs but no public mutation methods."""
    from oipd.interface.warning_diagnostics import WarningDiagnostics

    diagnostics = WarningDiagnostics()

    assert not hasattr(diagnostics, "replace_scope_events")
    assert not hasattr(diagnostics, "clear")
    assert not hasattr(diagnostics, "to_frame")


def test_vol_curve_fit_records_data_quality_events_and_one_summary_warning(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolCurve.fit translates fallback and stale reports into one summary warning."""
    from oipd import VolCurve
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    def fake_fit_vol_curve_internal(*args, **kwargs):
        """Return fake fit metadata containing two data-quality issues."""
        return _fake_vol_curve, _fit_metadata(
            mid_price_filled=np.int64(3),
            staleness_report={
                "removed_count": np.int64(4),
                "max_staleness_days": np.int64(3),
                "min_age": np.int64(5),
                "max_age": np.int64(8),
                "strike_desc": np.int64(2),
            },
        )

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        fake_fit_vol_curve_internal,
    )

    vol_curve = VolCurve()
    with pytest.warns(DataQualityWarning) as recorded_warnings:
        vol_curve.fit(single_expiry_chain, market_inputs)

    assert len(recorded_warnings) == 1
    warning_message = str(recorded_warnings[0].message)
    assert ".warning_diagnostics.events" in warning_message
    assert "Filled 3 missing mid prices" not in warning_message
    assert "Filtered 4 option rows" not in warning_message

    events = vol_curve.warning_diagnostics.events
    assert [event.event_type for event in events] == [
        "price_fallback",
        "stale_quote_filter",
    ]
    assert {event.category for event in events} == {"data_quality"}
    for event in events:
        for detail_value in event.details.values():
            _assert_json_like_detail_value(detail_value)


def test_vol_curve_fit_records_butterfly_event_and_one_model_risk_warning(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolCurve.fit translates SVI butterfly diagnostics into ModelRiskWarning."""
    from oipd import VolCurve
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import ModelRiskWarning

    def fake_fit_vol_curve_internal(*args, **kwargs):
        """Return fake fit metadata containing butterfly diagnostics."""
        return _fake_vol_curve, _fit_metadata(
            diagnostics=_FakeDiagnostics(
                min_g=np.float64(-0.002),
                butterfly_warning=np.float64(-0.002),
            )
        )

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        fake_fit_vol_curve_internal,
    )

    vol_curve = VolCurve()
    with pytest.warns(ModelRiskWarning) as recorded_warnings:
        vol_curve.fit(single_expiry_chain, market_inputs)

    assert len(recorded_warnings) == 1
    assert ".warning_diagnostics.events" in str(recorded_warnings[0].message)

    events = vol_curve.warning_diagnostics.events
    assert len(events) == 1
    assert events[0].category == "model_risk"
    assert events[0].event_type == "butterfly_arbitrage"
    assert events[0].details["min_g"] == -0.002
    assert events[0].details["butterfly_min_g_threshold"] == -1e-6
    assert events[0].details["diagnostics_status"] == "warning"
    assert events[0].details["diagnostic_source"] == "svi.butterfly_warning"


def test_vol_curve_fit_uses_material_min_g_fallback_for_butterfly_warning(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolCurve.fit records material min_g fallback when butterfly_warning is absent."""
    from oipd import VolCurve
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import ModelRiskWarning

    def fake_fit_vol_curve_internal(*args, **kwargs):
        """Return fake fit metadata containing material min_g diagnostics."""
        return _fake_vol_curve, _fit_metadata(
            diagnostics=_FakeDiagnostics(
                min_g=np.float64(-2e-6),
                butterfly_warning=None,
            )
        )

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        fake_fit_vol_curve_internal,
    )

    vol_curve = VolCurve()
    with pytest.warns(ModelRiskWarning):
        vol_curve.fit(single_expiry_chain, market_inputs)

    events = vol_curve.warning_diagnostics.events
    assert len(events) == 1
    assert events[0].event_type == "butterfly_arbitrage"
    assert events[0].details["min_g"] == -2e-6
    assert events[0].details["butterfly_min_g_threshold"] == -1e-6
    assert events[0].details["diagnostic_source"] == "svi.min_g"


def test_vol_curve_fit_ignores_immaterial_negative_butterfly_dust(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolCurve.fit does not warn for min_g above the material SVI threshold."""
    from oipd import VolCurve
    from oipd.interface import volatility as volatility_interface

    def fake_fit_vol_curve_internal(*args, **kwargs):
        """Return fake fit metadata with immaterial negative butterfly dust."""
        return _fake_vol_curve, _fit_metadata(
            diagnostics=_FakeDiagnostics(
                min_g=np.float64(-5e-7),
                butterfly_warning=None,
            )
        )

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        fake_fit_vol_curve_internal,
    )

    vol_curve = VolCurve()
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        vol_curve.fit(single_expiry_chain, market_inputs)

    assert recorded_warnings == []
    assert vol_curve.warning_diagnostics.events == []


def test_vol_curve_fit_warning_as_error_does_not_commit_partial_state(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolCurve.fit is transactional when summary warnings are treated as errors."""
    from oipd import VolCurve
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    def fake_fit_vol_curve_internal(*args, **kwargs):
        """Return fake fit metadata containing a data-quality issue."""
        return _fake_vol_curve, _fit_metadata(mid_price_filled=2)

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        fake_fit_vol_curve_internal,
    )

    vol_curve = VolCurve()
    with pytest.raises(DataQualityWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataQualityWarning)
            vol_curve.fit(single_expiry_chain, market_inputs)

    with pytest.raises(ValueError, match="Call fit before evaluating the curve"):
        vol_curve.implied_vol([100.0])
    assert vol_curve.warning_diagnostics.events == []


def test_vol_surface_fit_aggregates_data_quality_reports_into_one_warning(
    multi_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolSurface.fit emits one data-quality warning for repeated expiry reports."""
    from oipd import VolSurface
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    fit_warning_reports = {
        "price_fallback": [
            {
                "expiry_str": "2025-02-01",
                "filled_count": np.int64(2),
                "price_method": "mid",
            },
            {
                "expiry_str": "2025-04-01",
                "filled_count": np.int64(1),
                "price_method": "mid",
            },
        ],
        "stale_quote_filter": [
            {
                "expiry_str": "2025-02-01",
                "removed_count": np.int64(4),
                "max_staleness_days": np.int64(3),
                "min_age": np.int64(5),
                "max_age": np.int64(8),
                "strike_desc": np.int64(2),
            }
        ],
        "skipped_expiry": [],
    }

    def fake_fit_surface(*args, **kwargs):
        """Return fake surface metadata containing data-quality reports."""
        return _fake_surface_model(fit_warning_reports, market_inputs)

    monkeypatch.setattr(volatility_interface, "fit_surface", fake_fit_surface)
    monkeypatch.setattr(
        volatility_interface,
        "build_interpolator_from_fitted_surface",
        _fake_interpolator,
    )

    vol_surface = VolSurface()
    with pytest.warns(DataQualityWarning) as recorded_warnings:
        vol_surface.fit(multi_expiry_chain, market_inputs)

    assert len(recorded_warnings) == 1
    assert ".warning_diagnostics.events" in str(recorded_warnings[0].message)
    events = vol_surface.warning_diagnostics.events
    assert [event.event_type for event in events] == [
        "price_fallback",
        "price_fallback",
        "stale_quote_filter",
    ]
    assert {event.details["expiry"] for event in events} == {
        "2025-02-01",
        "2025-04-01",
    }
    for event in events:
        for detail_value in event.details.values():
            _assert_json_like_detail_value(detail_value)


def test_vol_surface_fit_warning_as_error_does_not_commit_partial_state(
    multi_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolSurface.fit is transactional when summary warnings are treated as errors."""
    from oipd import VolSurface
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    fit_warning_reports = {
        "price_fallback": [
            {
                "expiry_str": "2025-02-01",
                "filled_count": 2,
                "price_method": "mid",
            }
        ],
        "stale_quote_filter": [],
        "skipped_expiry": [],
    }

    def fake_fit_surface(*args, **kwargs):
        """Return fake surface metadata containing data-quality reports."""
        return _fake_surface_model(fit_warning_reports, market_inputs)

    monkeypatch.setattr(volatility_interface, "fit_surface", fake_fit_surface)
    monkeypatch.setattr(
        volatility_interface,
        "build_interpolator_from_fitted_surface",
        _fake_interpolator,
    )

    vol_surface = VolSurface()
    with pytest.raises(DataQualityWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataQualityWarning)
            vol_surface.fit(multi_expiry_chain, market_inputs)

    assert vol_surface.expiries == ()
    assert vol_surface.warning_diagnostics.events == []


def test_vol_surface_fit_records_skipped_expiry_and_one_workflow_warning(
    multi_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """VolSurface.fit translates skipped expiry reports into WorkflowWarning."""
    from oipd import VolSurface
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import WorkflowWarning

    fit_warning_reports = {
        "price_fallback": [],
        "stale_quote_filter": [],
        "skipped_expiry": [
            {
                "expiry_str": "2025-07-01",
                "exception_type": "ValueError",
                "reason": "ValueError: Need at least 5 strikes",
            }
        ],
    }

    def fake_fit_surface(*args, **kwargs):
        """Return fake surface metadata containing a skipped-expiry report."""
        return _fake_surface_model(fit_warning_reports, market_inputs)

    monkeypatch.setattr(volatility_interface, "fit_surface", fake_fit_surface)
    monkeypatch.setattr(
        volatility_interface,
        "build_interpolator_from_fitted_surface",
        _fake_interpolator,
    )

    vol_surface = VolSurface()
    with pytest.warns(WorkflowWarning) as recorded_warnings:
        vol_surface.fit(multi_expiry_chain, market_inputs)

    assert len(recorded_warnings) == 1
    assert ".warning_diagnostics.events" in str(recorded_warnings[0].message)
    events = vol_surface.warning_diagnostics.events
    assert len(events) == 1
    assert events[0].category == "workflow"
    assert events[0].event_type == "skipped_expiry"
    assert events[0].details["expiry"] == "2025-07-01"


def test_discrete_surface_warning_reports_are_defensively_copied(market_inputs):
    """DiscreteSurface detaches warning reports on input and property access."""
    from oipd.market_inputs import resolve_market
    from oipd.pipelines.vol_surface.models import DiscreteSurface

    expiry = pd.Timestamp("2025-02-01")
    reports = {
        "price_fallback": [
            {
                "expiry_str": "2025-02-01",
                "filled_count": 2,
                "price_method": "mid",
            }
        ]
    }
    surface = DiscreteSurface(
        {expiry: {"curve": _fake_vol_curve, "metadata": _fit_metadata()}},
        {expiry: resolve_market(market_inputs)},
        {expiry: pd.DataFrame({"expiry": [expiry]})},
        fit_warning_reports=reports,
    )

    reports["price_fallback"][0]["filled_count"] = 99
    returned_reports = surface.fit_warning_reports
    returned_reports["price_fallback"][0]["filled_count"] = 123

    assert surface.fit_warning_reports["price_fallback"][0]["filled_count"] == 2


def test_vol_surface_exact_slice_carries_expiry_warning_diagnostics(
    multi_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """Exact VolSurface slices carry relevant per-expiry warning diagnostics."""
    from oipd import VolSurface
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    fit_warning_reports = {
        "price_fallback": [
            {
                "expiry_str": "2025-02-01",
                "filled_count": 2,
                "price_method": "mid",
            }
        ],
        "stale_quote_filter": [],
        "skipped_expiry": [],
    }

    def fake_fit_surface(*args, **kwargs):
        """Return fake surface metadata with one per-expiry warning report."""
        return _fake_surface_model(fit_warning_reports, market_inputs)

    monkeypatch.setattr(volatility_interface, "fit_surface", fake_fit_surface)
    monkeypatch.setattr(
        volatility_interface,
        "build_interpolator_from_fitted_surface",
        _fake_interpolator,
    )

    vol_surface = VolSurface()
    with pytest.warns(DataQualityWarning):
        vol_surface.fit(multi_expiry_chain, market_inputs)

    vol_curve = vol_surface.slice(pd.Timestamp("2025-02-01"))

    assert len(vol_curve.warning_diagnostics.events) == 1
    assert vol_curve.warning_diagnostics.events[0].event_type == "price_fallback"


def test_direct_refit_clears_inherited_surface_warning_diagnostics(
    multi_expiry_chain,
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """Direct VolCurve.fit clears warning events inherited from VolSurface.slice."""
    from oipd import VolSurface
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    fit_warning_reports = {
        "price_fallback": [
            {
                "expiry_str": "2025-02-01",
                "filled_count": 2,
                "price_method": "mid",
            }
        ],
        "stale_quote_filter": [],
        "skipped_expiry": [],
    }

    def fake_fit_surface(*args, **kwargs):
        """Return fake surface metadata with one inherited warning report."""
        return _fake_surface_model(fit_warning_reports, market_inputs)

    def clean_fit_vol_curve_internal(*args, **kwargs):
        """Return clean fit metadata for direct slice refit."""
        return _fake_vol_curve, _fit_metadata(mid_price_filled=0)

    monkeypatch.setattr(volatility_interface, "fit_surface", fake_fit_surface)
    monkeypatch.setattr(
        volatility_interface,
        "build_interpolator_from_fitted_surface",
        _fake_interpolator,
    )

    vol_surface = VolSurface()
    with pytest.warns(DataQualityWarning):
        vol_surface.fit(multi_expiry_chain, market_inputs)
    vol_curve = vol_surface.slice(pd.Timestamp("2025-02-01"))
    assert len(vol_curve.warning_diagnostics.events) == 1

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        clean_fit_vol_curve_internal,
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        vol_curve.fit(single_expiry_chain, market_inputs)

    assert recorded_warnings == []
    assert vol_curve.warning_diagnostics.events == []


def test_refitting_replaces_prior_fit_scope_warning_events(
    single_expiry_chain,
    market_inputs,
    monkeypatch,
):
    """Repeated fitting replaces stale fit-scope warning diagnostics."""
    from oipd import VolCurve
    from oipd.interface import volatility as volatility_interface
    from oipd.warnings import DataQualityWarning

    metadata_sequence = [
        _fit_metadata(mid_price_filled=2),
        _fit_metadata(mid_price_filled=0),
    ]

    def fake_fit_vol_curve_internal(*args, **kwargs):
        """Return the next fake metadata payload."""
        return _fake_vol_curve, metadata_sequence.pop(0)

    monkeypatch.setattr(
        volatility_interface,
        "fit_vol_curve_internal",
        fake_fit_vol_curve_internal,
    )

    vol_curve = VolCurve()
    with pytest.warns(DataQualityWarning):
        vol_curve.fit(single_expiry_chain, market_inputs)
    assert len(vol_curve.warning_diagnostics.events) == 1

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        vol_curve.fit(single_expiry_chain, market_inputs)

    assert recorded_warnings == []
    assert vol_curve.warning_diagnostics.events == []


def test_same_scope_replacement_removes_only_stale_events_from_that_scope():
    """Internal same-scope replacement preserves other scopes and new events."""
    from oipd.interface.warning_diagnostics import WarningDiagnostics, WarningEvent

    diagnostics = WarningDiagnostics()
    diagnostics._replace_scope_events(
        "vol_curve.fit",
        [
            WarningEvent(
                category="data_quality",
                event_type="old_price_fallback",
                severity="warning",
                scope="vol_curve.fit",
                message="Old fit event.",
                details={"count": 1},
            )
        ],
    )
    diagnostics._replace_scope_events(
        "prob_curve.materialize",
        [
            WarningEvent(
                category="model_risk",
                event_type="cdf_repair",
                severity="severe",
                scope="prob_curve.materialize",
                message="CDF was repaired.",
                details={"max_violation": 0.02},
            )
        ],
    )

    diagnostics._replace_scope_events(
        "vol_curve.fit",
        [
            WarningEvent(
                category="data_quality",
                event_type="price_fallback",
                severity="warning",
                scope="vol_curve.fit",
                message="Missing mids used last prices.",
                details={"count": 3},
            ),
            WarningEvent(
                category="data_quality",
                event_type="stale_quote_filter",
                severity="warning",
                scope="vol_curve.fit",
                message="Stale rows were filtered.",
                details={"removed_count": 2},
            ),
        ],
    )

    event_types = [event.event_type for event in diagnostics.events]

    assert event_types == [
        "cdf_repair",
        "price_fallback",
        "stale_quote_filter",
    ]


def test_same_scope_replacement_with_fewer_events_removes_stale_same_scope_events():
    """Internal replacement with fewer events removes stale same-scope events."""
    from oipd.interface.warning_diagnostics import WarningDiagnostics, WarningEvent

    diagnostics = WarningDiagnostics()
    diagnostics._replace_scope_events(
        "surface.fit",
        [
            WarningEvent(
                category="workflow",
                event_type="skipped_expiry",
                severity="warning",
                scope="surface.fit",
                message="One expiry was skipped.",
                details={"expiry": "2025-02-01"},
            ),
            WarningEvent(
                category="data_quality",
                event_type="stale_quote_filter",
                severity="warning",
                scope="surface.fit",
                message="Stale rows were filtered.",
                details={"removed_count": 5},
            ),
        ],
    )

    diagnostics._replace_scope_events(
        "surface.fit",
        [
            WarningEvent(
                category="workflow",
                event_type="skipped_expiry",
                severity="warning",
                scope="surface.fit",
                message="One expiry was skipped.",
                details={"expiry": "2025-03-01"},
            ),
        ],
    )

    events = diagnostics.events

    assert len(events) == 1
    assert events[0].event_type == "skipped_expiry"
    assert events[0].details["expiry"] == "2025-03-01"


def test_scope_replacement_rejects_scope_mismatch_without_mutating_state():
    """Internal replacement is atomic when an event has the wrong scope."""
    from oipd.interface.warning_diagnostics import WarningDiagnostics, WarningEvent

    original_event = WarningEvent(
        category="workflow",
        event_type="skipped_expiry",
        severity="warning",
        scope="surface.fit",
        message="One expiry was skipped.",
        details={"expiry": "2025-02-01"},
    )
    diagnostics = WarningDiagnostics([original_event])

    with pytest.raises(ValueError):
        diagnostics._replace_scope_events(
            "prob_surface.materialize",
            [
                WarningEvent(
                    category="model_risk",
                    event_type="cdf_repair",
                    severity="severe",
                    scope="prob_curve.materialize",
                    message="CDF was repaired.",
                    details={"max_violation": 0.02},
                )
            ],
        )

    assert diagnostics.events == [original_event]


def test_summary_worst_severity_uses_ordered_scale():
    """Summary reports the worst severity using info < warning < severe."""
    from oipd.interface.warning_diagnostics import WarningDiagnostics, WarningEvent

    diagnostics = WarningDiagnostics(
        [
            WarningEvent(
                category="workflow",
                event_type="skipped_expiry",
                severity="info",
                scope="surface.fit",
                message="One expiry was skipped.",
                details={},
            ),
            WarningEvent(
                category="data_quality",
                event_type="stale_quote_filter",
                severity="warning",
                scope="surface.fit",
                message="Stale rows were filtered.",
                details={},
            ),
            WarningEvent(
                category="model_risk",
                event_type="cdf_repair",
                severity="severe",
                scope="prob_curve.materialize",
                message="CDF was repaired.",
                details={},
            ),
        ]
    )

    assert diagnostics.summary.worst_severity == "severe"


def test_mutating_returned_events_list_does_not_mutate_diagnostics():
    """The events property returns a copy, not the internal diagnostics list."""
    from oipd.interface.warning_diagnostics import WarningDiagnostics, WarningEvent

    original_event = WarningEvent(
        category="numerical",
        event_type="finite_difference_instability",
        severity="warning",
        scope="prob_curve.materialize",
        message="Finite differences were unstable.",
        details={"grid_points": 5},
    )
    diagnostics = WarningDiagnostics([original_event])

    returned_events = diagnostics.events
    returned_events.append(
        WarningEvent(
            category="workflow",
            event_type="skipped_expiry",
            severity="info",
            scope="surface.fit",
            message="One expiry was skipped.",
            details={},
        )
    )

    assert diagnostics.events == [original_event]


def test_warning_event_details_are_defensively_copied_and_read_only():
    """WarningEvent details are detached from caller-owned mutable inputs."""
    from oipd.interface.warning_diagnostics import WarningEvent

    details = {"count": 1, "expiries": ["2025-02-01"]}
    event = WarningEvent(
        category="data_quality",
        event_type="price_fallback",
        severity="warning",
        scope="vol_curve.fit",
        message="Missing mids used last prices.",
        details=details,
    )

    details["count"] = 99
    details["expiries"].append("2025-03-01")

    assert event.details["count"] == 1
    assert event.details["expiries"] == ("2025-02-01",)
    with pytest.raises(TypeError):
        event.details["count"] = 2


def test_warning_event_trims_required_string_fields():
    """WarningEvent stores trimmed event identifiers and messages."""
    from oipd.interface.warning_diagnostics import WarningEvent

    event = WarningEvent(
        category="data_quality",
        event_type=" price_fallback ",
        severity="warning",
        scope=" vol_curve.fit ",
        message=" Missing mids used last prices. ",
        details={},
    )

    assert event.event_type == "price_fallback"
    assert event.scope == "vol_curve.fit"
    assert event.message == "Missing mids used last prices."


def test_warning_event_is_immutable():
    """WarningEvent fields cannot be reassigned after construction."""
    from dataclasses import FrozenInstanceError

    from oipd.interface.warning_diagnostics import WarningEvent

    event = WarningEvent(
        category="data_quality",
        event_type="price_fallback",
        severity="warning",
        scope="vol_curve.fit",
        message="Missing mids used last prices.",
        details={},
    )

    with pytest.raises(FrozenInstanceError):
        event.message = "Changed."


def test_warning_event_rejects_invalid_category():
    """WarningEvent rejects unsupported broad categories."""
    from oipd.interface.warning_diagnostics import WarningEvent

    with pytest.raises(ValueError):
        WarningEvent(
            category="market_microstructure",
            event_type="price_fallback",
            severity="warning",
            scope="vol_curve.fit",
            message="Missing mids used last prices.",
            details={},
        )


def test_warning_event_rejects_invalid_severity():
    """WarningEvent rejects unsupported severity labels."""
    from oipd.interface.warning_diagnostics import WarningEvent

    with pytest.raises(ValueError):
        WarningEvent(
            category="data_quality",
            event_type="price_fallback",
            severity="critical",
            scope="vol_curve.fit",
            message="Missing mids used last prices.",
            details={},
        )


@pytest.mark.parametrize("field_name", ["event_type", "scope", "message"])
@pytest.mark.parametrize("bad_value", [None, 123])
def test_warning_event_rejects_non_string_required_fields(field_name, bad_value):
    """WarningEvent requires event_type, scope, and message to be strings."""
    from oipd.interface.warning_diagnostics import WarningEvent

    kwargs = {
        "category": "data_quality",
        "event_type": "price_fallback",
        "severity": "warning",
        "scope": "vol_curve.fit",
        "message": "Missing mids used last prices.",
        "details": {},
    }
    kwargs[field_name] = bad_value

    with pytest.raises(TypeError):
        WarningEvent(**kwargs)


@pytest.mark.parametrize("field_name", ["event_type", "scope", "message"])
@pytest.mark.parametrize("bad_value", ["", "   "])
def test_warning_event_rejects_empty_required_fields_after_trimming(
    field_name,
    bad_value,
):
    """WarningEvent requires non-empty event_type, scope, and message."""
    from oipd.interface.warning_diagnostics import WarningEvent

    kwargs = {
        "category": "data_quality",
        "event_type": "price_fallback",
        "severity": "warning",
        "scope": "vol_curve.fit",
        "message": "Missing mids used last prices.",
        "details": {},
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError):
        WarningEvent(**kwargs)


def test_warning_event_rejects_non_mapping_details():
    """WarningEvent details must be a mapping."""
    from oipd.interface.warning_diagnostics import WarningEvent

    with pytest.raises(TypeError):
        WarningEvent(
            category="data_quality",
            event_type="price_fallback",
            severity="warning",
            scope="vol_curve.fit",
            message="Missing mids used last prices.",
            details=[("count", 1)],
        )


def test_warning_event_allows_nested_json_like_details():
    """WarningEvent accepts nested small audit payloads."""
    from oipd.interface.warning_diagnostics import WarningEvent

    event = WarningEvent(
        category="data_quality",
        event_type="price_fallback",
        severity="warning",
        scope="vol_curve.fit",
        message="Missing mids used last prices.",
        details={
            "count": 2,
            "expiry": "2025-02-01",
            "accepted": True,
            "max_violation": None,
            "nested": {"strikes": [90.0, 100.0], "labels": {"bid", "ask"}},
        },
    )

    assert event.details["nested"]["strikes"] == (90.0, 100.0)
    assert event.details["nested"]["labels"] == frozenset({"bid", "ask"})


@pytest.mark.parametrize(
    "unsupported_value",
    [
        np.array([1.0, 2.0]),
        pd.DataFrame({"strike": [100.0]}),
        object(),
    ],
)
def test_warning_event_rejects_unsupported_detail_values(unsupported_value):
    """WarningEvent rejects live objects instead of storing them in details."""
    from oipd.interface.warning_diagnostics import WarningEvent

    with pytest.raises(TypeError):
        WarningEvent(
            category="data_quality",
            event_type="price_fallback",
            severity="warning",
            scope="vol_curve.fit",
            message="Missing mids used last prices.",
            details={"unsupported": unsupported_value},
        )


def test_warning_class_for_category_returns_expected_warning_classes():
    """Category-to-warning-class mapping matches the public taxonomy."""
    from oipd.interface.warning_diagnostics import warning_class_for_category
    from oipd.warnings import (
        DataQualityWarning,
        ModelRiskWarning,
        NumericalWarning,
        WorkflowWarning,
    )

    assert warning_class_for_category("data_quality") is DataQualityWarning
    assert warning_class_for_category("model_risk") is ModelRiskWarning
    assert warning_class_for_category("numerical") is NumericalWarning
    assert warning_class_for_category("workflow") is WorkflowWarning
