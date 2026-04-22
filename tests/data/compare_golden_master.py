"""
Compare a generated golden-master candidate against the committed baseline.

This helper is intentionally non-mutating for committed golden files. It routes
``generate_golden_master.py`` output to a temporary or user-supplied candidate
path via ``OIPD_GOLDEN_MASTER_OUT`` before comparing JSON payloads.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent.parent
GENERATOR_PATH = DATA_DIR / "generate_golden_master.py"
PRICE_GRID_RTOL = 1e-12
PRICE_GRID_ATOL = 1e-10
COMMITTED_GOLDENS = {
    (DATA_DIR / "golden_master.json").resolve(),
    (DATA_DIR / "golden_master_linux.json").resolve(),
}


def _platform_golden_path() -> Path:
    """Return the committed golden file for the current platform."""
    if sys.platform.startswith("linux"):
        return DATA_DIR / "golden_master_linux.json"
    return DATA_DIR / "golden_master.json"


def _default_candidate_path() -> Path:
    """Create a temporary candidate path without keeping the file handle open."""
    fd, path = tempfile.mkstemp(prefix="oipd_golden_candidate_", suffix=".json")
    os.close(fd)
    return Path(path)


def _resolve_candidate_path(raw_path: str | None) -> Path:
    """Resolve and validate the candidate output path."""
    candidate_path = (
        Path(raw_path).expanduser() if raw_path else _default_candidate_path()
    )
    resolved = candidate_path.resolve()
    if resolved in COMMITTED_GOLDENS:
        raise ValueError(
            f"Refusing to write candidate output to committed golden file: {resolved}"
        )
    return resolved


def _load_generator():
    """Load the existing generation script as a module."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location(
        "oipd_generate_golden_master", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator from {GENERATOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_candidate(candidate_path: Path) -> None:
    """Generate a golden-master candidate at ``candidate_path``."""
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    old_output = os.environ.get("OIPD_GOLDEN_MASTER_OUT")
    os.environ["OIPD_GOLDEN_MASTER_OUT"] = str(candidate_path)
    try:
        module = _load_generator()
        module.generate_golden_master()
    finally:
        if old_output is None:
            os.environ.pop("OIPD_GOLDEN_MASTER_OUT", None)
        else:
            os.environ["OIPD_GOLDEN_MASTER_OUT"] = old_output


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def _float_array(values: Iterable[Any] | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(list(values), dtype=float)


def _diff_stats(reference: np.ndarray | None, candidate: np.ndarray | None) -> str:
    """Return max-absolute and RMS drift over the common prefix."""
    if reference is None or candidate is None:
        return "not available"
    n = min(reference.size, candidate.size)
    if n == 0:
        return "no common points"

    diff = candidate[:n] - reference[:n]
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    suffix = ""
    if reference.size != candidate.size:
        suffix = (
            " (length mismatch: "
            f"reference={reference.size}, candidate={candidate.size})"
        )
    return f"max_abs={max_abs:.6e}, rms={rms:.6e}, n={n}{suffix}"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value): .10e}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_range(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "empty"
    return f"[{float(np.min(finite)):.10e}, {float(np.max(finite)):.10e}]"


def _grid_summary(values: np.ndarray | None) -> str:
    if values is None:
        return "length=missing, range=missing"
    if values.size == 0:
        return "length=0, range=empty"

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"length={values.size}, range=empty, nonfinite={values.size}"

    nonfinite = int(values.size - finite.size)
    suffix = f", nonfinite={nonfinite}" if nonfinite else ""
    return f"length={values.size}, range={_fmt_range(finite)}{suffix}"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "not available"
    return "yes" if value else "no"


def _same_length(
    reference: np.ndarray | None, candidate: np.ndarray | None
) -> bool | None:
    if reference is None or candidate is None:
        return None
    return reference.size == candidate.size


def _same_price_values(
    reference: np.ndarray | None, candidate: np.ndarray | None
) -> bool | None:
    if reference is None or candidate is None:
        return None
    if reference.size != candidate.size:
        return False
    return bool(
        np.allclose(
            reference,
            candidate,
            rtol=PRICE_GRID_RTOL,
            atol=PRICE_GRID_ATOL,
            equal_nan=False,
        )
    )


def _aligned_diff_stats(reference: np.ndarray, candidate: np.ndarray) -> str:
    """Return drift stats for arrays that are known to share one x-axis."""
    if reference.size != candidate.size:
        return (
            "not comparable: length mismatch "
            f"(reference={reference.size}, candidate={candidate.size})"
        )
    if reference.size == 0:
        return "no comparable points"

    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(finite):
        return (
            "not comparable: no finite comparable points "
            f"(n={reference.size}, nonfinite_pairs={reference.size})"
        )

    diff = candidate[finite] - reference[finite]
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    nonfinite_pairs = int(reference.size - np.sum(finite))
    suffix = f", nonfinite_pairs={nonfinite_pairs}" if nonfinite_pairs else ""
    return f"max_abs={max_abs:.6e}, rms={rms:.6e}, n={int(np.sum(finite))}{suffix}"


def _validate_distribution_arrays(
    prices: np.ndarray | None, values: np.ndarray | None, label: str
) -> str | None:
    if prices is None:
        return f"not comparable: {label} price grid not available"
    if values is None:
        return f"not available: {label} values not available"
    if prices.size != values.size:
        return (
            f"not comparable: {label} price/value length mismatch "
            f"(prices={prices.size}, values={values.size})"
        )
    if prices.size == 0:
        return f"not comparable: {label} price grid is empty"
    if not np.all(np.isfinite(prices)):
        return f"not comparable: {label} price grid contains nonfinite values"
    return None


def _sorted_unique_grid(
    prices: np.ndarray, values: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, str | None]:
    order = np.argsort(prices)
    sorted_prices = prices[order]
    sorted_values = values[order]

    if not np.all(np.isfinite(sorted_values)):
        return (
            sorted_prices,
            sorted_values,
            (f"not comparable: {label} values contain nonfinite values"),
        )
    if sorted_prices.size < 2:
        return (
            sorted_prices,
            sorted_values,
            (
                f"not comparable: {label} price grid needs at least two points "
                "for interpolation"
            ),
        )
    if np.any(np.diff(sorted_prices) <= 0.0):
        return (
            sorted_prices,
            sorted_values,
            (f"not comparable: {label} price grid contains duplicate values"),
        )
    return sorted_prices, sorted_values, None


def _distribution_direct_stats(
    reference_prices: np.ndarray | None,
    candidate_prices: np.ndarray | None,
    reference_values: np.ndarray | None,
    candidate_values: np.ndarray | None,
) -> str:
    ref_error = _validate_distribution_arrays(
        reference_prices, reference_values, "reference"
    )
    if ref_error is not None:
        return ref_error

    cand_error = _validate_distribution_arrays(
        candidate_prices, candidate_values, "candidate"
    )
    if cand_error is not None:
        return cand_error

    assert reference_values is not None
    assert candidate_values is not None

    stats = _aligned_diff_stats(reference_values, candidate_values)
    return f"{stats}, mode=direct-grid-match"


def _distribution_interpolated_stats(
    reference_prices: np.ndarray | None,
    candidate_prices: np.ndarray | None,
    reference_values: np.ndarray | None,
    candidate_values: np.ndarray | None,
    *,
    approximate: bool,
) -> str:
    ref_error = _validate_distribution_arrays(
        reference_prices, reference_values, "reference"
    )
    if ref_error is not None:
        return ref_error

    cand_error = _validate_distribution_arrays(
        candidate_prices, candidate_values, "candidate"
    )
    if cand_error is not None:
        return cand_error

    assert reference_prices is not None
    assert candidate_prices is not None
    assert reference_values is not None
    assert candidate_values is not None

    sorted_candidate_prices, sorted_candidate_values, sort_error = _sorted_unique_grid(
        candidate_prices, candidate_values, "candidate"
    )
    if sort_error is not None:
        return sort_error

    finite_reference_values = np.isfinite(reference_values)
    overlap_low = max(
        float(np.min(reference_prices)), float(sorted_candidate_prices[0])
    )
    overlap_high = min(
        float(np.max(reference_prices)), float(sorted_candidate_prices[-1])
    )
    if overlap_low > overlap_high:
        return (
            "not comparable: price grids do not overlap "
            f"(overlap_range=empty, reference_range={_fmt_range(reference_prices)}, "
            f"candidate_range={_fmt_range(sorted_candidate_prices)})"
        )

    overlap = (
        (reference_prices >= overlap_low)
        & (reference_prices <= overlap_high)
        & finite_reference_values
    )
    if not np.any(overlap):
        return (
            "not comparable: no finite reference points in overlapping price domain "
            f"(overlap_range=[{overlap_low:.10e}, {overlap_high:.10e}])"
        )

    overlap_prices = reference_prices[overlap]
    interpolated_candidate = np.interp(
        overlap_prices, sorted_candidate_prices, sorted_candidate_values
    )
    stats = _aligned_diff_stats(reference_values[overlap], interpolated_candidate)
    mode = (
        "approximate/interpolated-on-reference-grid"
        if approximate
        else "interpolated-on-reference-grid"
    )
    return (
        f"{stats}, mode={mode}, overlap_n={overlap_prices.size}, "
        f"overlap_range={_fmt_range(overlap_prices)}"
    )


def _print_svi_param_diffs(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    print("\nSVI parameter drift")
    print("param                 reference        candidate             diff")

    ref_params = reference.get("svi_params", {})
    cand_params = candidate.get("svi_params", {})
    for key in sorted(set(ref_params) | set(cand_params)):
        ref_value = ref_params.get(key)
        cand_value = cand_params.get(key)
        if ref_value is None or cand_value is None:
            diff = "missing"
        else:
            diff = f"{float(cand_value) - float(ref_value): .10e}"
        print(
            f"{key:<16} {_fmt_float(ref_value):>16} "
            f"{_fmt_float(cand_value):>16} {diff:>16}"
        )


def _print_iv_diffs(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    print("\nIV drift")
    ref_points = reference.get("test_points", {})
    cand_points = candidate.get("test_points", {})
    ref_strikes = _float_array(ref_points.get("strikes"))
    cand_strikes = _float_array(cand_points.get("strikes"))
    ref_ivs = _float_array(ref_points.get("implied_vols"))
    cand_ivs = _float_array(cand_points.get("implied_vols"))

    if (
        ref_strikes is None
        or cand_strikes is None
        or ref_ivs is None
        or cand_ivs is None
    ):
        print("Per-strike IV drift not available.")
        return

    n = min(ref_strikes.size, cand_strikes.size, ref_ivs.size, cand_ivs.size)
    if n == 0:
        print("Per-strike IV drift not available.")
        return

    same_strikes = ref_strikes.size == cand_strikes.size and np.allclose(
        ref_strikes[:n], cand_strikes[:n], rtol=0.0, atol=0.0
    )

    if same_strikes:
        print(_diff_stats(ref_ivs, cand_ivs))
        print("strike             reference_iv    candidate_iv            diff")
        for strike, ref_iv, cand_iv in zip(ref_strikes[:n], ref_ivs[:n], cand_ivs[:n]):
            print(
                f"{strike:>10.4f} {_fmt_float(ref_iv):>20} "
                f"{_fmt_float(cand_iv):>15} {cand_iv - ref_iv:>16.10e}"
            )
    else:
        print(
            "not comparable: strike grid mismatch "
            f"(reference={ref_strikes.size}, candidate={cand_strikes.size})"
        )
        print("ref_strike cand_strike  reference_iv    candidate_iv            diff")
        for ref_strike, cand_strike, ref_iv, cand_iv in zip(
            ref_strikes[:n], cand_strikes[:n], ref_ivs[:n], cand_ivs[:n]
        ):
            print(
                f"{ref_strike:>10.4f} {cand_strike:>11.4f} "
                f"{_fmt_float(ref_iv):>14} {_fmt_float(cand_iv):>15} "
                f"{cand_iv - ref_iv:>16.10e}"
            )


def _print_distribution_diffs(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    print("\nDistribution drift")
    ref_dist = reference.get("distribution", {})
    cand_dist = candidate.get("distribution", {})

    ref_prices = _float_array(ref_dist.get("prices"))
    cand_prices = _float_array(cand_dist.get("prices"))
    ref_pdf = _float_array(ref_dist.get("pdf"))
    cand_pdf = _float_array(cand_dist.get("pdf"))
    ref_cdf = _float_array(ref_dist.get("cdf"))
    cand_cdf = _float_array(cand_dist.get("cdf"))

    same_length = _same_length(ref_prices, cand_prices)
    same_values = _same_price_values(ref_prices, cand_prices)
    grids_match = same_values is True

    print("Price grid drift")
    print(f"reference: {_grid_summary(ref_prices)}")
    print(f"candidate: {_grid_summary(cand_prices)}")
    print(
        f"same_length={_yes_no(same_length)}, "
        f"same_values_within_tolerance={_yes_no(same_values)} "
        f"(rtol={PRICE_GRID_RTOL:.1e}, atol={PRICE_GRID_ATOL:.1e})"
    )

    if grids_match:
        print(
            "PDF: "
            + _distribution_direct_stats(ref_prices, cand_prices, ref_pdf, cand_pdf)
        )
        print(
            "CDF: "
            + _distribution_direct_stats(ref_prices, cand_prices, ref_cdf, cand_cdf)
        )
    else:
        print(
            "PDF: "
            + _distribution_interpolated_stats(
                ref_prices,
                cand_prices,
                ref_pdf,
                cand_pdf,
                approximate=True,
            )
        )
        print(
            "CDF: "
            + _distribution_interpolated_stats(
                ref_prices,
                cand_prices,
                ref_cdf,
                cand_cdf,
                approximate=False,
            )
        )

    if cand_cdf is None or cand_cdf.size == 0:
        print("Candidate CDF bounds: not available")
        return

    finite = cand_cdf[np.isfinite(cand_cdf)]
    if finite.size == 0:
        print(
            "Candidate CDF bounds: min=nan, max=nan, below_0=0, "
            f"above_1=0, nonfinite={cand_cdf.size}, violation=yes"
        )
        return

    cdf_min = float(np.min(finite))
    cdf_max = float(np.max(finite))
    below_zero = int(np.sum(finite < 0.0))
    above_one = int(np.sum(finite > 1.0))
    nonfinite = int(cand_cdf.size - finite.size)
    violation = below_zero > 0 or above_one > 0 or nonfinite > 0
    print(
        "Candidate CDF bounds: "
        f"min={cdf_min:.10e}, max={cdf_max:.10e}, "
        f"below_0={below_zero}, above_1={above_one}, "
        f"nonfinite={nonfinite}, violation={'yes' if violation else 'no'}"
    )


def _print_report(reference_path: Path, candidate_path: Path) -> None:
    reference = _load_json(reference_path)
    candidate = _load_json(candidate_path)

    print("Golden master drift comparison")
    print(f"Reference: {reference_path}")
    print(f"Candidate: {candidate_path}")

    _print_svi_param_diffs(reference, candidate)
    _print_iv_diffs(reference, candidate)
    _print_distribution_diffs(reference, candidate)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a temporary golden-master candidate and compare it against "
            "the committed platform baseline."
        )
    )
    parser.add_argument(
        "--candidate-out",
        help="Path for generated candidate JSON. Defaults to a temporary file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    reference_path = _platform_golden_path()
    try:
        candidate_path = _resolve_candidate_path(args.candidate_out)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    generator_stdout = io.StringIO()
    generator_stderr = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(generator_stdout),
            contextlib.redirect_stderr(generator_stderr),
        ):
            _generate_candidate(candidate_path)
    except Exception as exc:
        print(
            f"Candidate generation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        stdout_text = generator_stdout.getvalue().strip()
        stderr_text = generator_stderr.getvalue().strip()
        if stdout_text:
            print("\nGenerator stdout:", file=sys.stderr)
            print(stdout_text, file=sys.stderr)
        if stderr_text:
            print("\nGenerator stderr:", file=sys.stderr)
            print(stderr_text, file=sys.stderr)
        return 1

    try:
        _print_report(reference_path, candidate_path)
    except Exception as exc:
        print(f"Golden comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
