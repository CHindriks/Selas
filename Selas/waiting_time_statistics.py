"""Waiting-time PDF, CDF, and Kolmogorov-Smirnov analysis for TIC flare data.

This module refactors the original notebook workflow into reusable functions.  It
loads flare and time-series CSV files, identifies observing gaps, fills eligible
observing gaps with simulated flares, computes waiting-time distributions, runs
KS tests against an exponential waiting-time model, optionally writes plots, and
exports a one-row summary table.
"""

from __future__ import annotations

__version__ = "2026-06-24-smoothed-mean-pdf-truncated-cdf-24d"

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import font_manager as fm
    from matplotlib.ticker import ScalarFormatter
except Exception:  # pragma: no cover - only matters on systems without matplotlib
    plt = None
    mpatches = None
    fm = None
    ScalarFormatter = None


BASE_FONTSIZE = 20
BASE_LINEWIDTH = 2.2
SCALE = 1.8
FONTSIZE = BASE_FONTSIZE * SCALE
LINEWIDTH = BASE_LINEWIDTH * SCALE

THESIS_MAIN_COLOR = "black"
THESIS_RAW_COLOR = "0.55"
THESIS_HIGHLIGHT_COLOR = "crimson"
THESIS_FLARE_WINDOW_COLOR = "lightskyblue"
THESIS_RAW_ALPHA = 0.45
THESIS_SPAN_ALPHA = 0.18


def _apply_original_plot_style() -> None:
    """Apply thesis-style Matplotlib defaults used by the reference figures."""

    if plt is None:
        return

    font_family = "DejaVu Serif"
    font_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf"),
        Path("/usr/share/fonts/dejavu-serif-fonts/DejaVuSerifCondensed.ttf"),
    )
    if fm is not None:
        for font_path in font_candidates:
            if font_path.exists():
                try:
                    custom_font = fm.FontProperties(fname=str(font_path))
                    font_family = custom_font.get_name()
                    break
                except Exception:
                    font_family = "DejaVu Serif"

    plt.rcParams.update({
        "font.family": font_family,
        "mathtext.fontset": "stix",
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "axes.linewidth": LINEWIDTH,
        "lines.linewidth": LINEWIDTH,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "legend.fontsize": 0.65 * FONTSIZE,
        "xtick.major.width": LINEWIDTH,
        "ytick.major.width": LINEWIDTH,
        "xtick.minor.width": LINEWIDTH * 0.8,
        "ytick.minor.width": LINEWIDTH * 0.8,
        "xtick.major.size": 7 * SCALE,
        "ytick.major.size": 7 * SCALE,
        "xtick.minor.size": 4 * SCALE,
        "ytick.minor.size": 4 * SCALE,
        "axes.formatter.useoffset": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


_apply_original_plot_style()

try:
    from scipy.stats import kstest
except Exception as exc:  # pragma: no cover - scipy is expected but handled defensively
    kstest = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the waiting-time analysis.

    Parameters
    ----------
    number_of_simulations : int, optional
        Number of simulated flare-time datasets to create.
    waiting_time_limit : float, optional
        Maximum adjacent waiting time, in days, retained for PDF/CDF/KS analysis.
    gap_threshold : float, optional
        Time jump, in days, above which a gap is identified in the light curve.
    max_gap_for_simulation : float, optional
        Maximum gap duration, in days, filled by simulated flares.
    binsize : float, optional
        Histogram bin width, in days, used for the empirical PDF.
    pdf_smoothing_width_days : float, optional
        Gaussian smoothing width, in days, applied to the per-simulation PDF
        histograms before averaging. A value of 0 disables smoothing.
    min_flares_for_target : int, optional
        Minimum flare count used in the target-selection heuristic.
    random_seed : int or None, optional
        Seed for reproducible gap-filling simulations. If None, random draws are
        not seeded.
    """

    number_of_simulations: int = 500
    waiting_time_limit: float = 24.0
    gap_threshold: float = 0.1
    max_gap_for_simulation: float = 24.0
    binsize: float = 0.3
    pdf_smoothing_width_days: float = 0.3
    min_flares_for_target: int = 5
    random_seed: int | None = None


@dataclass(frozen=True)
class GapInfo:
    """Observing-gap information.

    Parameters
    ----------
    indices : numpy.ndarray
        Indices where gaps begin.
    durations : numpy.ndarray
        Gap durations in days.
    starts : numpy.ndarray
        Start time of each gap in days.
    ends : numpy.ndarray
        End time of each gap in days.
    """

    indices: np.ndarray
    durations: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    """Validated flare and time-series quantities used by the analysis.

    Parameters
    ----------
    flare_times : numpy.ndarray
        Sorted flare start times in days.
    time : numpy.ndarray
        Sorted valid cadence times in days.
    time_span : float
        Full time span, including gaps, in days.
    net_observing_time : float
        Observing time after excluding long cadence gaps, in days.
    cadence_days : float
        Median cadence in days.
    mean_flaring_rate : float
        Number of flares divided by actual observed time, not by the full
        first-to-last cadence baseline.
    gaps : GapInfo
        Identified observing gaps.
    """

    flare_times: np.ndarray
    time: np.ndarray
    time_span: float
    net_observing_time: float
    cadence_days: float
    mean_flaring_rate: float
    gaps: GapInfo


def _as_float_array(values: Iterable[float], name: str) -> np.ndarray:
    """Convert values to a finite one-dimensional float array.

    Parameters
    ----------
    values : iterable of float
        Values to convert.
    name : str
        Human-readable name used in error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional finite float array.

    Raises
    ------
    ValueError
        If the input cannot be converted to a finite one-dimensional array.
    """

    try:
        arr = np.asarray(values, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} could not be converted to numeric values.") from exc

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    arr = arr[np.isfinite(arr)]
    return arr


def _require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    """Check that a table contains required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to validate.
    columns : iterable of str
        Required column names.
    table_name : str
        Human-readable table name used in error messages.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required column(s): {missing}")


def load_target_data(
    tic_id: str | int,
    base_dir: str | Path | None = None,
    flare_path: str | Path | None = None,
    timeseries_path: str | Path | None = None,
    flare_filename: str | None = None,
    timeseries_filename: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load flare and detrended time-series data for one TIC target.

    Parameters
    ----------
    tic_id : str or int
        TIC identifier.
    base_dir : str or pathlib.Path or None, optional
        Target directory. If None, ``../Data/Selas-TIC-ids/{tic_id}`` is used.
    flare_path : str or pathlib.Path or None, optional
        Explicit flare CSV path. If provided, this overrides ``flare_filename``.
    timeseries_path : str or pathlib.Path or None, optional
        Explicit time-series CSV path. If provided, this overrides
        ``timeseries_filename``.
    flare_filename : str or None, optional
        Flare CSV filename inside ``base_dir/Data``. If None,
        ``{tic_id}_flares.csv`` is used.
    timeseries_filename : str or None, optional
        Time-series CSV filename inside ``base_dir/Data``. If None,
        ``{tic_id}_detrended.csv`` is used.

    Returns
    -------
    flare_df : pandas.DataFrame
        Flare table sorted by ``tstart``.
    ts_df : pandas.DataFrame
        Detrended time-series table.
    target_dir : pathlib.Path
        Directory used as the target base directory.

    Raises
    ------
    FileNotFoundError
        If either required CSV file does not exist.
    ValueError
        If a loaded table is empty or missing required columns.
    """

    tic_id = str(tic_id)
    target_dir = Path(base_dir) if base_dir is not None else Path("../Data/Selas-TIC-ids") / tic_id
    data_dir = target_dir / "Data"

    flare_file = Path(flare_path) if flare_path is not None else data_dir / (flare_filename or f"{tic_id}_flares.csv")
    ts_file = Path(timeseries_path) if timeseries_path is not None else data_dir / (timeseries_filename or f"{tic_id}_detrended.csv")

    if not flare_file.exists():
        raise FileNotFoundError(f"Flare file not found: {flare_file}")
    if not ts_file.exists():
        raise FileNotFoundError(f"Time-series file not found: {ts_file}")

    flare_df = pd.read_csv(flare_file)
    ts_df = pd.read_csv(ts_file)

    if ts_df.empty:
        raise ValueError(f"Time-series file is empty: {ts_file}")

    _require_columns(flare_df, ["tstart"], "flare_df")
    _require_columns(ts_df, ["time"], "ts_df")

    flare_df = flare_df.copy()
    flare_df["tstart"] = pd.to_numeric(flare_df["tstart"], errors="coerce")
    flare_df = flare_df[np.isfinite(flare_df["tstart"])].sort_values("tstart").reset_index(drop=True)

    ts_df = ts_df.copy()
    ts_df["time"] = pd.to_numeric(ts_df["time"], errors="coerce")
    ts_df = ts_df[np.isfinite(ts_df["time"])].sort_values("time").reset_index(drop=True)

    if ts_df.empty:
        raise ValueError("Time-series table contains no finite time values.")

    return flare_df, ts_df, target_dir


def find_observing_gaps(time: Iterable[float], gap_threshold: float = 0.1) -> GapInfo:
    """Find observing gaps in a sorted time array.

    Parameters
    ----------
    time : iterable of float
        Cadence times in days.
    gap_threshold : float, optional
        Minimum time jump, in days, treated as a gap.

    Returns
    -------
    GapInfo
        Gap indices, durations, starts, and ends.

    Raises
    ------
    ValueError
        If ``gap_threshold`` is not positive.
    """

    if gap_threshold <= 0:
        raise ValueError("gap_threshold must be positive.")

    t = np.sort(_as_float_array(time, "time"))
    if t.size < 2:
        empty = np.array([], dtype=float)
        return GapInfo(np.array([], dtype=int), empty, empty, empty)

    dt = np.diff(t)
    indices = np.flatnonzero(dt > gap_threshold)
    return GapInfo(
        indices=indices.astype(int),
        durations=dt[indices].astype(float),
        starts=t[indices].astype(float),
        ends=t[indices + 1].astype(float),
    )


def compute_net_observing_time(time: Iterable[float]) -> tuple[float, float, float]:
    """Compute full span, actual observed time, and median cadence.

    Parameters
    ----------
    time : iterable of float
        Cadence times in days.

    Returns
    -------
    time_span : float
        Full time span, including gaps, in days.
    net_observing_time : float
        Actual observed time in days. This is the sum of cadence intervals
        smaller than 1.5 times the median cadence and therefore excludes
        baseline gaps where the target was not observed.
    cadence_days : float
        Median cadence in days.

    Raises
    ------
    ValueError
        If fewer than two finite time values are available.
    """

    t = np.sort(_as_float_array(time, "time"))
    if t.size < 2:
        raise ValueError("At least two finite time values are needed to compute observing time.")

    dt = np.diff(t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if finite_dt.size == 0:
        raise ValueError("Time array has no positive finite cadence intervals.")

    cadence_days = float(np.median(finite_dt))
    good_intervals = finite_dt[finite_dt < 1.5 * cadence_days]
    net_observing_time = float(np.sum(good_intervals))
    time_span = float(t[-1] - t[0])

    if net_observing_time <= 0:
        raise ValueError("Net observing time is not positive.")

    return time_span, net_observing_time, cadence_days


def prepare_analysis_data(
    flare_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    gap_threshold: float = 0.1,
) -> PreparedData:
    """Prepare validated flare times, cadence times, gaps, and flare rate.

    Parameters
    ----------
    flare_df : pandas.DataFrame
        Flare table containing a ``tstart`` column.
    ts_df : pandas.DataFrame
        Time-series table containing a ``time`` column.
    gap_threshold : float, optional
        Time jump, in days, above which a gap is identified.

    Returns
    -------
    PreparedData
        Validated arrays and derived observing quantities.

    Raises
    ------
    ValueError
        If required columns are missing, data are empty, or observing time is
        invalid.
    """

    _require_columns(flare_df, ["tstart"], "flare_df")
    _require_columns(ts_df, ["time"], "ts_df")

    flare_times = np.sort(_as_float_array(flare_df["tstart"], "flare_df['tstart']"))
    time = np.sort(_as_float_array(ts_df["time"], "ts_df['time']"))

    if time.size < 2:
        raise ValueError("At least two finite time values are needed.")

    time_span, net_observing_time, cadence_days = compute_net_observing_time(time)
    gaps = find_observing_gaps(time, gap_threshold=gap_threshold)
    mean_flaring_rate = float(flare_times.size / net_observing_time)

    if not np.isfinite(mean_flaring_rate) or mean_flaring_rate < 0:
        raise ValueError("Mean flaring rate is invalid.")

    return PreparedData(
        flare_times=flare_times,
        time=time,
        time_span=time_span,
        net_observing_time=net_observing_time,
        cadence_days=cadence_days,
        mean_flaring_rate=mean_flaring_rate,
        gaps=gaps,
    )


def expected_flares_in_gaps(
    gap_starts: Iterable[float],
    gap_ends: Iterable[float],
    flare_rate: float,
    max_gap: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected flare counts for eligible observing gaps.

    Parameters
    ----------
    gap_starts : iterable of float
        Gap start times in days.
    gap_ends : iterable of float
        Gap end times in days.
    flare_rate : float
        Mean flaring rate in inverse days.
    max_gap : float, optional
        Maximum gap duration, in days, filled with simulated flares.

    Returns
    -------
    starts : numpy.ndarray
        Start times of eligible gaps.
    ends : numpy.ndarray
        End times of eligible gaps.
    expected_counts : numpy.ndarray
        Expected number of flares in each eligible gap.

    Raises
    ------
    ValueError
        If starts and ends have different lengths or invalid values are passed.
    """

    starts = _as_float_array(gap_starts, "gap_starts")
    ends = _as_float_array(gap_ends, "gap_ends")

    if starts.size != ends.size:
        raise ValueError("gap_starts and gap_ends must have the same length.")
    if starts.size == 0:
        return starts, ends, np.array([], dtype=float)
    if flare_rate < 0 or not np.isfinite(flare_rate):
        raise ValueError("flare_rate must be finite and non-negative.")
    if max_gap <= 0:
        raise ValueError("max_gap must be positive.")

    durations = ends - starts
    valid = np.isfinite(durations) & (durations > 0) & (durations <= max_gap)
    starts = starts[valid]
    ends = ends[valid]
    expected_counts = flare_rate * (ends - starts)
    return starts, ends, expected_counts.astype(float)


def generate_gap_flare_times(
    gap_starts: Iterable[float],
    gap_ends: Iterable[float],
    flare_rate: float,
    max_gap: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate simulated flare times following the original notebook logic.

    Parameters
    ----------
    gap_starts : iterable of float
        Gap start times in days.
    gap_ends : iterable of float
        Gap end times in days.
    flare_rate : float
        Mean flaring rate in inverse days, computed on the net observed
        baseline.
    max_gap : float, optional
        Maximum gap duration used when computing the Poisson expectation.
    rng : numpy.random.Generator or None, optional
        Random-number generator. If None, a new default generator is used.

    Returns
    -------
    numpy.ndarray
        Simulated flare times in days. Empty if no eligible gap flares are drawn.

    Notes
    -----
    This intentionally mirrors the original notebook: expected flare counts are
    computed only for gaps shorter than ``max_gap``. The generated flare times
    then use the same gap-ordering behavior as the notebook implementation so
    that the statistics and plots match the original workflow as closely as
    possible.
    """

    rng = np.random.default_rng() if rng is None else rng
    starts_all = _as_float_array(gap_starts, "gap_starts")
    ends_all = _as_float_array(gap_ends, "gap_ends")

    if starts_all.size != ends_all.size:
        raise ValueError("gap_starts and gap_ends must have the same length.")
    if starts_all.size == 0:
        return np.array([], dtype=float)
    if flare_rate < 0 or not np.isfinite(flare_rate):
        raise ValueError("flare_rate must be finite and non-negative.")

    starts_valid, ends_valid, expected_counts = expected_flares_in_gaps(
        starts_all,
        ends_all,
        flare_rate,
        max_gap=max_gap,
    )
    _ = starts_valid, ends_valid
    if expected_counts.size == 0:
        return np.array([], dtype=float)

    counts = rng.poisson(expected_counts)

    # Notebook-equivalent behavior: zip only eligible gaps with the counts array.
    # Gaps longer than max_gap are excluded from simulated flare generation.
    chunks = [
        rng.uniform(start, end, size=int(count))
        for start, end, count in zip(starts_valid, ends_valid, counts)
        if count > 0 and np.isfinite(start) and np.isfinite(end) and end > start
    ]
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks).astype(float)


def simulate_flare_datasets(
    flare_times_observed: Iterable[float],
    gaps: GapInfo,
    flare_rate: float,
    number_of_simulations: int = 500,
    max_gap: float = 30.0,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """Create observed-plus-gap-filled flare-time datasets.

    Parameters
    ----------
    flare_times_observed : iterable of float
        Observed flare start times in days.
    gaps : GapInfo
        Observing gaps to fill with simulated flare times.
    flare_rate : float
        Mean flaring rate in inverse days.
    number_of_simulations : int, optional
        Number of datasets to generate.
    max_gap : float, optional
        Maximum gap duration, in days, used for simulated flare expectations.
    random_seed : int or None, optional
        Seed for reproducible simulations.

    Returns
    -------
    list of numpy.ndarray
        Sorted flare-time arrays, one per simulation.

    Raises
    ------
    ValueError
        If the number of simulations or flare rate is invalid.
    """

    if not isinstance(number_of_simulations, int) or number_of_simulations <= 0:
        raise ValueError("number_of_simulations must be a positive integer.")
    if flare_rate < 0 or not np.isfinite(flare_rate):
        raise ValueError("flare_rate must be finite and non-negative.")

    observed = np.sort(_as_float_array(flare_times_observed, "flare_times_observed"))
    rng = np.random.default_rng(random_seed)
    datasets: list[np.ndarray] = []

    for _ in range(number_of_simulations):
        simulated = generate_gap_flare_times(
            gaps.starts,
            gaps.ends,
            flare_rate,
            max_gap=max_gap,
            rng=rng,
        )
        if simulated.size == 0:
            combined = observed.copy()
        else:
            combined = np.sort(np.concatenate([observed, simulated]).astype(float))
        datasets.append(combined)

    return datasets


def adjacent_waiting_times(flare_times: Iterable[float], max_waiting_time: float = 24.0) -> np.ndarray:
    """Compute adjacent flare waiting times following the notebook workflow.

    Parameters
    ----------
    flare_times : iterable of float
        Flare start times in days.
    max_waiting_time : float, optional
        Maximum waiting time, in days, retained in the output.

    Returns
    -------
    numpy.ndarray
        Adjacent waiting times less than or equal to ``max_waiting_time``.

    Notes
    -----
    The original notebook subtracts the first flare time and then calls
    ``np.diff``. This is mathematically identical to taking differences of the
    sorted flare times, so this function does that directly while keeping the
    same cutoff behavior.
    """

    times = np.sort(_as_float_array(flare_times, "flare_times"))
    if times.size < 2:
        return np.array([], dtype=float)

    waiting_times = np.diff(times)
    valid = np.isfinite(waiting_times) & (waiting_times <= max_waiting_time)
    return waiting_times[valid].astype(float)


def pairwise_waiting_times(flare_times: Iterable[float], max_waiting_time: float | None = None) -> np.ndarray:
    """Compute positive pairwise flare waiting times.

    Parameters
    ----------
    flare_times : iterable of float
        Flare start times in days.
    max_waiting_time : float or None, optional
        Maximum waiting time to retain. If None, all positive pairwise waiting
        times are returned.

    Returns
    -------
    numpy.ndarray
        Positive pairwise waiting times.
    """

    times = list(_as_float_array(flare_times, "flare_times"))
    waits = [times[j] - times[i] for i in range(len(times)) for j in range(i + 1, len(times))]
    waits = np.asarray(waits, dtype=float)
    waits = waits[np.isfinite(waits) & (waits > 0)]
    if max_waiting_time is not None:
        waits = waits[waits <= max_waiting_time]
    return waits.astype(float)


def compute_waiting_time_sets(
    flare_datasets: list[np.ndarray],
    max_waiting_time: float = 24.0,
) -> tuple[list[np.ndarray], list[int]]:
    """Compute adjacent waiting times for simulated flare datasets.

    Parameters
    ----------
    flare_datasets : list of numpy.ndarray
        Flare-time arrays, one per simulation.
    max_waiting_time : float, optional
        Maximum waiting time, in days, retained for the analysis.

    Returns
    -------
    waiting_time_sets : list of numpy.ndarray
        Adjacent waiting-time arrays, one per simulation.
    flare_counts : list of int
        Number of flares in each dataset.
    """

    waiting_time_sets = [adjacent_waiting_times(dataset, max_waiting_time=max_waiting_time) for dataset in flare_datasets]
    flare_counts = [int(len(dataset)) for dataset in flare_datasets]
    return waiting_time_sets, flare_counts



def _truncated_exponential_normalization(flare_rate: float, max_waiting_time: float) -> float:
    """Return the probability mass of an exponential distribution on [0, max_waiting_time]."""

    if flare_rate <= 0 or not np.isfinite(flare_rate):
        return float("nan")
    if max_waiting_time <= 0 or not np.isfinite(max_waiting_time):
        return float("nan")
    return float(-np.expm1(-flare_rate * max_waiting_time))


def truncated_exponential_pdf(
    x: Iterable[float],
    flare_rate: float,
    max_waiting_time: float = 24.0,
) -> np.ndarray:
    """Evaluate the exponential PDF normalized on [0, max_waiting_time].

    The empirical waiting-time samples are cut at ``max_waiting_time``.  The
    matching theoretical PDF therefore has to be the conditional/truncated
    exponential PDF, not the unbounded exponential PDF.  This makes the theory
    curve integrate to one over the same 0--24 day interval as the saved PDF.
    """

    values = np.asarray(x, dtype=float)
    norm = _truncated_exponential_normalization(flare_rate, max_waiting_time)
    if not np.isfinite(norm) or norm <= 0:
        return np.full_like(values, np.nan, dtype=float)

    pdf = flare_rate * np.exp(-flare_rate * values) / norm
    pdf = np.where((values >= 0) & (values <= max_waiting_time), pdf, np.nan)
    return pdf.astype(float)


def truncated_exponential_cdf(
    x: Iterable[float],
    flare_rate: float,
    max_waiting_time: float = 24.0,
) -> np.ndarray:
    """Evaluate the exponential CDF normalized to reach one at max_waiting_time."""

    values = np.asarray(x, dtype=float)
    norm = _truncated_exponential_normalization(flare_rate, max_waiting_time)
    if not np.isfinite(norm) or norm <= 0:
        return np.full_like(values, np.nan, dtype=float)

    cdf = -np.expm1(-flare_rate * values) / norm
    cdf = np.where(values < 0, 0.0, cdf)
    cdf = np.where(values >= max_waiting_time, 1.0, cdf)
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf.astype(float)


def _gaussian_kernel1d(sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    """Create a normalized one-dimensional Gaussian kernel in bin units."""

    if sigma_bins <= 0 or not np.isfinite(sigma_bins):
        return np.array([1.0], dtype=float)

    radius = max(1, int(np.ceil(truncate * sigma_bins)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0 or not np.isfinite(kernel_sum):
        return np.array([1.0], dtype=float)
    return (kernel / kernel_sum).astype(float)


def smooth_pdf_density(
    density: Iterable[float],
    bin_width_days: float,
    smoothing_width_days: float = 1.0,
) -> np.ndarray:
    """Smooth a binned PDF while preserving unit area.

    A normal histogram PDF is very spiky when each simulation contains only a
    few adjacent waiting times.  This function smooths the binned density with a
    Gaussian kernel and renormalizes the result so that the integral over the
    saved 0--24 day grid remains one.
    """

    values = np.asarray(density, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if values.size == 0:
        return values.astype(float)
    if smoothing_width_days <= 0 or not np.isfinite(smoothing_width_days):
        return values.astype(float)
    if bin_width_days <= 0 or not np.isfinite(bin_width_days):
        raise ValueError("bin_width_days must be a positive finite value.")

    sigma_bins = float(smoothing_width_days) / float(bin_width_days)
    kernel = _gaussian_kernel1d(sigma_bins)
    smoothed = np.convolve(values, kernel, mode="same")

    area = float(np.sum(smoothed) * bin_width_days)
    if area > 0 and np.isfinite(area):
        smoothed = smoothed / area
    return smoothed.astype(float)


def prepare_pdf_statistics(
    waiting_time_sets: list[np.ndarray],
    binsize: float = 0.3,
    max_waiting_time: float = 24.0,
    pdf_smoothing_width_days: float = 1.0,
) -> pd.DataFrame:
    """Create raw and smoothed mean empirical PDFs from per-simulation histograms.

    The raw PDF is a true fixed-bin histogram averaged over simulations.  With a
    small number of waiting times per simulation this can be dominated by
    repeated observed flare separations, which appear as narrow spikes when the
    bin centers are connected by a line.  The recommended plotting column,
    ``pdf_empirical_density_smooth``, is therefore a Gaussian-smoothed version
    of the per-simulation histograms averaged over simulations.  The unsmoothed
    result is still saved as ``pdf_empirical_density_raw``.
    """

    if binsize <= 0:
        raise ValueError("binsize must be positive.")
    if max_waiting_time <= 0 or not np.isfinite(max_waiting_time):
        raise ValueError("max_waiting_time must be a positive finite value.")
    if pdf_smoothing_width_days < 0 or not np.isfinite(pdf_smoothing_width_days):
        raise ValueError("pdf_smoothing_width_days must be non-negative and finite.")

    n_bins = max(1, int(np.ceil(max_waiting_time / binsize)))
    bin_edges = np.linspace(0.0, float(max_waiting_time), n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bin_width = float(np.mean(bin_widths))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    per_sim_pdf_raw = []
    per_sim_pdf_smooth = []
    n_waiting_times_per_sim = []

    for w in waiting_time_sets:
        arr = np.asarray(w, dtype=float)
        arr = arr[np.isfinite(arr) & (arr >= 0) & (arr <= max_waiting_time)]
        if arr.size == 0:
            continue

        counts, _ = np.histogram(arr, bins=bin_edges)
        density_raw = counts / (arr.size * bin_widths)
        density_smooth = smooth_pdf_density(
            density_raw,
            bin_width_days=bin_width,
            smoothing_width_days=pdf_smoothing_width_days,
        )
        per_sim_pdf_raw.append(density_raw.astype(float))
        per_sim_pdf_smooth.append(density_smooth.astype(float))
        n_waiting_times_per_sim.append(int(arr.size))

    if not per_sim_pdf_raw:
        return pd.DataFrame()

    raw_stack = np.vstack(per_sim_pdf_raw)
    smooth_stack = np.vstack(per_sim_pdf_smooth)
    smooth_mean = np.mean(smooth_stack, axis=0)
    raw_mean = np.mean(raw_stack, axis=0)

    return pd.DataFrame({
        "waiting_time_days": x,
        "bin_left_days": bin_edges[:-1],
        "bin_right_days": bin_edges[1:],
        "pdf_empirical_density": raw_mean,
        "pdf_empirical_density_smooth": smooth_mean,
        "pdf_empirical_density_raw": raw_mean,
        "pdf_empirical_density_std": np.std(smooth_stack, axis=0),
        "pdf_empirical_density_p16": np.percentile(smooth_stack, 16, axis=0),
        "pdf_empirical_density_p84": np.percentile(smooth_stack, 84, axis=0),
        "pdf_empirical_density_raw_std": np.std(raw_stack, axis=0),
        "pdf_smoothing_width_days": float(pdf_smoothing_width_days),
        "n_simulations_used": len(per_sim_pdf_raw),
        "n_waiting_times_total": int(np.sum(n_waiting_times_per_sim)),
        "n_waiting_times_mean_per_simulation": float(np.mean(n_waiting_times_per_sim)),
    })

def prepare_pdf(
    waiting_time_sets: list[np.ndarray],
    binsize: float = 0.3,
    max_waiting_time: float = 24.0,
    pdf_smoothing_width_days: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create the mean empirical PDF from per-simulation histograms.

    Parameters
    ----------
    waiting_time_sets : list of numpy.ndarray
        Waiting-time arrays, one per simulation.
    binsize : float, optional
        Histogram bin width in days.
    max_waiting_time : float, optional
        Maximum waiting time, in days, retained in the PDF.

    Returns
    -------
    pdf : numpy.ndarray
        Mean histogram probability density values across simulations.
    x : numpy.ndarray
        Bin centers in days.
    n_waiting_times : int
        Total number of waiting-time samples used across valid simulations.
    """

    pdf_df = prepare_pdf_statistics(
        waiting_time_sets,
        binsize=binsize,
        max_waiting_time=max_waiting_time,
        pdf_smoothing_width_days=pdf_smoothing_width_days,
    )
    if pdf_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float), 0

    return (
        pdf_df["pdf_empirical_density_smooth"].to_numpy(dtype=float),
        pdf_df["waiting_time_days"].to_numpy(dtype=float),
        int(pdf_df["n_waiting_times_total"].iloc[0]),
    )


def prepare_cdf(
    waiting_time_sets: list[np.ndarray],
    flare_rate: float,
    n_points: int = 500,
    max_waiting_time: float = 24.0,
) -> pd.DataFrame:
    """Create the numerical data used for the mean CDF plot.

    The theoretical CDF is the truncated exponential CDF on the same interval as
    the empirical waiting-time data.  Therefore ``cdf_theory`` is exactly one at
    ``max_waiting_time``.
    """

    if n_points <= 1:
        raise ValueError("n_points must be greater than one.")
    if max_waiting_time <= 0 or not np.isfinite(max_waiting_time):
        raise ValueError("max_waiting_time must be a positive finite value.")
    if flare_rate <= 0 or not np.isfinite(flare_rate):
        return pd.DataFrame()

    arrays = []
    for w in waiting_time_sets:
        arr = np.sort(np.asarray(w, dtype=float))
        arr = arr[np.isfinite(arr) & (arr >= 0) & (arr <= max_waiting_time)]
        if arr.size > 0:
            arrays.append(arr)
    if not arrays:
        return pd.DataFrame()

    t = np.linspace(0.0, float(max_waiting_time), int(n_points))
    empirical = np.vstack([
        np.searchsorted(w, t, side="right") / len(w)
        for w in arrays
    ])
    cdf_theory = truncated_exponential_cdf(t, flare_rate, max_waiting_time=max_waiting_time)
    if cdf_theory.size > 0:
        cdf_theory[-1] = 1.0

    return pd.DataFrame({
        "waiting_time_days": t,
        "cdf_theory": cdf_theory,
        "cdf_empirical_mean": np.mean(empirical, axis=0),
        "cdf_empirical_std": np.std(empirical, axis=0),
        "cdf_empirical_p16": np.percentile(empirical, 16, axis=0),
        "cdf_empirical_p84": np.percentile(empirical, 84, axis=0),
        "n_simulations_used": len(arrays),
    })


def prepare_pdf_data_table(
    waiting_time_sets: list[np.ndarray],
    flare_rate: float,
    binsize: float = 0.3,
    max_waiting_time: float = 24.0,
    pdf_smoothing_width_days: float = 1.0,
) -> pd.DataFrame:
    """Create and save-ready PDF data from per-simulation mean histograms."""

    if flare_rate <= 0 or not np.isfinite(flare_rate):
        return pd.DataFrame()

    pdf_df = prepare_pdf_statistics(
        waiting_time_sets,
        binsize=binsize,
        max_waiting_time=max_waiting_time,
        pdf_smoothing_width_days=pdf_smoothing_width_days,
    )
    if pdf_df.empty:
        return pd.DataFrame()

    x = pdf_df["waiting_time_days"].to_numpy(dtype=float)
    fitted = truncated_exponential_pdf(x, flare_rate, max_waiting_time=max_waiting_time)
    fitted_std = float(np.nanstd(fitted))

    pdf_df = pdf_df.copy()
    pdf_df["pdf_theory_density"] = fitted
    pdf_df["pdf_theory_normalization_on_0_to_max_waiting_time"] = _truncated_exponential_normalization(
        flare_rate,
        max_waiting_time,
    )
    pdf_df["pdf_theory_std_used_for_bands"] = fitted_std

    for k in (1, 2, 3):
        pdf_df[f"pdf_theory_minus_{k}sigma"] = fitted - k * fitted_std
        pdf_df[f"pdf_theory_plus_{k}sigma"] = fitted + k * fitted_std

    return pdf_df


def save_pdf_cdf_data(
    pdf: np.ndarray,
    x: np.ndarray,
    waiting_time_sets: list[np.ndarray],
    flare_rate: float,
    results_dir: str | Path,
    tic_id: str | int,
    cdf_n_points: int = 500,
    max_waiting_time: float = 24.0,
    binsize: float = 0.3,
    pdf_smoothing_width_days: float = 1.0,
) -> dict[str, str | None]:
    """Save PDF and CDF numerical data tables to CSV files.

    ``pdf`` and ``x`` are retained in the signature for backward compatibility,
    but the saved PDF table is recomputed from ``waiting_time_sets`` as a mean of
    per-simulation histograms on fixed 0--``max_waiting_time`` day bins.
    """

    _ = pdf, x
    if max_waiting_time <= 0 or not np.isfinite(max_waiting_time):
        raise ValueError("max_waiting_time must be a positive finite value.")

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path: str | None = None
    cdf_path: str | None = None

    pdf_df = prepare_pdf_data_table(
        waiting_time_sets,
        flare_rate,
        binsize=binsize,
        max_waiting_time=max_waiting_time,
        pdf_smoothing_width_days=pdf_smoothing_width_days,
    )
    if not pdf_df.empty:
        path = output_dir / f"{tic_id}_pdf_data.csv"
        pdf_df.to_csv(path, index=False)
        pdf_path = str(path)

    cdf_df = prepare_cdf(waiting_time_sets, flare_rate, n_points=cdf_n_points, max_waiting_time=max_waiting_time)
    if not cdf_df.empty:
        path = output_dir / f"{tic_id}_cdf_data.csv"
        cdf_df.to_csv(path, index=False)
        cdf_path = str(path)

    return {"pdf_data_path": pdf_path, "cdf_data_path": cdf_path}


def run_ks_tests(
    waiting_time_sets: list[np.ndarray],
    flare_rate: float,
    max_waiting_time: float = 24.0,
) -> tuple[list[float], list[float], str | None]:
    """Run one-sample KS tests against the truncated exponential model."""

    if kstest is None:
        return [], [], f"scipy.stats.kstest is unavailable: {_SCIPY_IMPORT_ERROR}"
    if flare_rate <= 0 or not np.isfinite(flare_rate):
        return [], [], "mean flaring rate is not positive"
    if max_waiting_time <= 0 or not np.isfinite(max_waiting_time):
        return [], [], "max waiting time is not positive"

    def cdf_theory(values: np.ndarray, lam: float = flare_rate, limit: float = max_waiting_time) -> np.ndarray:
        return truncated_exponential_cdf(values, lam, max_waiting_time=limit)

    D_values: list[float] = []
    p_values: list[float] = []

    for waiting_times in waiting_time_sets:
        valid = np.asarray(waiting_times, dtype=float)
        valid = valid[np.isfinite(valid) & (valid >= 0) & (valid <= max_waiting_time)]
        if valid.size == 0:
            continue
        D, p_value = kstest(np.sort(valid), cdf_theory)
        if np.isfinite(D) and np.isfinite(p_value):
            D_values.append(float(D))
            p_values.append(float(p_value))

    if not D_values:
        return [], [], "no valid waiting times in every dataset"
    return D_values, p_values, None


def classify_target(
    n_flares: int,
    p_value_median: float | None,
    D_value_median: float | None,
    min_flares: int = 20,
) -> bool:
    """Apply the notebook target-selection heuristic.

    Parameters
    ----------
    n_flares : int
        Number of observed flares.
    p_value_median : float or None
        Median KS p-value.
    D_value_median : float or None
        Median KS statistic.
    min_flares : int, optional
        Minimum number of flares required before a target can be selected.

    Returns
    -------
    bool
        True if the target passes the heuristic, otherwise False.
    """

    if n_flares <= min_flares or p_value_median is None or D_value_median is None:
        return False
    if not np.isfinite(p_value_median) or not np.isfinite(D_value_median):
        return False

    return bool(
        (p_value_median < 10**-0.2 and D_value_median > 0.2)
        or (p_value_median < 10**-0.5 and D_value_median > 0.1)
    )


def _safe_savefig(fig, path: Path) -> None:
    """Save a Matplotlib figure as a thesis-style PDF and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : pathlib.Path
        Output PDF path.
    """

    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _apply_scalar_formatter(ax, format_x: bool = True, format_y: bool = True) -> None:
    """Disable scientific and offset notation on numerical axes."""

    if ScalarFormatter is None:
        return

    if format_x:
        formatter_x = ScalarFormatter(useOffset=False)
        formatter_x.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter_x)

    if format_y:
        formatter_y = ScalarFormatter(useOffset=False)
        formatter_y.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter_y)


def _style_axis(
    ax,
    panel_label: str | None = None,
    format_x: bool = True,
    format_y: bool = True,
    hide_x_ticklabels: bool = False,
) -> None:
    """Apply common thesis-style ticks, spines, axis formatting, and labels."""

    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH)

    ax.tick_params(
        axis="both",
        which="major",
        width=LINEWIDTH,
        length=7 * SCALE,
        direction="out",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=LINEWIDTH * 0.8,
        length=4 * SCALE,
        direction="out",
    )
    ax.minorticks_on()
    _apply_scalar_formatter(ax, format_x=format_x, format_y=format_y)

    if hide_x_ticklabels:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    if panel_label is not None:
        ax.text(
            0.025,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            zorder=100,
        )


def _legend_if_needed(ax, loc: str = "best") -> None:
    """Add a clean legend only when labeled artists are present."""

    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if not keep:
        return
    handles, labels = zip(*keep)
    legend = ax.legend(handles, labels, loc=loc, frameon=False)
    for line in legend.get_lines():
        line.set_linewidth(LINEWIDTH)


def plot_waiting_time_distribution(
    flare_times: Iterable[float],
    results_dir: str | Path,
    tic_id: str | int,
    waiting_time_limit: float = 24.0,
) -> Path | None:
    """Plot adjacent and pairwise observed waiting-time histograms in thesis style.

    Parameters
    ----------
    flare_times : iterable of float
        Observed flare times in days.
    results_dir : str or pathlib.Path
        Directory where the plot is saved.
    tic_id : str or int
        TIC identifier used in the filename.
    waiting_time_limit : float, optional
        Maximum adjacent waiting time, in days, used for the first histogram.

    Returns
    -------
    pathlib.Path or None
        Saved figure path, or None if no data could be plotted.
    """

    if plt is None:
        return None

    adjacent = adjacent_waiting_times(flare_times, max_waiting_time=waiting_time_limit)
    adjacent = adjacent[adjacent > 0]
    stacked = pairwise_waiting_times(flare_times)
    stacked = stacked[stacked <= waiting_time_limit]

    if adjacent.size == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(24, 7), sharey=True)
    bins = np.linspace(0.0, float(waiting_time_limit), 101)
    if np.unique(bins).size < 2:
        bins = 1

    hist_kwargs = {
        "color": THESIS_RAW_COLOR,
        "edgecolor": "black",
        "alpha": THESIS_RAW_ALPHA,
        "linewidth": LINEWIDTH * 0.35,
    }

    axes[0].hist(adjacent, bins=bins, **hist_kwargs)
    axes[0].set_xlim(0, waiting_time_limit)
    axes[0].set_xlabel("Waiting time [day]")
    axes[0].set_ylabel("Adjacent frequency")

    if stacked.size > 0:
        axes[1].hist(stacked, bins=bins, **hist_kwargs)
    axes[1].set_xlim(0, waiting_time_limit)
    axes[1].set_xlabel("Waiting time [day]")
    axes[1].set_ylabel("Pairwise frequency")

    for label, ax in zip(("A", "B"), axes):
        _style_axis(ax, panel_label=label)

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.96, wspace=0.20)

    path = Path(results_dir) / f"{tic_id}_wtd.pdf"
    _safe_savefig(fig, path)
    return path


def plot_mean_cdf(
    waiting_time_sets: list[np.ndarray],
    flare_rate: float,
    results_dir: str | Path,
    tic_id: str | int,
    max_waiting_time: float = 24.0,
) -> Path | None:
    """Plot the mean empirical CDF and truncated exponential model CDF."""

    if plt is None or flare_rate <= 0 or not np.isfinite(flare_rate):
        return None

    cdf_df = prepare_cdf(
        waiting_time_sets,
        flare_rate,
        n_points=500,
        max_waiting_time=max_waiting_time,
    )
    if cdf_df.empty:
        return None

    t = cdf_df["waiting_time_days"].to_numpy(dtype=float)
    cdf_theory = cdf_df["cdf_theory"].to_numpy(dtype=float)
    mean_empirical = cdf_df["cdf_empirical_mean"].to_numpy(dtype=float)
    p16 = cdf_df["cdf_empirical_p16"].to_numpy(dtype=float)
    p84 = cdf_df["cdf_empirical_p84"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(t, p16, p84, color=THESIS_RAW_COLOR, alpha=0.25, linewidth=0)
    ax.plot(
        t,
        mean_empirical,
        label=f"Mean empirical CDF ({int(cdf_df['n_simulations_used'].iloc[0])} simulations)",
        color=THESIS_MAIN_COLOR,
        lw=LINEWIDTH,
    )
    ax.plot(
        t,
        cdf_theory,
        label=f"Truncated exponential model (0--{max_waiting_time:g} d)",
        color=THESIS_HIGHLIGHT_COLOR,
        lw=LINEWIDTH,
    )

    ax.set_xlim(0, max_waiting_time)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Waiting Time [day]")
    ax.set_ylabel("Cumulative Density")
    _style_axis(ax, panel_label="A")
    _legend_if_needed(ax, loc="lower right")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.96)

    path = Path(results_dir) / f"{tic_id}_mean_cdf.pdf"
    _safe_savefig(fig, path)
    return path


def _add_pdf_error_bands(x: np.ndarray, function: np.ndarray, function_std: float, ax) -> None:
    """Add thesis-style model spread bands to a PDF axis.

    Parameters
    ----------
    x : numpy.ndarray
        X values.
    function : numpy.ndarray
        Function values.
    function_std : float
        Standard deviation used for bands.
    ax : matplotlib.axes.Axes
        Axis to modify.
    """

    if not np.isfinite(function_std):
        return

    band_alphas = {1: 0.18, 2: 0.11, 3: 0.06}
    for k in (3, 2, 1):
        lower = function - k * function_std
        upper = function + k * function_std
        ax.fill_between(
            x,
            lower,
            upper,
            color=THESIS_HIGHLIGHT_COLOR,
            alpha=band_alphas[k],
            linewidth=0,
            zorder=1,
        )


def plot_pdf(
    pdf: np.ndarray,
    x: np.ndarray,
    flare_rate: float,
    results_dir: str | Path,
    tic_id: str | int,
    max_waiting_time: float = 24.0,
) -> Path | None:
    """Plot the mean empirical waiting-time PDF and truncated theory PDF."""

    if plt is None or len(pdf) == 0 or len(x) == 0 or flare_rate <= 0 or not np.isfinite(flare_rate):
        return None

    pdf = np.asarray(pdf, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(pdf) & np.isfinite(x) & (x >= 0) & (x <= max_waiting_time)
    pdf = pdf[mask]
    x = x[mask]
    if pdf.size == 0 or x.size == 0:
        return None

    fitted = truncated_exponential_pdf(x, flare_rate, max_waiting_time=max_waiting_time)
    fitted_std = float(np.nanstd(fitted))

    fig, ax = plt.subplots(figsize=(12, 7))
    _add_pdf_error_bands(x, fitted, fitted_std, ax)
    ax.plot(x, pdf, color=THESIS_MAIN_COLOR, label="Smoothed mean empirical PDF", lw=LINEWIDTH, zorder=3)
    ax.plot(
        x,
        fitted,
        color=THESIS_HIGHLIGHT_COLOR,
        label=f"Truncated exponential model (0--{max_waiting_time:g} d)",
        lw=LINEWIDTH,
        zorder=4,
    )

    ax.set_xlim(0, max_waiting_time)
    ymax = np.nanmax([np.nanmax(pdf), np.nanmax(fitted + 3 * fitted_std)])
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, 1.1 * ymax)
    ax.set_xlabel("Waiting Time [day]")
    ax.set_ylabel("Probability Density")
    _style_axis(ax, panel_label="A")
    _legend_if_needed(ax, loc="upper right")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.96)

    path = Path(results_dir) / f"{tic_id}_pdf.pdf"
    _safe_savefig(fig, path)
    return path


def plot_ks_results(
    D_values: Iterable[float],
    p_values: Iterable[float],
    results_dir: str | Path,
    tic_id: str | int,
) -> Path | None:
    """Plot KS D statistics, log p-values, and their scatter relation.

    Parameters
    ----------
    D_values : iterable of float
        KS D statistics.
    p_values : iterable of float
        KS p-values.
    results_dir : str or pathlib.Path
        Directory where the plot is saved.
    tic_id : str or int
        TIC identifier used in the filename.

    Returns
    -------
    pathlib.Path or None
        Saved figure path, or None if no valid values could be plotted.
    """

    if plt is None:
        return None

    D = _as_float_array(D_values, "D_values")
    p = _as_float_array(p_values, "p_values")
    mask = np.isfinite(D) & np.isfinite(p) & (p > 0)
    D = D[mask]
    p = p[mask]
    if D.size == 0:
        return None

    fig, ax = plt.subplots(1, 3, figsize=(36, 7))

    median_D = float(np.median(D))
    std_D = float(np.std(D))
    median_p = float(np.median(p))
    median_logp = float(np.log10(median_p))
    log_p = np.log10(p)
    std_logp = float(np.std(log_p))

    hist_kwargs = {
        "color": THESIS_RAW_COLOR,
        "alpha": THESIS_RAW_ALPHA,
        "edgecolor": "black",
        "linewidth": LINEWIDTH * 0.35,
        "zorder": 2,
    }
    span_alphas = {1: 0.16, 2: 0.10, 3: 0.06}

    ax[0].hist(D, bins=30, **hist_kwargs)
    for k in (3, 2, 1):
        ax[0].axvspan(
            median_D - k * std_D,
            median_D + k * std_D,
            color=THESIS_HIGHLIGHT_COLOR,
            alpha=span_alphas[k],
            linewidth=0,
            zorder=1,
        )
    ax[0].axvline(median_D, color=THESIS_HIGHLIGHT_COLOR, linestyle="--", lw=LINEWIDTH, zorder=3)
    ax[0].set_xlabel("D statistic")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(log_p, bins=30, **hist_kwargs)
    for k in (3, 2, 1):
        ax[1].axvspan(
            median_logp - k * std_logp,
            median_logp + k * std_logp,
            color=THESIS_HIGHLIGHT_COLOR,
            alpha=span_alphas[k],
            linewidth=0,
            zorder=1,
        )
    ax[1].axvline(median_logp, color=THESIS_HIGHLIGHT_COLOR, linestyle="--", lw=LINEWIDTH, zorder=3)
    ax[1].set_xlabel("log10(p)")
    ax[1].set_ylabel("Frequency")
    ax[1].invert_xaxis()

    for k in (3, 2, 1):
        ax[2].axvspan(
            median_D - k * std_D,
            median_D + k * std_D,
            color=THESIS_HIGHLIGHT_COLOR,
            alpha=span_alphas[k],
            linewidth=0,
            zorder=1,
        )
        ax[2].axhspan(
            median_logp - k * std_logp,
            median_logp + k * std_logp,
            color=THESIS_HIGHLIGHT_COLOR,
            alpha=span_alphas[k],
            linewidth=0,
            zorder=1,
        )
    ax[2].scatter(
        D,
        log_p,
        color=THESIS_RAW_COLOR,
        alpha=THESIS_RAW_ALPHA,
        edgecolors="none",
        s=45 * SCALE,
        zorder=3,
    )
    ax[2].scatter(
        median_D,
        median_logp,
        color=THESIS_HIGHLIGHT_COLOR,
        edgecolor="black",
        linewidth=LINEWIDTH * 0.35,
        s=160 * SCALE,
        zorder=4,
    )
    ax[2].set_xlabel("D statistic")
    ax[2].set_ylabel("log10(p)")

    for label, axis in zip(("A", "B", "C"), ax):
        _style_axis(axis, panel_label=label)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.96, wspace=0.28)

    path = Path(results_dir) / f"{tic_id}_ks_test.pdf"
    _safe_savefig(fig, path)
    return path


def print_run_summary(summary: dict[str, object]) -> None:
    """Print a log-friendly summary block for one TIC run.

    Parameters
    ----------
    summary : dict
        Summary values returned by ``run_waiting_time_statistics``.
    """

    sep = "=" * 65

    def raw(key: str, default: object = None) -> object:
        return summary.get(key, default)

    def is_missing(val: object) -> bool:
        if val is None:
            return True
        try:
            return bool(pd.isna(val))
        except Exception:
            return False

    def fmt_float(key: str, suffix: str = "", precision: int = 2) -> str:
        val = raw(key)
        if is_missing(val):
            return "n/a"
        try:
            val = float(val)
        except Exception:
            return "n/a"
        if not np.isfinite(val):
            return "n/a"
        return f"{val:.{precision}f}{suffix}"

    def fmt_sci(key: str, suffix: str = "") -> str:
        val = raw(key)
        if is_missing(val):
            return "n/a"
        try:
            val = float(val)
        except Exception:
            return "n/a"
        if not np.isfinite(val):
            return "n/a"
        return f"{val:.2e}{suffix}"

    def fmt_int(key: str) -> str:
        val = raw(key)
        if is_missing(val):
            return "n/a"
        try:
            return f"{int(val):,}"
        except Exception:
            return "n/a"

    def fmt_bool_yes_no(key: str) -> str:
        val = raw(key)
        if is_missing(val):
            return "n/a"
        return "yes" if bool(val) else "no"

    def fmt_plain(key: str) -> str:
        val = raw(key)
        if is_missing(val):
            return "n/a"
        return str(val)

    def fmt_skip_reason() -> str:
        val = raw("skip_reason")
        if is_missing(val):
            return "none"
        return str(val)

    ks_d_mean = fmt_sci("KS_D_mean")
    ks_d_median = fmt_sci("KS_D_median")
    ks_p_mean = fmt_sci("KS_p_mean")
    ks_p_median = fmt_sci("KS_p_median")

    print(sep)
    print(f"  Cadence                     : {fmt_float('cadence_minutes', ' min')}")
    print(f"  Net observing time          : {fmt_float('net_observing_time_days', ' d')}")
    print(f"  Flares detected             : {fmt_int('n_flares')}")
    print(f"  Mean flaring rate           : {fmt_float('mean_flaring_rate', ' 1/d')}")
    print(f"    Rate denominator          : observed time, gaps excluded")
    print(f"  Simulations                 : {fmt_int('number_of_simulations')}")
    print(f"  Gap threshold               : {fmt_float('gap_threshold_days', ' d')}")
    print(f"  Gaps detected               : {fmt_int('n_gaps')}")
    print(f"  Eligible simulated gaps     : {fmt_int('n_eligible_gaps')}")
    print(f"  Expected gap flares total   : {fmt_sci('expected_gap_flares_total')}")
    print(f"  Max gap filled              : {fmt_float('max_gap_for_simulation_days', ' d')}")
    print(f"  Waiting-time cutoff         : {fmt_float('waiting_time_limit_days', ' d')}")
    print(f"  PDF binsize                 : {fmt_float('binsize_days', ' d')}")
    print(f"  PDF smoothing width        : {fmt_float('pdf_smoothing_width_days', ' d')}")
    print(f"  Waiting times               : {fmt_int('n_waiting_times')}")
    print(f"  Waiting times below cutoff  : {fmt_int('n_waiting_times_below_limit')}")
    print(f"  PDF waiting times           : {fmt_int('n_pdf_waiting_times')}")
    print(f"  Plots requested             : {fmt_bool_yes_no('make_plots')}")
    print(f"  KS testing performed        : {fmt_bool_yes_no('ks_performed')}")
    print(f"  KS D mean / median          : {ks_d_mean} / {ks_d_median}")
    print(f"  KS p mean / median          : {ks_p_mean} / {ks_p_median}")
    print(f"  Skip reason                 : {fmt_skip_reason()}")
    print(f"  Target flag                 : {fmt_plain('target_flag')}")
    print(f"  Summary CSV                 : {fmt_plain('summary_path')}")
    print(f"  PDF data CSV                : {fmt_plain('pdf_data_path')}")
    print(f"  CDF data CSV                : {fmt_plain('cdf_data_path')}")
    print(sep)

def _empty_summary(
    tic_id: str | int,
    target_dir: Path,
    results_dir: Path,
    config: AnalysisConfig,
    make_plots: bool,
    skip_reason: str,
) -> dict[str, object]:
    """Build a summary dictionary for a skipped run.

    Parameters
    ----------
    tic_id : str or int
        TIC identifier.
    target_dir : pathlib.Path
        Target base directory.
    results_dir : pathlib.Path
        Results directory.
    config : AnalysisConfig
        Analysis configuration.
    make_plots : bool
        Whether plots were requested.
    skip_reason : str
        Reason the run was skipped.

    Returns
    -------
    dict
        Summary dictionary with missing numeric values represented by None.
    """

    return {
        "TIC_id": str(tic_id),
        "target_dir": str(target_dir),
        "results_dir": str(results_dir),
        "time_span_days": None,
        "net_observing_time_days": None,
        "mean_flaring_rate_basis": "observed_time_gaps_excluded",
        "n_cadences": 0,
        "cadence_minutes": None,
        "n_flares": 0,
        "mean_flaring_rate": None,
        "n_gaps": 0,
        "n_eligible_gaps": 0,
        "expected_gap_flares_total": None,
        "number_of_simulations": config.number_of_simulations,
        "gap_threshold_days": config.gap_threshold,
        "max_gap_for_simulation_days": config.max_gap_for_simulation,
        "waiting_time_limit_days": config.waiting_time_limit,
        "binsize_days": config.binsize,
        "pdf_smoothing_width_days": config.pdf_smoothing_width_days,
        "n_waiting_times": 0,
        "n_waiting_times_below_limit": 0,
        "n_pdf_waiting_times": 0,
        "KS_D_mean": None,
        "KS_p_mean": None,
        "KS_D_median": None,
        "KS_p_median": None,
        "ks_performed": False,
        "skip_reason": skip_reason,
        "target_flag": False,
        "make_plots": bool(make_plots),
        "pdf_data_path": None,
        "cdf_data_path": None,
        "summary_path": None,
    }


def run_waiting_time_statistics(
    tic_id: str | int,
    base_dir: str | Path | None = None,
    flare_path: str | Path | None = None,
    timeseries_path: str | Path | None = None,
    config: AnalysisConfig | None = None,
    make_plots: bool = True,
    save_summary: bool = True,
    save_simulated_flares: bool = True,
    save_distribution_data: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Run the full waiting-time PDF/CDF/KS analysis for one TIC target.

    Parameters
    ----------
    tic_id : str or int
        TIC identifier.
    base_dir : str or pathlib.Path or None, optional
        Target directory. If None, ``../Data/Selas-TIC-ids/{tic_id}`` is used.
    flare_path : str or pathlib.Path or None, optional
        Explicit flare CSV path. If None, the default target-data path is used.
    timeseries_path : str or pathlib.Path or None, optional
        Explicit detrended time-series CSV path. If None, the default
        target-data path is used.
    config : AnalysisConfig or None, optional
        Analysis configuration. If None, defaults are used.
    make_plots : bool, optional
        If True, diagnostic plots are created when valid data exist.
    save_summary : bool, optional
        If True, write the one-row summary CSV.
    save_simulated_flares : bool, optional
        If True, write simulated flare-time datasets to CSV.
    save_distribution_data : bool, optional
        If True, write the numerical PDF and CDF data tables to CSV.
    verbose : bool, optional
        If True, print a log-friendly run summary.

    Returns
    -------
    dict
        One-row summary of the analysis, including skip reasons for invalid or
        insufficient data.

    Raises
    ------
    ValueError
        If configuration values are invalid.
    """

    config = AnalysisConfig() if config is None else config
    tic_id = str(tic_id)
    target_dir = Path(base_dir) if base_dir is not None else Path("../Data/Selas-TIC-ids") / tic_id
    results_dir = target_dir / "Results" / "Waiting_time_statistics"
    results_dir.mkdir(parents=True, exist_ok=True)

    if config.number_of_simulations <= 0:
        raise ValueError("config.number_of_simulations must be positive.")
    if config.waiting_time_limit <= 0:
        raise ValueError("config.waiting_time_limit must be positive.")
    if config.pdf_smoothing_width_days < 0 or not np.isfinite(config.pdf_smoothing_width_days):
        raise ValueError("config.pdf_smoothing_width_days must be non-negative and finite.")

    try:
        flare_df, ts_df, target_dir = load_target_data(
            tic_id=tic_id,
            base_dir=target_dir,
            flare_path=flare_path,
            timeseries_path=timeseries_path,
        )
        results_dir = target_dir / "Results" / "Waiting_time_statistics"
        results_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_analysis_data(flare_df, ts_df, gap_threshold=config.gap_threshold)
    except Exception as exc:
        summary = _empty_summary(tic_id, target_dir, results_dir, config, make_plots, skip_reason=str(exc))
        if save_summary:
            summary_path = results_dir / f"{tic_id}_summary.csv"
            pd.DataFrame([summary]).to_csv(summary_path, index=False)
            summary["summary_path"] = str(summary_path)
        if verbose:
            print_run_summary(summary)
        return summary

    observed_waiting_times = adjacent_waiting_times(prepared.flare_times, max_waiting_time=config.waiting_time_limit)
    all_observed_waiting_times = adjacent_waiting_times(prepared.flare_times, max_waiting_time=np.inf)

    starts, ends, expected_counts = expected_flares_in_gaps(
        prepared.gaps.starts,
        prepared.gaps.ends,
        prepared.mean_flaring_rate,
        max_gap=config.max_gap_for_simulation,
    )

    if prepared.flare_times.size < 2:
        ks_skip_reason = "fewer than two observed flares"
        flare_datasets = [prepared.flare_times.copy() for _ in range(config.number_of_simulations)]
        waiting_time_sets = [np.array([], dtype=float) for _ in range(config.number_of_simulations)]
        flare_counts = [int(prepared.flare_times.size)] * config.number_of_simulations
        D_values, p_values = [], []
    else:
        flare_datasets = simulate_flare_datasets(
            prepared.flare_times,
            prepared.gaps,
            prepared.mean_flaring_rate,
            number_of_simulations=config.number_of_simulations,
            max_gap=config.max_gap_for_simulation,
            random_seed=config.random_seed,
        )
        waiting_time_sets, flare_counts = compute_waiting_time_sets(
            flare_datasets,
            max_waiting_time=config.waiting_time_limit,
        )
        D_values, p_values, ks_skip_reason = run_ks_tests(
            waiting_time_sets,
            prepared.mean_flaring_rate,
            max_waiting_time=config.waiting_time_limit,
        )

    if save_simulated_flares:
        simulated_path = target_dir / "Data" / "simulated_flares.csv"
        simulated_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flare_datasets).to_csv(simulated_path, index=False)

    pdf, x, n_pdf_waiting_times = prepare_pdf(
        waiting_time_sets,
        binsize=config.binsize,
        max_waiting_time=config.waiting_time_limit,
        pdf_smoothing_width_days=config.pdf_smoothing_width_days,
    )

    if save_distribution_data:
        distribution_data_paths = save_pdf_cdf_data(
            pdf,
            x,
            waiting_time_sets,
            prepared.mean_flaring_rate,
            results_dir,
            tic_id,
            max_waiting_time=config.waiting_time_limit,
            binsize=config.binsize,
            pdf_smoothing_width_days=config.pdf_smoothing_width_days,
        )
    else:
        distribution_data_paths = {"pdf_data_path": None, "cdf_data_path": None}

    plot_paths: list[str] = []
    if make_plots:
        for path in (
            plot_waiting_time_distribution(prepared.flare_times, results_dir, tic_id, config.waiting_time_limit),
            plot_mean_cdf(
                waiting_time_sets,
                prepared.mean_flaring_rate,
                results_dir,
                tic_id,
                max_waiting_time=config.waiting_time_limit,
            ),
            plot_pdf(
                pdf,
                x,
                prepared.mean_flaring_rate,
                results_dir,
                tic_id,
                max_waiting_time=config.waiting_time_limit,
            ),
            plot_ks_results(D_values, p_values, results_dir, tic_id),
        ):
            if path is not None:
                plot_paths.append(str(path))

    KS_D_mean = float(np.mean(D_values)) if D_values else None
    KS_p_mean = float(np.mean(p_values)) if p_values else None
    KS_D_median = float(np.median(D_values)) if D_values else None
    KS_p_median = float(np.median(p_values)) if p_values else None
    ks_performed = bool(D_values)

    target_flag = classify_target(
        int(prepared.flare_times.size),
        KS_p_median,
        KS_D_median,
        min_flares=config.min_flares_for_target,
    )

    if observed_waiting_times.size == 0 and prepared.flare_times.size >= 2:
        skip_reason = f"no adjacent waiting times <= {config.waiting_time_limit:g} days"
    else:
        skip_reason = ks_skip_reason

    summary: dict[str, object] = {
        "TIC_id": tic_id,
        "target_dir": str(target_dir),
        "results_dir": str(results_dir),
        "time_span_days": prepared.time_span,
        "net_observing_time_days": prepared.net_observing_time,
        "mean_flaring_rate_basis": "observed_time_gaps_excluded",
        "n_cadences": int(prepared.time.size),
        "cadence_minutes": prepared.cadence_days * 24.0 * 60.0,
        "n_flares": int(prepared.flare_times.size),
        "mean_flaring_rate": prepared.mean_flaring_rate,
        "n_gaps": int(prepared.gaps.durations.size),
        "n_eligible_gaps": int(expected_counts.size),
        "expected_gap_flares_total": float(np.sum(expected_counts)) if expected_counts.size else 0.0,
        "number_of_simulations": config.number_of_simulations,
        "gap_threshold_days": config.gap_threshold,
        "max_gap_for_simulation_days": config.max_gap_for_simulation,
        "waiting_time_limit_days": config.waiting_time_limit,
        "binsize_days": config.binsize,
        "pdf_smoothing_width_days": config.pdf_smoothing_width_days,
        "n_waiting_times": int(all_observed_waiting_times.size),
        "n_waiting_times_below_limit": int(observed_waiting_times.size),
        "n_pdf_waiting_times": int(n_pdf_waiting_times),
        "KS_D_mean": KS_D_mean,
        "KS_p_mean": KS_p_mean,
        "KS_D_median": KS_D_median,
        "KS_p_median": KS_p_median,
        "ks_performed": ks_performed,
        "skip_reason": None if ks_performed else skip_reason,
        "target_flag": target_flag,
        "make_plots": bool(make_plots),
        "plot_paths": plot_paths,
        "pdf_data_path": distribution_data_paths.get("pdf_data_path"),
        "cdf_data_path": distribution_data_paths.get("cdf_data_path"),
        "summary_path": None,
    }

    if save_summary:
        summary_path = results_dir / f"{tic_id}_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        summary["summary_path"] = str(summary_path)

    if verbose:
        print_run_summary(summary)

    return summary


# Backwards-compatible alias for the original notebook function name.
def all_waiting_time_statistics(
    TIC_id: str | int,
    manual_inspection: bool = False,
    output: bool = True,
    number_of_simulations: int = 500,
) -> dict[str, object]:
    """Run waiting-time statistics using the original notebook function name.

    Parameters
    ----------
    TIC_id : str or int
        TIC identifier.
    manual_inspection : bool, optional
        Retained for compatibility. It is not used by this refactored workflow.
    output : bool, optional
        If True, diagnostic plots are created when valid data exist.
    number_of_simulations : int, optional
        Number of simulated flare-time datasets to create.

    Returns
    -------
    dict
        Summary dictionary returned by ``run_waiting_time_statistics``.

    Notes
    -----
    This wrapper keeps old notebook calls working while routing all logic through
    ``run_waiting_time_statistics``.
    """

    _ = manual_inspection
    config = AnalysisConfig(number_of_simulations=number_of_simulations)
    return run_waiting_time_statistics(TIC_id, config=config, make_plots=output)
