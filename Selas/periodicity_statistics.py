"""Reusable Rayleigh-periodicity workflow for TIC flare data.

Cleaned refactor of the original periodicity notebook.  Scientific steps are
preserved; input validation, graceful skip paths, and isolated plotting helpers
have been added or improved.
"""

from __future__ import annotations

import ast
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from tqdm import tqdm

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(suppress=True, precision=6)

# Thesis-style plotting constants used by every figure.
BASE_FONTSIZE = 20
BASE_LINEWIDTH = 2.2
SCALE = 1.8
FONTSIZE = BASE_FONTSIZE * SCALE
LINEWIDTH = BASE_LINEWIDTH * SCALE

# Single-panel figures use the thesis reference size. Multi-panel figures scale
# their total width/height so each panel keeps the same visual scale.
DEFAULT_FIGSIZE = (12, 7)
TWO_PANEL_FIGSIZE = (24, 7)
VERTICAL_TWO_PANEL_FIGSIZE = (12, 14)


# ---------------------------------------------------------------------------
# Configuration and data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodicityConfig:
    """Configuration values for one periodicity run.

    Parameters
    ----------
    min_period : float
        Minimum Rayleigh trial period in days.
    max_period : float
        Maximum Rayleigh trial period in days.
    phase_tol : float
        Fractional phase tolerance for the non-uniform period-grid spacing.
    phase_bins : int
        Phase-coverage bins used in the missing-flare simulation.
    prominence : float
        Peak prominence for minima detection in ``-log10(p)``.
    smooth_sigma : float
        Gaussian smoothing width (array-index units) applied to ``-log10(p)``.
    basin_prominence : float
        Not used.  Kept for compatibility with the original notebook.
    rel_gap_rotation : float
        Relative tolerance for matching minima to stellar rotation periods.
    rel_gap_distinct : float
        Relative tolerance for keeping candidate periods distinct.
    n_best : int
        Number of non-rotation candidate periods to report.
    phase_window : float
        Half-width of the central phase window used in phase statistics.
    align_range : float
        Phase range retained after alignment.
    min_flares_for_rayleigh : int
        Minimum observed flare events required before running the Rayleigh search.
    rayleigh_significance_threshold : float or None
        Fixed p-value threshold for significance.  ``None`` uses ``1/gridsize``.
    rayleigh_statistic_threshold : float or None
        Optional threshold on the Rayleigh statistic.  ``None`` disables it.
    max_grid_points : int
        Maximum allowed number of period-grid points.
    random_seed : int or None
        RNG seed.  ``None`` gives stochastic behaviour.
    n_jobs : int
        Parallel workers for the Rayleigh search.  ``1`` = serial, ``-1`` = all
        cores.  Falls back to serial if joblib is unavailable.
    save_figures : bool
        Save plots to the period-statistics directory.
    show_figures : bool
        Display figures interactively.
    debug_mode : bool
        Add diagnostic markers to the Rayleigh plot.
    """

    min_period: float = 1.0
    max_period: float = 12.0
    phase_tol: float = 0.25
    phase_bins: int = 51
    prominence: float = 0.25
    smooth_sigma: float = 0.1
    basin_prominence: float = 0.1  # API-compatibility only; not used
    rel_gap_rotation: float = 0.05
    rel_gap_distinct: float = 0.05
    n_best: int = 3
    phase_window: float = 0.25
    align_range: float = 0.5
    min_flares_for_rayleigh: int = 5
    rayleigh_significance_threshold: float | None = None
    rayleigh_statistic_threshold: float | None = None
    max_grid_points: int = 100_000_000
    random_seed: int | None = None
    n_jobs: int = 20
    save_figures: bool = True
    show_figures: bool = True
    debug_mode: bool = False


@dataclass
class InputData:
    """Container for all input dataframes used by the workflow.

    Attributes
    ----------
    flares : pandas.DataFrame
        Flare table from ``Data/<TIC>_flares.csv``.
    time_series : pandas.DataFrame
        Detrended time-series from ``Data/<TIC>_detrended.csv``.
    star_properties : pandas.DataFrame
        Stellar properties from ``Data/<TIC>_star_properties.csv``.
    stellar_rotation : pandas.DataFrame
        Stellar rotation from ``Data/<TIC>_df_stellar_rotation.csv``.
    waiting_time_summary : pandas.DataFrame
        Waiting-time summary from ``Results/Waiting_time_statistics/<TIC>_summary.csv``.
    paths : dict
        Resolved paths used in the run.
    """

    flares: pd.DataFrame
    time_series: pd.DataFrame
    star_properties: pd.DataFrame
    stellar_rotation: pd.DataFrame
    waiting_time_summary: pd.DataFrame
    paths: dict[str, Path]


@dataclass
class RayleighResult:
    """Output from a Rayleigh period search.

    Attributes
    ----------
    periods : numpy.ndarray
        Trial periods in days.
    p_values : numpy.ndarray
        Rayleigh p-values for each trial period.
    statistics : numpy.ndarray
        Rayleigh statistics ``z = R^2 / n``.
    n_events : numpy.ndarray
        Event count (observed + simulated) per trial period.
    performed : bool
        Whether the search was performed.
    skipped : bool
        Whether the search was skipped.
    skip_reason : str
        Human-readable skip reason.
    output_file : pathlib.Path or None
        Written CSV file, or ``None`` if skipped.
    """

    periods: np.ndarray
    p_values: np.ndarray
    statistics: np.ndarray
    n_events: np.ndarray
    performed: bool
    skipped: bool
    skip_reason: str = "none"
    output_file: Path | None = None


class PeriodicityInputError(ValueError):
    """Raised when required input files or columns are missing or invalid."""


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def configure_matplotlib(
    font_path: str | Path | None = "/usr/share/fonts/dejavu-serif-fonts/DejaVuSerifCondensed.ttf",
) -> None:
    """Apply thesis-style Matplotlib defaults used by all plots.

    Parameters
    ----------
    font_path : str, pathlib.Path, or None
        Path to DejaVu Serif Condensed.  Falls back to the configured serif
        font when the file is absent.
    """

    rc = {
        "mathtext.fontset": "stix",
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "axes.linewidth": LINEWIDTH,
        "lines.linewidth": LINEWIDTH,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "legend.fontsize": FONTSIZE * 0.72,
    }
    if font_path is not None and Path(font_path).exists():
        custom_font = fm.FontProperties(fname=str(font_path))
        rc["font.family"] = custom_font.get_name()
    plt.rcParams.update(rc)


def resolve_tic_base_path(tic_id: str | int, base_path: str | Path) -> Path:
    """Resolve the directory containing one TIC's data.

    Parameters
    ----------
    tic_id : str or int
        TIC identifier.
    base_path : str or pathlib.Path
        Direct TIC directory or a parent containing a ``<tic_id>`` subdirectory.

    Returns
    -------
    pathlib.Path
        Resolved TIC directory.

    Raises
    ------
    PeriodicityInputError
        When neither candidate path contains a ``Data`` subdirectory.
    """

    tic_id = str(tic_id)
    base = Path(base_path).expanduser()
    if (base / "Data").exists():
        return base
    candidate = base / tic_id
    if (candidate / "Data").exists():
        return candidate
    raise PeriodicityInputError(
        f"Could not resolve TIC data directory. Expected either '{base}/Data' "
        f"or '{candidate}/Data' to exist."
    )


def read_csv_checked(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    """Read a CSV file with clear error messages.

    Parameters
    ----------
    path : pathlib.Path
        CSV file to read.
    allow_empty : bool
        Return an empty dataframe for empty files instead of raising.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    PeriodicityInputError
        When the file is missing, empty (and ``allow_empty`` is False), or
        malformed.
    """

    if not path.exists():
        raise PeriodicityInputError(f"Required input file does not exist: {path}")
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise PeriodicityInputError(f"Required input file is empty: {path}") from None
    except Exception as exc:
        raise PeriodicityInputError(f"Could not read CSV file '{path}': {exc}") from exc
    if df.empty and not allow_empty:
        raise PeriodicityInputError(f"Required input file contains no rows: {path}")
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise ``PeriodicityInputError`` if any required column is missing.

    Parameters
    ----------
    df : pandas.DataFrame
    columns : iterable of str
    name : str
        Human-readable name used in error messages.
    """

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise PeriodicityInputError(
            f"{name} is missing required column(s): {', '.join(missing)}"
        )


def finite_numeric_array(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    """Convert values to a finite 1-D float array, dropping NaN/inf.

    Parameters
    ----------
    values : array-like
    name : str
        Name used in error messages.
    allow_empty : bool
        Allow an empty result.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    PeriodicityInputError
        When no finite values remain and ``allow_empty`` is False.
    """

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 and not allow_empty:
        raise PeriodicityInputError(f"{name} contains no finite numeric values.")
    return arr


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------


def load_periodicity_inputs(tic_id: str | int, base_path: str | Path) -> InputData:
    """Load and validate all dataframes needed for one TIC.

    Parameters
    ----------
    tic_id : str or int
    base_path : str or pathlib.Path
        Direct TIC directory or parent containing TIC subdirectories.

    Returns
    -------
    InputData

    Raises
    ------
    PeriodicityInputError
        When required files, rows, columns, or finite numeric data are missing.
    """

    tic_id = str(tic_id)
    tic_base = resolve_tic_base_path(tic_id, base_path)
    paths = {
        "base_dir": tic_base,
        "flare": tic_base / "Data" / f"{tic_id}_flares.csv",
        "time_series": tic_base / "Data" / f"{tic_id}_detrended.csv",
        "star_properties": tic_base / "Data" / f"{tic_id}_star_properties.csv",
        "stellar_rotation": tic_base / "Data" / f"{tic_id}_df_stellar_rotation.csv",
        "waiting_time_summary": tic_base / "Results" / "Waiting_time_statistics" / f"{tic_id}_summary.csv",
        "period_results": tic_base / "Results" / "Period_statistics",
    }

    time_series = read_csv_checked(paths["time_series"])
    star_properties = read_csv_checked(paths["star_properties"])
    stellar_rotation = read_csv_checked(paths["stellar_rotation"])
    waiting_time_summary = read_csv_checked(paths["waiting_time_summary"])
    flares = read_csv_checked(paths["flare"], allow_empty=True)

    require_columns(time_series, ["time"], "Detrended time-series dataframe")
    require_columns(stellar_rotation, ["stellar_rotation_period"], "Stellar-rotation dataframe")
    waiting_time_summary["total_observing_time_with_gaps"] = (
        np.max(time_series["time"]) - np.min(time_series["time"])
    )
    require_columns(
        waiting_time_summary, ["total_observing_time_with_gaps"], "Waiting-time summary dataframe"
    )
    if not flares.empty:
        require_columns(flares, ["tstart"], "Flare dataframe")

    if not flares.empty:
        flares = flares.copy()
        flares["tstart"] = pd.to_numeric(flares["tstart"], errors="coerce")
        flares = flares[np.isfinite(flares["tstart"])].sort_values("tstart").reset_index(drop=True)

    time_series = time_series.copy()
    time_series["time"] = pd.to_numeric(time_series["time"], errors="coerce")
    time_series = time_series[np.isfinite(time_series["time"])].reset_index(drop=True)
    if time_series.empty:
        raise PeriodicityInputError(
            "Detrended time-series dataframe has no finite 'time' values."
        )

    return InputData(
        flares=flares,
        time_series=time_series,
        star_properties=star_properties,
        stellar_rotation=stellar_rotation,
        waiting_time_summary=waiting_time_summary,
        paths=paths,
    )


def parse_stellar_rotation_periods(stellar_rotation_df: pd.DataFrame) -> np.ndarray:
    """Extract all stellar rotation periods from the stellar-rotation table.

    Handles ``np.float64(...)`` strings, plain lists, comma-separated strings,
    and falls back to the scalar ``stellar_rotation_period`` column.

    Parameters
    ----------
    stellar_rotation_df : pandas.DataFrame

    Returns
    -------
    numpy.ndarray
        Sorted finite positive rotation periods.

    Raises
    ------
    PeriodicityInputError
        When no valid periods are found.
    """

    candidates: list[float] = []
    if "all_stellar_rotation_periods" in stellar_rotation_df.columns:
        raw = stellar_rotation_df["all_stellar_rotation_periods"].iloc[0]
        if isinstance(raw, str):
            candidates.extend(
                float(v) for v in re.findall(r"np\.float64\((.*?)\)", raw)
            )
            if not candidates:
                try:
                    parsed = ast.literal_eval(raw)
                    candidates.extend(np.asarray(parsed, dtype=float).ravel().tolist())
                except Exception:
                    candidates.extend(
                        float(v)
                        for v in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", raw)
                    )
        elif isinstance(raw, (list, tuple, np.ndarray, pd.Series)):
            candidates.extend(np.asarray(raw, dtype=float).ravel().tolist())

    if not candidates and "stellar_rotation_period" in stellar_rotation_df.columns:
        candidates.extend(
            pd.to_numeric(stellar_rotation_df["stellar_rotation_period"], errors="coerce")
            .dropna()
            .astype(float)
            .tolist()
        )

    periods = np.asarray(candidates, dtype=float)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size == 0:
        raise PeriodicityInputError("No valid stellar rotation periods were found.")
    return np.unique(periods)


def get_total_observing_time(input_data: InputData) -> float:
    """Return the observing baseline used for period-grid spacing.

    Parameters
    ----------
    input_data : InputData

    Returns
    -------
    float
        Positive observing time in days.

    Raises
    ------
    PeriodicityInputError
        When the summary value is missing or invalid.
    """

    series = pd.to_numeric(
        input_data.waiting_time_summary["total_observing_time_with_gaps"], errors="coerce"
    ).dropna()
    if series.empty or not np.isfinite(series.iloc[0]) or series.iloc[0] <= 0:
        raise PeriodicityInputError(
            "Waiting-time summary has no positive finite 'total_observing_time_with_gaps'."
        )
    return float(series.iloc[0])


# ---------------------------------------------------------------------------
# Rayleigh analysis
# ---------------------------------------------------------------------------


def build_period_grid(
    total_observing_time: float,
    min_period: float,
    max_period: float,
    phase_tol: float,
    max_grid_points: int,
) -> np.ndarray:
    """Build the non-uniform period grid from the original notebook.

    The step size at each period ``P`` is ``phase_tol * P^2 / T_obs``,
    giving finer resolution at shorter periods.

    Parameters
    ----------
    total_observing_time : float
        Observation baseline in days.
    min_period : float
    max_period : float
    phase_tol : float
        Fractional phase tolerance.
    max_grid_points : int
        Hard cap on grid size.

    Returns
    -------
    numpy.ndarray
        Trial-period grid.

    Raises
    ------
    PeriodicityInputError
        When parameters are invalid or the grid would exceed ``max_grid_points``.
    """

    if not np.isfinite(total_observing_time) or total_observing_time <= 0:
        raise PeriodicityInputError("total_observing_time must be a positive finite value.")
    if not all(np.isfinite(v) for v in [min_period, max_period, phase_tol]):
        raise PeriodicityInputError("Period-grid settings must be finite.")
    if min_period <= 0 or max_period <= 0 or max_period < min_period:
        raise PeriodicityInputError("Require 0 < min_period <= max_period.")
    if phase_tol <= 0:
        raise PeriodicityInputError("phase_tol must be positive.")
    if max_grid_points < 1:
        raise PeriodicityInputError("max_grid_points must be at least 1.")

    periods: list[float] = []
    period = float(min_period)
    while period <= max_period:
        periods.append(period)
        if len(periods) >= max_grid_points:
            raise PeriodicityInputError(
                f"Period grid exceeded max_grid_points={max_grid_points}. "
                "Increase phase_tol or narrow the period range."
            )
        step = phase_tol * (period * period) / total_observing_time
        period += max(step, 1e-12)

    if not periods:
        raise PeriodicityInputError("Period grid is empty.")
    return np.asarray(periods, dtype=float)


def compute_corrected_phases(
    flare_times: np.ndarray,
    observation_times: np.ndarray,
    trial_period: float,
    *,
    n_bins: int = 51,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Correct flare phases for uneven observation phase coverage.

    Under-covered phase bins are filled with Poisson-simulated flare phases
    drawn uniformly inside each bin, relative to the best-covered bin.

    Parameters
    ----------
    flare_times : numpy.ndarray
        Observed flare start times.
    observation_times : numpy.ndarray
        Observation cadence times used to estimate phase coverage.
    trial_period : float
        Folding period in days.
    n_bins : int
        Number of phase bins.
    rng : numpy.random.Generator or None
        Random generator.  A fresh one is created when ``None``.

    Returns
    -------
    all_flares : numpy.ndarray
        Observed plus simulated flare phases.
    diagnostics : dict
        Keys: ``folded_time``, ``folded_flares``, ``simulated_flares``,
        ``obs_counts``, ``bin_edges``, ``simulated_missing_counts``.

    Raises
    ------
    PeriodicityInputError
        When ``trial_period`` or ``n_bins`` are invalid.
    """

    if rng is None:
        rng = np.random.default_rng()
    if not np.isfinite(trial_period) or trial_period <= 0:
        raise PeriodicityInputError("trial_period must be positive and finite.")
    if n_bins < 1:
        raise PeriodicityInputError("n_bins must be at least 1.")

    # Step 1: fold times onto [0, 1).
    flares = finite_numeric_array(flare_times, "flare_times")
    times = finite_numeric_array(observation_times, "observation_times")
    folded_time = (times % trial_period) / trial_period
    folded_flares = (flares % trial_period) / trial_period

    # Step 2: histogram observation coverage and compute missing counts.
    obs_counts, bin_edges = np.histogram(folded_time, bins=n_bins, range=(0, 1))

    if obs_counts.size == 0 or obs_counts.max() <= 0:
        simulated_flares = np.array([], dtype=float)
        simulated_missing_counts = np.zeros(n_bins, dtype=int)
    else:
        missing_counts = obs_counts.max() - obs_counts
        expected_amount_flares_per_t = len(flares) / max(len(times), 1)
        expected_missing_flares = np.clip(
            missing_counts * expected_amount_flares_per_t, 0, None
        )
        simulated_missing_counts = rng.poisson(expected_missing_flares)

        # Step 3: simulate missing flares uniformly inside each under-sampled bin.
        simulated: list[np.ndarray] = []
        for idx, n_sim in enumerate(simulated_missing_counts):
            if n_sim > 0:
                simulated.append(
                    rng.uniform(bin_edges[idx], bin_edges[idx + 1], size=int(n_sim))
                )
        simulated_flares = np.concatenate(simulated) if simulated else np.array([], dtype=float)

    all_flares = np.concatenate([folded_flares, simulated_flares])
    diagnostics = {
        "folded_time": folded_time,
        "folded_flares": folded_flares,
        "simulated_flares": simulated_flares,
        "obs_counts": obs_counts,
        "bin_edges": bin_edges,
        "simulated_missing_counts": simulated_missing_counts,
    }
    return all_flares, diagnostics


def rayleigh_test_from_phases(phases: np.ndarray) -> tuple[float, float, int]:
    """Compute the Rayleigh statistic and p-value from phase values.

    Uses the finite-sample approximation from Zar (1999):
    ``p = exp(sqrt(1 + 4n + 4(n^2 - R^2)) - (1 + 2n))``.

    Parameters
    ----------
    phases : numpy.ndarray
        Phase values in cycles; wrapped to ``[0, 1)`` internally.

    Returns
    -------
    statistic : float
        Rayleigh statistic ``z = R^2 / n``.
    p_value : float
        Rayleigh p-value.
    n_events : int
        Number of phase values used.

    Raises
    ------
    PeriodicityInputError
        When fewer than two finite phases are available.
    """

    clean = finite_numeric_array(phases, "phases") % 1.0
    n_events = int(clean.size)
    if n_events < 2:
        raise PeriodicityInputError("Rayleigh test requires at least two phase values.")

    theta = 2.0 * np.pi * clean
    c_sum = float(np.sum(np.cos(theta)))
    s_sum = float(np.sum(np.sin(theta)))
    r_squared = c_sum * c_sum + s_sum * s_sum
    statistic = r_squared / n_events

    exponent = (
        np.sqrt(1.0 + 4.0 * n_events + 4.0 * (n_events * n_events - r_squared))
        - (1.0 + 2.0 * n_events)
    )
    p_value = float(np.exp(exponent))
    return statistic, p_value, n_events


def _evaluate_rayleigh_period(
    trial_period: float,
    flare_times: np.ndarray,
    observation_times: np.ndarray,
    phase_bins: int,
    seed: int | None = None,
) -> tuple[float, float, int]:
    """Compute one corrected Rayleigh test for serial or parallel execution.

    Parameters
    ----------
    trial_period : float
    flare_times : numpy.ndarray
    observation_times : numpy.ndarray
    phase_bins : int
    seed : int or None
        Per-worker RNG seed for reproducible parallel runs.

    Returns
    -------
    statistic, p_value, n_events : tuple
        Returns ``(nan, nan, 0)`` on error so the caller's arrays stay aligned.
    """

    worker_rng = np.random.default_rng(seed)
    try:
        phases, _ = compute_corrected_phases(
            flare_times,
            observation_times,
            float(trial_period),
            n_bins=phase_bins,
            rng=worker_rng,
        )
        return rayleigh_test_from_phases(phases)
    except PeriodicityInputError:
        return np.nan, np.nan, 0


def run_rayleigh_period_search(
    tic_id: str | int,
    results_dir: str | Path,
    flares_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    total_observing_time: float,
    config: PeriodicityConfig,
    *,
    rng: np.random.Generator | None = None,
) -> RayleighResult:
    """Search trial periods with the Rayleigh phase test.

    Supports parallel execution via joblib when available.  Falls back to
    serial execution automatically when joblib is missing or fails.

    Parameters
    ----------
    tic_id : str or int
    results_dir : str or pathlib.Path
    flares_df : pandas.DataFrame
        Must contain ``tstart``.
    ts_df : pandas.DataFrame
        Must contain ``time``.
    total_observing_time : float
        Observation baseline used for grid spacing.
    config : PeriodicityConfig
    rng : numpy.random.Generator or None

    Returns
    -------
    RayleighResult
    """

    # Step 1: validate inputs and build the period grid.
    tic_id = str(tic_id)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    require_columns(flares_df, ["tstart"], "Flare dataframe")
    require_columns(ts_df, ["time"], "Time-series dataframe")

    flare_times = finite_numeric_array(flares_df["tstart"], "flare tstart", allow_empty=True)
    observation_times = finite_numeric_array(ts_df["time"], "time-series time", allow_empty=True)

    if flare_times.size == 0:
        return RayleighResult(
            np.array([]), np.array([]), np.array([]), np.array([]),
            False, True, "no finite flare times",
        )
    if flare_times.size < config.min_flares_for_rayleigh:
        return RayleighResult(
            np.array([]), np.array([]), np.array([]), np.array([]),
            False, True,
            f"only {flare_times.size} flare(s); at least {config.min_flares_for_rayleigh} required",
        )
    if observation_times.size == 0:
        return RayleighResult(
            np.array([]), np.array([]), np.array([]), np.array([]),
            False, True, "no finite observation times",
        )

    periods = build_period_grid(
        total_observing_time,
        config.min_period,
        config.max_period,
        config.phase_tol,
        config.max_grid_points,
    )
    p_values = np.full(periods.shape, np.nan, dtype=float)
    statistics = np.full(periods.shape, np.nan, dtype=float)
    n_events = np.zeros(periods.shape, dtype=int)

    # Step 2: run Rayleigh tests — parallel when possible, serial otherwise.
    n_jobs = int(config.n_jobs) if config.n_jobs is not None else 1
    use_parallel = n_jobs != 1 and Parallel is not None and delayed is not None

    seed_rng = rng if rng is not None else np.random.default_rng()
    period_seeds = seed_rng.integers(
        0, np.iinfo(np.uint32).max, size=len(periods), dtype=np.uint32
    )
    if use_parallel:
        iterator = tqdm(periods, desc="Processing periods") if tqdm is not None else periods
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_evaluate_rayleigh_period)(
                    float(trial_period),
                    flare_times,
                    observation_times,
                    config.phase_bins,
                    int(period_seeds[idx]),
                )
                for idx, trial_period in enumerate(iterator)
            )
            if results:
                statistics[:], p_values[:], n_events[:] = map(np.asarray, zip(*results))
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Parallel Rayleigh search failed ({exc!r}); falling back to serial.",
                RuntimeWarning,
            )
            use_parallel = False

    if not use_parallel:
        iterator = tqdm(periods, desc="Processing periods") if tqdm is not None else periods
        for idx, trial_period in enumerate(iterator):
            try:
                worker_seed = int(period_seeds[idx])
                worker_rng = np.random.default_rng(worker_seed)
                phases, _ = compute_corrected_phases(
                    flare_times,
                    observation_times,
                    float(trial_period),
                    n_bins=config.phase_bins,
                    rng=worker_rng,
                )
                statistic, p_value, n_used = rayleigh_test_from_phases(phases)
            except PeriodicityInputError:
                continue
            statistics[idx] = statistic
            p_values[idx] = p_value
            n_events[idx] = n_used

    # Step 3: check for at least one valid result and write CSV.
    valid = np.isfinite(p_values) & np.isfinite(statistics)
    if not np.any(valid):
        return RayleighResult(
            periods, p_values, statistics, n_events,
            False, True, "no valid Rayleigh tests could be computed",
        )

    out = results_dir / f"{tic_id}_rayleigh_results.csv"
    pd.DataFrame({
        "T": periods,
        "p_values": p_values,
        "rayleigh_statistic": statistics,
        "n_events": n_events,
    }).to_csv(out, index=False)
    rayleigh = RayleighResult(periods, p_values, statistics, n_events, True, False, "none", out)
    # Save the per-period RNG seeds so downstream steps (jackknife) can
    # reproduce the exact phase-correction simulations used during the
    # Rayleigh search.
    try:
        setattr(rayleigh, "period_seeds", period_seeds)
    except Exception:
        pass
    return rayleigh


# Minimum refinement half-window in grid-index units used by find_distinct_minima.
# This ensures the basin search covers a physically meaningful range of the
# period grid even when smooth_sigma is very small (e.g. the default 0.1).
_MIN_REFINE_PTS: int = 20


def find_distinct_minima(
    p_values: np.ndarray,
    prominence: float,
    smooth_sigma: float,
    basin_prominence: float = 0.1,  # API-compatibility only; not used
) -> np.ndarray:
    """Find distinct local minima in a Rayleigh p-value curve.

    Uses Gaussian-smoothed ``-log10(p)`` for robust peak detection, then
    refines each candidate to the true minimum in the *original unsmoothed*
    p-value array using a window large enough to cover the local basin.

    The refinement window is ``max(_MIN_REFINE_PTS, int(5 * smooth_sigma))``
    grid points on each side.  The floor of ``_MIN_REFINE_PTS = 20`` prevents
    the window collapsing to 1-3 points when ``smooth_sigma`` is small (e.g.
    the default 0.1), which was the source of offset detections in the original
    code.

    Parameters
    ----------
    p_values : numpy.ndarray
        Rayleigh p-values.
    prominence : float
        Minimum peak prominence in the smoothed ``-log10(p)`` curve.
    smooth_sigma : float
        Gaussian smoothing width in array-index units.
    basin_prominence : float
        Not used.  Kept for API compatibility with the original notebook.

    Returns
    -------
    numpy.ndarray
        Sorted integer indices of detected minima in ``p_values``.
    """

    # Step 1: validate and sanitize input.
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=int)
    valid = np.isfinite(p_values) & (p_values > 0)
    if not np.any(valid):
        return np.array([], dtype=int)

    # Step 2: build a NaN-masked safe array (NaN where invalid) and a
    #         fill-replaced copy for the smoothing transform.  The NaN-masked
    #         version is used in the refinement step so nanargmin naturally
    #         skips invalid entries without relying on the fill value.
    safe_nan = np.where(valid, p_values, np.nan)
    fill_value = float(np.nanmax(safe_nan))          # worst valid p-value
    safe_filled = np.where(np.isfinite(safe_nan), safe_nan, fill_value)

    # Step 3: compute -log10 and smooth for peak detection.
    y = -np.log10(np.clip(safe_filled, np.finfo(float).tiny, None))
    y_smooth = (
        gaussian_filter1d(y, sigma=float(smooth_sigma)) if smooth_sigma > 0 else y.copy()
    )

    # Step 4: detect peaks in the smoothed curve.
    peak_indices, _ = find_peaks(y_smooth, prominence=prominence)
    if peak_indices.size == 0:
        return np.array([], dtype=int)

    # Step 5: refine each smoothed-curve peak to the true minimum in the
    #         original unsmoothed p-value array.
    #
    #         WHY the window floor matters:
    #         With smooth_sigma = 0.1 (default), gaussian_filter1d applies
    #         essentially no smoothing (kernel spans ~0.3 index units), so
    #         detected peaks are almost identical to raw -log10(p) peaks.
    #         However, the original code used int(2 * 0.1) = 0, clamped to 1,
    #         giving a ±1 window (3 points).  Any asymmetry in the p-value
    #         basin can place the true minimum 2-5 indices away from the
    #         detected peak, which a 3-point window will miss entirely.
    #         Enforcing a floor of _MIN_REFINE_PTS ensures the basin is
    #         always searched over a meaningful neighborhood.
    refine_half = max(_MIN_REFINE_PTS, int(5 * smooth_sigma))
    n = len(p_values)
    refined: list[int] = []
    for idx in peak_indices:
        lo = max(0, int(idx) - refine_half)
        hi = min(n, int(idx) + refine_half + 1)
        window = safe_nan[lo:hi]        # NaN where invalid; nanargmin skips them
        if np.all(np.isnan(window)):
            continue
        refined.append(lo + int(np.nanargmin(window)))

    return np.array(sorted(set(refined)), dtype=int)


def find_stellar_rotation_minima(
    t_minima: np.ndarray,
    p_values_minima: np.ndarray,
    all_stellar_rotation_periods: np.ndarray,
    rel_gap: float = 0.05,
) -> pd.DataFrame:
    """Find Rayleigh minima associated with stellar rotation periods.

    Parameters
    ----------
    t_minima : numpy.ndarray
        Periods at detected minima.
    p_values_minima : numpy.ndarray
        P-values at detected minima.
    all_stellar_rotation_periods : numpy.ndarray
        Stellar rotation periods to match.
    rel_gap : float
        Relative tolerance for a match.

    Returns
    -------
    pandas.DataFrame
        Matched stellar-rotation minima, possibly empty.
    """

    _cols = [
        "stellar_rotation_period", "associated_minimum_T", "real_multiple",
        "associated_minimum_p", "minima_array_index", "multiple", "relative_gap",
    ]
    t_minima = np.asarray(t_minima, dtype=float)
    p_values_minima = np.asarray(p_values_minima, dtype=float)
    stellar_periods = np.asarray(all_stellar_rotation_periods, dtype=float)
    if t_minima.size == 0 or p_values_minima.size == 0 or stellar_periods.size == 0:
        return pd.DataFrame(columns=_cols)

    rows: list[dict[str, Any]] = []
    for stellar_period in stellar_periods[np.isfinite(stellar_periods) & (stellar_periods > 0)]:
        closest_idx = int(np.nanargmin(np.abs(t_minima - stellar_period)))
        found_minimum = float(t_minima[closest_idx])
        relative_gap = abs(found_minimum - stellar_period) / abs(stellar_period)
        if relative_gap <= rel_gap:
            rows.append({
                "stellar_rotation_period": stellar_period,
                "associated_minimum_T": found_minimum,
                "real_multiple": stellar_period,
                "associated_minimum_p": float(p_values_minima[closest_idx]),
                "minima_array_index": closest_idx,
                "multiple": 1,
                "relative_gap": relative_gap,
            })

    if rows:
        return pd.DataFrame(rows).drop_duplicates(["minima_array_index"])
    return pd.DataFrame(columns=_cols)


def find_stellar_rotation_family(
    t_minima: np.ndarray,
    p_values_minima: np.ndarray,
    stellar_rotation_minima: pd.DataFrame,
    rel_gap: float = 0.05,
) -> pd.DataFrame:
    """Find harmonics and multiples of stellar-rotation minima.

    Parameters
    ----------
    t_minima : numpy.ndarray
    p_values_minima : numpy.ndarray
    stellar_rotation_minima : pandas.DataFrame
        Direct stellar-rotation matches from ``find_stellar_rotation_minima``.
    rel_gap : float

    Returns
    -------
    pandas.DataFrame
        Stellar-rotation family table, possibly empty.
    """

    _cols = [
        "stellar_rotation_period", "associated_minimum_T", "real_multiple",
        "associated_minimum_p", "minima_array_index", "multiple", "relative_gap",
    ]
    if stellar_rotation_minima is None or stellar_rotation_minima.empty:
        return pd.DataFrame(columns=_cols)

    t_minima = np.asarray(t_minima, dtype=float)
    p_values_minima = np.asarray(p_values_minima, dtype=float)
    if t_minima.size == 0:
        return pd.DataFrame(columns=_cols)

    rows: list[dict[str, Any]] = []
    for stellar_period in stellar_rotation_minima["associated_minimum_T"]:
        for multiple in [0.25, 1 / 3, 0.5, 2, 3, 4, 5]:
            target = float(stellar_period) / multiple
            if not np.isfinite(target) or target == 0:
                continue
            relative_gaps = np.abs(t_minima - target) / abs(target)
            valid_idxs = np.where(relative_gaps <= rel_gap)[0]
            if valid_idxs.size > 0:
                best_idx = int(valid_idxs[np.nanargmin(p_values_minima[valid_idxs])])
                rows.append({
                    "stellar_rotation_period": float(stellar_period),
                    "associated_minimum_T": float(t_minima[best_idx]),
                    "real_multiple": target,
                    "associated_minimum_p": float(p_values_minima[best_idx]),
                    "minima_array_index": best_idx,
                    "multiple": multiple,
                    "relative_gap": float(relative_gaps[best_idx]),
                })

    if rows:
        return pd.DataFrame(rows).drop_duplicates(["minima_array_index"])
    return pd.DataFrame(columns=_cols)


def stellar_rotation_period_family_combinations(
    t_minima: np.ndarray,
    p_values_minima: np.ndarray,
    stellar_rotation_family: pd.DataFrame,
    rel_gap: float,
) -> pd.DataFrame:
    """Add linear combinations of stellar-rotation family periods.

    Parameters
    ----------
    t_minima : numpy.ndarray
    p_values_minima : numpy.ndarray
    stellar_rotation_family : pandas.DataFrame
    rel_gap : float

    Returns
    -------
    pandas.DataFrame
        Combined and sorted stellar-rotation family table.
    """

    if stellar_rotation_family is None or stellar_rotation_family.empty:
        return stellar_rotation_family if stellar_rotation_family is not None else pd.DataFrame()

    t_minima = np.asarray(t_minima, dtype=float)
    p_values_minima = np.asarray(p_values_minima, dtype=float)
    associated = (
        pd.to_numeric(stellar_rotation_family["associated_minimum_T"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    differences = np.abs(np.diff(associated))
    rows: list[dict[str, Any]] = []

    for period in associated:
        for diff in differences:
            for sign, label in [(-1, "-"), (1, "+")]:
                target = period + sign * diff
                if not np.isfinite(target) or target <= 0:
                    continue
                relative_gaps = np.abs(t_minima - target) / abs(target)
                valid_idxs = np.where(relative_gaps <= rel_gap)[0]
                if valid_idxs.size > 0:
                    best_idx = int(valid_idxs[np.nanargmin(p_values_minima[valid_idxs])])
                    rows.append({
                        "stellar_rotation_period": period,
                        "associated_minimum_T": float(t_minima[best_idx]),
                        "real_multiple": target,
                        "associated_minimum_p": float(p_values_minima[best_idx]),
                        "minima_array_index": best_idx,
                        "multiple": f"{period:.2f}{label}{diff:.2f}",
                        "relative_gap": float(relative_gaps[best_idx]),
                    })

    if rows:
        extra = pd.DataFrame(rows).drop_duplicates(["minima_array_index"])
        stellar_rotation_family = pd.concat([extra, stellar_rotation_family], ignore_index=True)
    return stellar_rotation_family.sort_values(by="minima_array_index").reset_index(drop=True)


def combi_period_finder(
    t_minima_non_stellar: np.ndarray,
    p_values_minima_non_stellar: np.ndarray,
    stellar_rotation_family: pd.DataFrame,
    stellar_rotation_period: float,  # API-compatibility only; not used
    harmonic_tolerance: float = 0.05,
    beat_tolerance: float = 0.05,
    max_harmonic_order: int = 40,
) -> pd.DataFrame:
    """Score non-stellar candidate periods using harmonics and beat periods.

    Scoring: ``-log10(p)`` plus a harmonic bonus (+1.5 per harmonic match)
    and a beat-period bonus (+3.0).

    Parameters
    ----------
    t_minima_non_stellar : numpy.ndarray
    p_values_minima_non_stellar : numpy.ndarray
    stellar_rotation_family : pandas.DataFrame
        Must contain ``associated_minimum_T``.
    stellar_rotation_period : float
        Not used.  Kept for API compatibility with the original notebook.
    harmonic_tolerance : float
    beat_tolerance : float
    max_harmonic_order : int

    Returns
    -------
    pandas.DataFrame
        Candidates sorted by descending score then ascending p-value.
    """

    _cols = [
        "period", "minimum_p_value", "score", "amount_of_harmonics",
        "found_beat_period", "beat_with_stellar_period", "beat_order", "harmonic_matches",
    ]
    candidate_periods = np.asarray(t_minima_non_stellar, dtype=float)
    p_values = np.asarray(p_values_minima_non_stellar, dtype=float)
    valid = (
        np.isfinite(candidate_periods) & (candidate_periods > 0)
        & np.isfinite(p_values) & (p_values > 0)
    )
    candidate_periods = candidate_periods[valid]
    p_values = p_values[valid]
    if candidate_periods.size == 0:
        return pd.DataFrame(columns=_cols)

    # Step 1: gather stellar family periods for beat matching.
    stellar_periods = np.array([], dtype=float)
    if stellar_rotation_family is not None and "associated_minimum_T" in stellar_rotation_family.columns:
        sp = (
            pd.to_numeric(stellar_rotation_family["associated_minimum_T"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        stellar_periods = sp[np.isfinite(sp) & (sp > 0)]

    def is_close_rel(a: float, b: float, tol: float) -> bool:
        return b != 0 and abs(a - b) / abs(b) <= tol

    def find_harmonics(period: float, all_periods: np.ndarray) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for other in all_periods:
            if np.isclose(other, period):
                continue
            for order in range(2, max_harmonic_order + 1):
                if is_close_rel(other, order * period, harmonic_tolerance):
                    matches.append({"matched_period": float(other), "relation": f"{order}x"})
                    break
                if is_close_rel(other, period / order, harmonic_tolerance):
                    matches.append({"matched_period": float(other), "relation": f"1/{order}x"})
                    break
        unique: list[dict[str, Any]] = []
        seen: set[tuple[float, str]] = set()
        for match in matches:
            key = (round(match["matched_period"], 10), match["relation"])
            if key not in seen:
                seen.add(key)
                unique.append(match)
        return unique

    def find_beat_match(candidate_period: float) -> dict[str, Any] | None:
        best_match = None
        for stellar_t in stellar_periods:
            f_stellar = 1.0 / stellar_t
            for other_t in candidate_periods:
                if np.isclose(other_t, candidate_period):
                    continue
                delta_f = abs(1.0 / other_t - f_stellar)
                if delta_f <= 0:
                    continue
                predicted_beat_t = 1.0 / delta_f
                rel_err = abs(predicted_beat_t - candidate_period) / candidate_period
                if rel_err <= beat_tolerance:
                    match = {
                        "beat_with_stellar_period": float(stellar_t),
                        "other_period_used": float(other_t),
                        "predicted_beat_period": float(predicted_beat_t),
                        "relative_error": float(rel_err),
                    }
                    if best_match is None or rel_err < best_match["relative_error"]:
                        best_match = match
        return best_match

    # Step 2: score each candidate.
    rows: list[dict[str, Any]] = []
    for period, p_val in zip(candidate_periods, p_values):
        harmonic_matches = find_harmonics(float(period), candidate_periods)
        beat_match = find_beat_match(float(period))
        amount_of_harmonics = len(harmonic_matches)
        found_beat_period = beat_match is not None
        p_score = -np.log10(max(float(p_val), 1e-300))
        total_score = p_score + amount_of_harmonics * 1.5 + (3.0 if found_beat_period else 0.0)
        rows.append({
            "period": float(period),
            "minimum_p_value": float(p_val),
            "score": float(total_score),
            "amount_of_harmonics": amount_of_harmonics,
            "found_beat_period": found_beat_period,
            "beat_with_stellar_period": None if beat_match is None else beat_match["beat_with_stellar_period"],
            "beat_order": None if beat_match is None else beat_match["other_period_used"],
            "harmonic_matches": harmonic_matches,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(by=["score", "minimum_p_value"], ascending=[False, True])
        .reset_index(drop=True)
    )


def find_best_non_rotation_periods(
    periods: np.ndarray,
    p_values: np.ndarray,
    stellar_rotation_family: pd.DataFrame | None,
    minima_idx: np.ndarray,
    rel_gap_rotation: float = 0.05,
    rel_gap_distinct: float = 0.05,
    n_best: int = 3,
) -> tuple[list[float], list[float]]:
    """Select the best distinct minima not tied to stellar rotation.

    Parameters
    ----------
    periods : numpy.ndarray
    p_values : numpy.ndarray
    stellar_rotation_family : pandas.DataFrame or None
        Minima to reject.
    minima_idx : numpy.ndarray
        Candidate minima indices in ``periods``.
    rel_gap_rotation : float
        Relative gap for rejecting stellar-rotation-related periods.
    rel_gap_distinct : float
        Relative gap for enforcing distinctness between selected periods.
    n_best : int

    Returns
    -------
    best_periods, best_p_values : list of float
        May contain fewer than ``n_best`` entries.
    """

    periods = np.asarray(periods, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    minima_idx = np.asarray(minima_idx, dtype=int)
    minima_idx = minima_idx[(minima_idx >= 0) & (minima_idx < len(periods))]

    if minima_idx.size == 0:
        return [], []

    # Step 1: collect stellar-rotation family periods to reject.
    associated_t = np.array([], dtype=float)
    if (
        stellar_rotation_family is not None
        and not stellar_rotation_family.empty
        and "associated_minimum_T" in stellar_rotation_family.columns
    ):
        associated_t = (
            pd.to_numeric(stellar_rotation_family["associated_minimum_T"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        associated_t = associated_t[np.isfinite(associated_t) & (associated_t > 0)]

    # Step 2: iterate candidates from lowest to highest p-value.
    sorted_minima_idx = minima_idx[np.argsort(p_values[minima_idx])]
    best_periods: list[float] = []
    best_p_values: list[float] = []
    for idx in sorted_minima_idx:
        candidate_period = float(periods[idx])
        candidate_p_value = float(p_values[idx])
        if not np.isfinite(candidate_period) or not np.isfinite(candidate_p_value):
            continue
        if associated_t.size > 0:
            if np.any(np.abs(candidate_period - associated_t) / associated_t <= rel_gap_rotation):
                continue
        if any(
            abs(candidate_period - chosen) / chosen <= rel_gap_distinct
            for chosen in best_periods
            if chosen != 0
        ):
            continue
        best_periods.append(candidate_period)
        best_p_values.append(candidate_p_value)
        if len(best_periods) == n_best:
            break

    return best_periods, best_p_values


def summarize_rayleigh_analysis(
    tic_id: str | int,
    rayleigh: RayleighResult,
    stellar_rotation_family: pd.DataFrame | None,
    minima_idx: np.ndarray,
    config: PeriodicityConfig,
) -> pd.DataFrame:
    """Create the Rayleigh summary table used downstream.

    Column names match the original notebook.  Significance thresholds
    follow the original grid-size convention when no fixed threshold is set.

    Parameters
    ----------
    tic_id : str or int
    rayleigh : RayleighResult
    stellar_rotation_family : pandas.DataFrame or None
    minima_idx : numpy.ndarray
    config : PeriodicityConfig

    Returns
    -------
    pandas.DataFrame
        One-row summary.
    """

    _skip_row: dict[str, Any] = {
        "TIC_id": str(tic_id),
        "gridsize": 0,
        "RT_best_period": np.nan,
        "RT_best_p_value": np.nan,
        "RT_best_statistic": np.nan,
        "RT_best_n_events": 0,
        "RT_best_period_1": np.nan,
        "RT_best_p_value_1": np.nan,
        "RT_best_period_2": np.nan,
        "RT_best_p_value_2": np.nan,
        "RT_best_period_3": np.nan,
        "RT_best_p_value_3": np.nan,
        "one_detection": np.nan,
        "one_detection_one_sigma": np.nan,
        "one_detection_two_sigma": np.nan,
        "one_detection_three_sigma": np.nan,
        "n_points_exceed_threshold": 0,
        "n_points_exceed_1sigma": 0,
        "n_points_exceed_2sigma": 0,
        "n_points_exceed_3sigma": 0,
        "rayleigh_performed": False,
        "rayleigh_skipped": True,
        "rayleigh_skip_reason": rayleigh.skip_reason,
        "rayleigh_significant": False,
        "rayleigh_significance_threshold": config.rayleigh_significance_threshold,
        "rayleigh_statistic_threshold": config.rayleigh_statistic_threshold,
    }

    if rayleigh.skipped or rayleigh.periods.size == 0:
        return pd.DataFrame([_skip_row])

    valid = np.isfinite(rayleigh.p_values) & (rayleigh.p_values > 0)
    if not np.any(valid):
        skipped = RayleighResult(
            rayleigh.periods, rayleigh.p_values, rayleigh.statistics, rayleigh.n_events,
            False, True, "all Rayleigh p-values invalid",
        )
        return summarize_rayleigh_analysis(tic_id, skipped, stellar_rotation_family, minima_idx, config)

    # Step 1: compute grid size, best period, and significance thresholds.
    gridsize = len(rayleigh.periods)
    best_idx = int(np.nanargmin(np.where(valid, rayleigh.p_values, np.nan)))
    best_period = float(rayleigh.periods[best_idx])
    best_p_value = float(rayleigh.p_values[best_idx])
    best_statistic = float(rayleigh.statistics[best_idx])
    best_n_events = int(rayleigh.n_events[best_idx])

    one_detection = 1 / gridsize
    one_detection_one_sigma = 0.3819660112501051 / gridsize
    one_detection_two_sigma = 0.1715728752538099 / gridsize
    one_detection_three_sigma = 0.0916730868040161 / gridsize

    # Step 2: select best non-rotation candidate periods.
    best_periods, best_p_values = find_best_non_rotation_periods(
        rayleigh.periods,
        rayleigh.p_values,
        stellar_rotation_family,
        minima_idx,
        config.rel_gap_rotation,
        config.rel_gap_distinct,
        config.n_best,
    )
    while len(best_periods) < config.n_best:
        best_periods.append(np.nan)
        best_p_values.append(np.nan)

    # Step 3: evaluate significance.
    p_threshold = (
        config.rayleigh_significance_threshold
        if config.rayleigh_significance_threshold is not None
        else one_detection
    )
    p_sig = bool(best_p_value < p_threshold)
    z_sig = (
        True
        if config.rayleigh_statistic_threshold is None
        else bool(best_statistic >= config.rayleigh_statistic_threshold)
    )

    return pd.DataFrame([{
        "TIC_id": str(tic_id),
        "gridsize": gridsize,
        "RT_best_period": best_period,
        "RT_best_p_value": best_p_value,
        "RT_best_statistic": best_statistic,
        "RT_best_n_events": best_n_events,
        "RT_best_period_1": best_periods[0],
        "RT_best_p_value_1": best_p_values[0],
        "RT_best_period_2": best_periods[1],
        "RT_best_p_value_2": best_p_values[1],
        "RT_best_period_3": best_periods[2],
        "RT_best_p_value_3": best_p_values[2],
        "one_detection": one_detection,
        "one_detection_one_sigma": one_detection_one_sigma,
        "one_detection_two_sigma": one_detection_two_sigma,
        "one_detection_three_sigma": one_detection_three_sigma,
        "n_points_exceed_threshold": int(np.sum(rayleigh.p_values < one_detection)),
        "n_points_exceed_1sigma": int(np.sum(rayleigh.p_values < one_detection_one_sigma)),
        "n_points_exceed_2sigma": int(np.sum(rayleigh.p_values < one_detection_two_sigma)),
        "n_points_exceed_3sigma": int(np.sum(rayleigh.p_values < one_detection_three_sigma)),
        "rayleigh_performed": True,
        "rayleigh_skipped": False,
        "rayleigh_skip_reason": "none",
        "rayleigh_significant": bool(p_sig and z_sig),
        "rayleigh_significance_threshold": p_threshold,
        "rayleigh_statistic_threshold": config.rayleigh_statistic_threshold,
    }])


# ---------------------------------------------------------------------------
# Phase statistics
# ---------------------------------------------------------------------------


def phase_align_min_avg_circular_distance(
    df_flares: pd.DataFrame,
    summary_df: pd.DataFrame,
    half_width: float = 0.25,
) -> tuple[float, int]:
    """Compute the phase offset from the circular mean.

    Parameters
    ----------
    df_flares : pandas.DataFrame
        Must contain ``tstart``.
    summary_df : pandas.DataFrame
        Must contain ``RT_best_period_1``.
    half_width : float
        Half-width for counting events in the centered window.

    Returns
    -------
    offset : float
        Phase offset in ``[0, 1)``.
    count_in_window : int
        Flares inside ``[-half_width, half_width]`` after alignment.

    Raises
    ------
    PeriodicityInputError
        When ``RT_best_period_1`` is invalid or no phases can be computed.
    """

    require_columns(df_flares, ["tstart"], "Flare dataframe")
    require_columns(summary_df, ["RT_best_period_1"], "Summary dataframe")

    best_period = pd.to_numeric(summary_df["RT_best_period_1"], errors="coerce").iloc[0]
    if not np.isfinite(best_period) or best_period <= 0:
        raise PeriodicityInputError("Cannot align phases because RT_best_period_1 is invalid.")

    flare_times = finite_numeric_array(df_flares["tstart"], "flare tstart")
    phases0 = (flare_times / best_period) % 1.0
    if phases0.size == 0:
        raise PeriodicityInputError("No phases computed for phase alignment.")

    offset_grid = np.linspace(0, 1, 10000, endpoint=False)
    counts = []

    for test_offset in offset_grid:
        shifted_phase = ((phases0 - test_offset + 0.5) % 1) - 0.5
        n_inside = int(np.sum((shifted_phase >= -half_width) & (shifted_phase <= half_width)))
        counts.append(n_inside)

    counts = np.array(counts, dtype=int)
    best_offset = float(offset_grid[np.argmax(counts)])
    best_count = int(counts.max())

    phases = ((phases0 - best_offset + 0.5) % 1) - 0.5
    count_in_window = int(np.sum((phases >= -half_width) & (phases <= half_width)))
    return best_offset, count_in_window


def phase_window_statistics(
    df_flares: pd.DataFrame,
    summary_df: pd.DataFrame,
    phase_window: float = 0.25,
    align_range: float = 0.5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute phase-window statistics for the selected candidate period.

    Parameters
    ----------
    df_flares : pandas.DataFrame
        Must contain ``tstart``.
    summary_df : pandas.DataFrame
        Rayleigh summary; must contain ``RT_best_period_1``.
    phase_window : float
        Half-width of the central phase window.
    align_range : float
        Phase range retained after alignment.

    Returns
    -------
    phases : numpy.ndarray
        Aligned flare phases.
    summary_df : pandas.DataFrame
        Copy with ``offset``, ``planet_induced_percentage``, and related
        columns added.

    Raises
    ------
    PeriodicityInputError
        When the best period or flare phases are invalid.
    """

    summary_df = summary_df.copy()
    offset, _ = phase_align_min_avg_circular_distance(
        df_flares, summary_df, half_width=phase_window
    )
    best_period = pd.to_numeric(summary_df["RT_best_period_1"], errors="coerce").iloc[0]
    flare_times = finite_numeric_array(df_flares["tstart"], "flare tstart")
    phases = (flare_times / best_period) % 1.0
    phases = (phases - offset + 0.5) % 1.0 - 0.5
    phases = phases[(phases >= -align_range) & (phases <= align_range)]

    total = int(len(phases))
    in_range = int(np.sum((phases >= -phase_window) & (phases <= phase_window)))
    planet_induced_percentage = np.nan
    if total > 0:
        planet_induced_percentage = ((in_range - ((total - in_range) / 2)) / total) * 100

    summary_df["offset"] = [offset]
    summary_df["planet_induced_percentage"] = [planet_induced_percentage]
    summary_df["phase_window"] = [phase_window]
    summary_df["align_range"] = [align_range]
    summary_df["phase_events_total"] = [total]
    summary_df["phase_events_in_window"] = [in_range]
    return phases, summary_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _set_thesis_tick_style(ax) -> None:
    """Apply thesis-style tick widths, lengths, and spine widths to an axis."""

    ax.tick_params(
        axis="both",
        which="major",
        width=LINEWIDTH,
        length=7 * SCALE,
        direction="out",
        pad=10,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=LINEWIDTH * 0.8,
        length=4 * SCALE,
        direction="out",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH)


def _plain_scalar_formatter() -> ScalarFormatter:
    """Return a ScalarFormatter with offset and scientific notation disabled."""

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    return formatter


def _apply_scalar_formatter(ax, *, xaxis: bool = True, yaxis: bool = True) -> None:
    """Disable scientific and offset notation on linear numeric axes."""

    if xaxis:
        ax.xaxis.set_major_formatter(_plain_scalar_formatter())
    if yaxis:
        ax.yaxis.set_major_formatter(_plain_scalar_formatter())


def _integer_log_tick_formatter(x: float, pos: int | None = None) -> str:
    """Format selected logarithmic period ticks as plain integers."""

    _ = pos
    integer_ticks = np.arange(1, 13)
    if np.isclose(x, integer_ticks).any():
        return f"{int(round(x))}"
    return ""


def _center_phase(phases: np.ndarray) -> np.ndarray:
    """Shift phases from [0, 1) to the reference plotting range [-0.5, 0.5)."""

    values = np.asarray(phases, dtype=float)
    values = values[np.isfinite(values)]
    return values - 0.5


def _add_panel_label(ax, label: str) -> None:
    """Add a reference-style panel label in the upper-left corner."""

    ax.text(
        0.03,
        0.93,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTSIZE * 1.05,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 4,
        },
        zorder=100,
    )


def _style_axis(ax, *, panel_label: str | None = None, scalar_x: bool = True, scalar_y: bool = True) -> None:
    """Apply shared thesis styling to one axis."""

    _set_thesis_tick_style(ax)
    _apply_scalar_formatter(ax, xaxis=scalar_x, yaxis=scalar_y)
    if panel_label is not None:
        _add_panel_label(ax, panel_label)


def _save_or_show_figure(
    fig,
    path: Path | None,
    *,
    save: bool = True,
    show: bool = True,
) -> Path | None:
    """Save a figure as a 300-dpi tight PDF and handle display/closing."""

    saved_path: Path | None = None
    if save and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved_path = path
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def plot_phase_correction(
    tic_id: str | int,
    results_dir: str | Path,
    diagnostics: dict[str, np.ndarray],
    trial_period: float,
    *,
    save: bool = True,
    show: bool = True,
) -> Path | None:
    """Plot phase-coverage correction diagnostics as a reference-style 2x2 PDF.

    The scientific inputs are unchanged: the panels use the folded phases already
    produced by ``compute_corrected_phases``.  For readability, phases are shifted
    from ``[0, 1)`` to ``[-0.5, 0.5)`` and the final panel stacks detected and
    simulated flares.

    Parameters
    ----------
    tic_id : str or int
    results_dir : str or pathlib.Path
    diagnostics : dict
        Output of ``compute_corrected_phases``.
    trial_period : float
        Period used to create the diagnostics. Retained for API compatibility.
    save : bool
    show : bool

    Returns
    -------
    pathlib.Path or None
        Saved plot path, or ``None`` if not saved.
    """

    _ = trial_period
    required = ["folded_time", "folded_flares", "simulated_flares"]
    if any(key not in diagnostics for key in required):
        return None

    folded_time = _center_phase(diagnostics["folded_time"])
    folded_flares = _center_phase(diagnostics["folded_flares"])
    simulated_flares = _center_phase(diagnostics["simulated_flares"])

    bins = np.linspace(-0.5, 0.5, 51)
    counts_time, edges = np.histogram(folded_time, bins=bins)
    counts_detected, _ = np.histogram(folded_flares, bins=bins)
    counts_simulated, _ = np.histogram(simulated_flares, bins=bins)
    counts_all = counts_detected + counts_simulated
    bin_widths = np.diff(edges)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    grey_color = "0.60"
    sim_color = "crimson"
    phase_xticks = [-0.4, -0.2, 0.0, 0.2, 0.4]

    axes[0].bar(
        edges[:-1],
        counts_time,
        width=bin_widths,
        align="edge",
        color=grey_color,
        alpha=0.85,
        linewidth=0,
        edgecolor="none",
    )
    axes[1].bar(
        edges[:-1],
        counts_detected,
        width=bin_widths,
        align="edge",
        color=grey_color,
        alpha=0.95,
        linewidth=0,
        edgecolor="none",
    )
    axes[2].bar(
        edges[:-1],
        counts_simulated,
        width=bin_widths,
        align="edge",
        color=sim_color,
        alpha=1.0,
        linewidth=0,
        edgecolor="none",
    )
    axes[3].bar(
        edges[:-1],
        counts_detected,
        width=bin_widths,
        align="edge",
        color=grey_color,
        alpha=0.95,
        linewidth=0,
        edgecolor="none",
    )
    sim_nonzero = counts_simulated > 0
    axes[3].bar(
        edges[:-1][sim_nonzero],
        counts_simulated[sim_nonzero],
        width=bin_widths[sim_nonzero],
        align="edge",
        bottom=counts_detected[sim_nonzero],
        color=sim_color,
        alpha=1.0,
        linewidth=0,
        edgecolor="none",
    )

    for ax, label in zip(axes, ["A", "B", "C", "D"]):
        ax.set_xlim(-0.5, 0.5)
        ax.xaxis.set_major_locator(FixedLocator(phase_xticks))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axis(ax, panel_label=label, scalar_x=True, scalar_y=True)

    axes[0].set_ylabel("Count", labelpad=18)
    axes[2].set_ylabel("Count", labelpad=18)
    axes[1].set_ylabel("")
    axes[3].set_ylabel("")

    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    axes[2].set_xlabel("Orbital phase", labelpad=14)
    axes[3].set_xlabel("Orbital phase", labelpad=14)

    axes[0].set_ylim(0, max(1, counts_time.max()) * 1.05)
    axes[1].set_ylim(0, max(1, counts_detected.max()) * 1.15)
    axes[2].set_ylim(0, max(1, counts_simulated.max()) * 1.20)
    axes[3].set_ylim(0, max(1, counts_all.max()) * 1.20)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.13,
        top=0.98,
        wspace=0.18,
        hspace=0.25,
    )

    fig_path = Path(results_dir) / f"{tic_id}_best_period_phase_simulation.pdf"
    return _save_or_show_figure(fig, fig_path, save=save, show=show)


def plot_rayleigh_test(
    tic_id: str | int,
    results_dir: str | Path,
    periods: np.ndarray,
    p_values: np.ndarray,
    summary_df: pd.DataFrame,
    minima_idx: np.ndarray,
    stellar_rotation_df: pd.DataFrame,
    *,
    debug_mode: bool = False,
    save: bool = True,
    show: bool = True,
) -> Path | None:
    """Plot Rayleigh p-values and selected candidate periods as a thesis-style PDF.

    The plot follows the compact reference style: a thick black Rayleigh curve,
    crimson highlighted candidate points, a crimson dashed three-sigma detection
    threshold, and plain integer labels on the logarithmic period axis.
    """

    periods = np.asarray(periods, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    valid = np.isfinite(periods) & np.isfinite(p_values) & (periods > 0) & (p_values > 0)
    if not np.any(valid) or summary_df.empty:
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    ax.plot(
        periods[valid],
        p_values[valid],
        color="black",
        alpha=0.95,
        lw=LINEWIDTH,
        zorder=30,
    )

    best_period_columns = ["RT_best_period", "RT_best_period_1", "RT_best_period_2", "RT_best_period_3"]
    plotted_periods: list[float] = []
    for column in best_period_columns:
        candidate_period = summary_df.get(column, pd.Series([np.nan])).iloc[0]
        if not (np.isfinite(candidate_period) and candidate_period > 0):
            continue
        if any(np.isclose(candidate_period, existing, rtol=1e-8, atol=1e-10) for existing in plotted_periods):
            continue
        idx = int(np.nanargmin(np.abs(periods - candidate_period)))
        if 0 <= idx < len(periods) and np.isfinite(p_values[idx]) and p_values[idx] > 0:
            ax.scatter(
                periods[idx],
                p_values[idx],
                color="crimson",
                alpha=0.85,
                s=80 * SCALE**2,
                linewidths=0,
                zorder=60,
            )
            plotted_periods.append(float(candidate_period))

    three_sigma = summary_df.get("one_detection_three_sigma", pd.Series([np.nan])).iloc[0]
    if np.isfinite(three_sigma) and three_sigma > 0:
        ax.axhline(
            three_sigma,
            color="crimson",
            alpha=0.85,
            lw=LINEWIDTH,
            ls="--",
            zorder=5,
        )

    if debug_mode:
        minima_idx = np.asarray(minima_idx, dtype=int)
        minima_idx = minima_idx[(minima_idx >= 0) & (minima_idx < len(periods))]
        if minima_idx.size > 0:
            ax.scatter(
                periods[minima_idx],
                p_values[minima_idx],
                color="grey",
                alpha=0.45,
                s=10 * SCALE**2,
                linewidths=0,
                zorder=40,
            )
        if "stellar_rotation_period" in stellar_rotation_df.columns:
            stellar_period = pd.to_numeric(
                stellar_rotation_df["stellar_rotation_period"], errors="coerce"
            ).iloc[0]
            if np.isfinite(stellar_period) and stellar_period > 0:
                for harmonic in [2, 3, 4, 5, 6]:
                    ax.axvline(
                        stellar_period / harmonic,
                        color="grey",
                        ls=":",
                        lw=LINEWIDTH * 0.65,
                        alpha=0.45,
                        zorder=2,
                    )
                ax.axvline(
                    stellar_period,
                    color="grey",
                    ls=":",
                    lw=LINEWIDTH * 0.9,
                    alpha=0.55,
                    zorder=2,
                )

    ax.set_xlim(1, 12)
    ax.set_xlabel("Trial period [days]")
    ax.set_ylabel("p-value")
    ax.set_xscale("log")
    ax.set_yscale("log")

    rt_xticks = np.arange(1, 13, 2)
    ax.set_xticks(rt_xticks)
    ax.xaxis.set_major_locator(FixedLocator(rt_xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_integer_log_tick_formatter))

    positive_p_values = p_values[valid]
    y_min = 0.9 * float(np.nanmin(positive_p_values))
    if np.isfinite(three_sigma) and three_sigma > 0:
        y_min = min(y_min, 0.9 * float(three_sigma))
    y_min = max(y_min, np.finfo(float).tiny)
    ax.set_ylim(0.9*y_min, 1.0)

    _style_axis(ax, panel_label="A", scalar_x=False, scalar_y=False)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.96)

    fig_path = Path(results_dir) / f"{tic_id}_rayleigh_test.pdf"
    return _save_or_show_figure(fig, fig_path, save=save, show=show)


def plot_final_flare_phase(
    phases: np.ndarray,
    summary_df: pd.DataFrame,
    results_dir: str | Path | None = None,
    *,
    save: bool = True,
    show: bool = True,
) -> Path | None:
    """Plot the final phase-folded flare distribution as a thesis-style PDF."""

    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if phases.size == 0 or summary_df.empty:
        return None

    best_period = summary_df.get("RT_best_period", pd.Series([np.nan])).iloc[0]
    tic_id = summary_df.get("TIC_id", pd.Series(["unknown"])).iloc[0]
    if not np.isfinite(best_period):
        best_period = summary_df.get("RT_best_period_1", pd.Series([np.nan])).iloc[0]
    if not np.isfinite(best_period):
        return None

    bins = np.linspace(-0.5, 0.5, 21)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    counts, _, _ = ax.hist(
        phases,
        bins=bins,
        color="grey",
        alpha=0.65,
        linewidth=0,
        zorder=5,
    )

    ax.axvspan(
        -0.25,
        0.25,
        color="lightblue",
        alpha=0.35,
        zorder=0,
    )
    ax.axvline(
        -0.25,
        color="grey",
        lw=LINEWIDTH,
        ls=":",
        zorder=2,
    )
    ax.axvline(
        0.25,
        color="grey",
        lw=LINEWIDTH,
        ls=":",
        zorder=2,
    )

    ax.set_xlim(-0.5, 0.5)
    if len(counts) > 0 and np.nanmax(counts) > 0:
        ax.set_ylim(0, 1.15 * np.nanmax(counts))
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Number of flares")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axis(ax, panel_label="A", scalar_x=True, scalar_y=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.96)

    fig_path = None
    if results_dir is not None:
        fig_path = Path(results_dir) / f"{tic_id}_final_flare_phase.pdf"
    return _save_or_show_figure(fig, fig_path, save=save and results_dir is not None, show=show)


# ---------------------------------------------------------------------------
# Main workflow and logging
# ---------------------------------------------------------------------------


def make_skip_summary(
    tic_id: str | int,
    reason: str,
    n_flares: int = 0,
    time_span: float | None = None,
) -> pd.DataFrame:
    """Create a one-row summary for a skipped workflow run.

    Parameters
    ----------
    tic_id : str or int
    reason : str
    n_flares : int
    time_span : float or None

    Returns
    -------
    pandas.DataFrame
    """

    return pd.DataFrame([{
        "TIC_id": str(tic_id),
        "number_of_flares": int(n_flares),
        "time_span": np.nan if time_span is None else time_span,
        "rayleigh_performed": False,
        "rayleigh_skipped": True,
        "rayleigh_skip_reason": reason,
        "rayleigh_significant": False,
    }])


def print_run_summary(
    tic_id: str | int,
    input_file: Path | str | None,
    n_flares: int,
    time_span: float | None,
    summary_df: pd.DataFrame,
    summary_file: Path | None,
    plots_made: bool,
    config: PeriodicityConfig,
) -> None:
    """Print a log-friendly summary block for one run.

    Parameters
    ----------
    tic_id : str or int
    input_file : pathlib.Path, str, or None
    n_flares : int
    time_span : float or None
    summary_df : pandas.DataFrame
    summary_file : pathlib.Path or None
    plots_made : bool
    config : PeriodicityConfig
    """

    row = (
        summary_df.iloc[0].to_dict()
        if summary_df is not None and not summary_df.empty
        else {}
    )

    def get(key: str, default: Any = np.nan) -> Any:
        return row.get(key, default)

    performed = bool(get("rayleigh_performed", False))
    skipped = bool(get("rayleigh_skipped", not performed))
    p_value = get("RT_best_p_value", np.nan)
    statistic = get("RT_best_statistic", np.nan)
    min_p = p_value
    if all(k in row for k in ["RT_best_p_value_1", "RT_best_p_value_2", "RT_best_p_value_3"]):
        vals = [v for v in [row[k] for k in ["RT_best_p_value_1", "RT_best_p_value_2", "RT_best_p_value_3"]]
                if np.isfinite(v)]
        if vals:
            min_p = min(vals)

    print("=" * 65)
    print(f"TIC ID                                     : {tic_id}")
    print(f"Input file                                 : {input_file or 'not available'}")
    print(f"Number of flares                           : {n_flares}")
    if time_span is not None and np.isfinite(time_span):
        print(f"Time span                                  : {time_span:.2f} d")
    else:
        print("Time span                                  : not available")
    print(f"Rayleigh tests performed                   : {'yes' if performed else 'no'}")
    print(f"Rayleigh tests skipped                     : {'yes' if skipped else 'no'}")
    print(f"Rayleigh skip reason                       : {get('rayleigh_skip_reason', 'none')}")
    print(f"Rayleigh statistic                         : {statistic if np.isfinite(statistic) else 'not available'}")
    print(f"Minimum Rayleigh p-value                   : {p_value if np.isfinite(p_value) else 'not available'}")
    print(f"Minimum Rayleigh p-value (from candidates) : {min_p if np.isfinite(min_p) else 'not available'}")
    print(f"Rayleigh significance threshold            : {get('rayleigh_significance_threshold', config.rayleigh_significance_threshold)}")
    print(f"Rayleigh statistic threshold               : {get('rayleigh_statistic_threshold', config.rayleigh_statistic_threshold)}")
    print(f"Rayleigh n_jobs                            : {config.n_jobs}")
    print(f"Rayleigh significant                       : {'yes' if bool(get('rayleigh_significant', False)) else 'no'}")
    print(f"Events used in Rayleigh test               : {get('RT_best_n_events', 0)}")
    print(f"Best period                                : {get('RT_best_period', 'not available')}")
    print(f"Other candidate periods                    : {get('RT_best_period_1', np.nan)}, {get('RT_best_period_2', np.nan)}, {get('RT_best_period_3', np.nan)}")
    print(f"Phase reference                            : offset={get('offset', 'not available')}")
    print(f"Phase range                                : +/-{get('align_range', config.align_range)}")
    print(f"Summary file written                       : {summary_file or 'not written'}")
    print(f"Plots made                                 : {'yes' if plots_made else 'no'}")
    print("=" * 65)


def run_periodicity_workflow(
    tic_id: str | int,
    base_path: str | Path,
    *,
    make_plots: bool = True,
    config: PeriodicityConfig | None = None,
) -> dict[str, Any]:
    """Run the full periodicity workflow for a single TIC.

    Parameters
    ----------
    tic_id : str or int
    base_path : str or pathlib.Path
        Direct TIC directory or parent containing a TIC subfolder.
    make_plots : bool
        Generate and display plots when ``True``.  Plots are still saved when
        ``config.save_figures`` is ``True`` regardless of this flag.
    config : PeriodicityConfig or None
        ``None`` uses default notebook-equivalent values.

    Returns
    -------
    dict
        Keys: ``skipped``, ``skip_reason``, ``input_data``, ``rayleigh``,
        ``minima_idx``, ``stellar_rotation_minima``, ``stellar_rotation_family``,
        ``phases``, ``summary``, ``summary_file``, ``plot_paths``.

    Raises
    ------
    PeriodicityInputError
        When required input files or columns are missing.
    """

    tic_id = str(tic_id)
    config = config or PeriodicityConfig()
    configure_matplotlib()
    rng = (
        np.random.default_rng(config.random_seed)
        if config.random_seed is not None
        else np.random.default_rng()
    )

    # Step 1: load inputs and resolve paths.
    input_data = load_periodicity_inputs(tic_id, base_path)
    results_dir = input_data.paths["period_results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    flare_times = finite_numeric_array(input_data.flares.get("tstart", []), "flare tstart", allow_empty=True)
    obs_times = finite_numeric_array(input_data.time_series["time"], "time-series time")
    time_span = float(np.nanmax(obs_times) - np.nanmin(obs_times)) if obs_times.size >= 2 else None
    n_flares = int(flare_times.size)
    summary_file = results_dir / f"{tic_id}_summary_df.csv"

    if n_flares == 0:
        reason = "no detected flares"
        summary_df = make_skip_summary(tic_id, reason, n_flares, time_span)
        summary_df.to_csv(summary_file, index=False)
        print_run_summary(tic_id, input_data.paths["flare"], n_flares, time_span, summary_df, summary_file, False, config)
        return {
            "skipped": True, "skip_reason": reason,
            "summary": summary_df, "summary_file": summary_file, "input_data": input_data,
        }

    # Step 2: parse stellar rotation metadata.
    all_stellar_rotation_periods = parse_stellar_rotation_periods(input_data.stellar_rotation)
    stellar_rotation_period = float(
        pd.to_numeric(input_data.stellar_rotation["stellar_rotation_period"], errors="coerce")
        .dropna()
        .iloc[0]
    )
    total_observing_time = get_total_observing_time(input_data)

    # Step 3: run the Rayleigh period search.
    rayleigh = run_rayleigh_period_search(
        tic_id,
        results_dir,
        input_data.flares,
        input_data.time_series,
        total_observing_time,
        config,
        rng=rng,
    )

    minima_idx = np.array([], dtype=int)
    stellar_rotation_minima = pd.DataFrame()
    stellar_rotation_family = pd.DataFrame()
    phases = np.array([], dtype=float)
    plot_paths: list[Path] = []

    if rayleigh.skipped:
        summary_df = summarize_rayleigh_analysis(tic_id, rayleigh, None, minima_idx, config)
        summary_df["number_of_flares"] = [n_flares]
        summary_df["time_span"] = [time_span]
        summary_df.to_csv(summary_file, index=False)
        print_run_summary(tic_id, input_data.paths["flare"], n_flares, time_span, summary_df, summary_file, False, config)
        return {
            "skipped": True,
            "skip_reason": rayleigh.skip_reason,
            "input_data": input_data,
            "rayleigh": rayleigh,
            "summary": summary_df,
            "summary_file": summary_file,
        }

    # Step 4: detect minima and classify stellar-rotation family.
    minima_idx = find_distinct_minima(
        rayleigh.p_values, config.prominence, config.smooth_sigma, config.basin_prominence
    )
    if minima_idx.size > 0:
        t_minima = rayleigh.periods[minima_idx]
        p_minima = rayleigh.p_values[minima_idx]
        stellar_rotation_minima = find_stellar_rotation_minima(
            t_minima, p_minima, all_stellar_rotation_periods, rel_gap=config.rel_gap_rotation
        )
        stellar_rotation_family = find_stellar_rotation_family(
            t_minima, p_minima, stellar_rotation_minima, rel_gap=config.rel_gap_rotation
        )
        if not stellar_rotation_minima.empty or not stellar_rotation_family.empty:
            stellar_rotation_family = pd.concat(
                [stellar_rotation_minima, stellar_rotation_family], ignore_index=True
            )
            if "multiple" in stellar_rotation_family.columns:
                stellar_rotation_family = stellar_rotation_family.sort_values(
                    by="multiple"
                ).reset_index(drop=True)
            stellar_rotation_family = stellar_rotation_period_family_combinations(
                t_minima, p_minima, stellar_rotation_family, config.rel_gap_rotation
            )
            family_file = results_dir / f"{tic_id}_stellar_rotation_family.csv"
            stellar_rotation_family.to_csv(family_file, index=False)
    else:
        stellar_rotation_family = pd.DataFrame()

    # Step 5: build summary dataframe.
    summary_df = summarize_rayleigh_analysis(
        tic_id, rayleigh, stellar_rotation_family, minima_idx, config
    )
    summary_df["number_of_flares"] = [n_flares]
    summary_df["time_span"] = [time_span]
    summary_df["tested_min_period"] = [config.min_period]
    summary_df["tested_max_period"] = [config.max_period]
    summary_df["phase_tol"] = [config.phase_tol]
    summary_df["phase_bins"] = [config.phase_bins]
    summary_df["stellar_rotation_period"] = [stellar_rotation_period]
    summary_df["all_stellar_rotation_periods"] = [list(map(float, all_stellar_rotation_periods))]

    try:
        phases, summary_df = phase_window_statistics(
            input_data.flares,
            summary_df,
            phase_window=config.phase_window,
            align_range=config.align_range,
        )
    except PeriodicityInputError as exc:
        summary_df["phase_skip_reason"] = [str(exc)]
        phases = np.array([], dtype=float)

    summary_df.to_csv(summary_file, index=False)

    # Step 6: generate plots.
    show_plots = bool(make_plots) and bool(config.show_figures)
    save_plots = bool(config.save_figures) or not bool(make_plots)
    plots_made = False

    rayleigh_plot = plot_rayleigh_test(
        tic_id, results_dir,
        rayleigh.periods, rayleigh.p_values,
        summary_df, minima_idx, input_data.stellar_rotation,
        debug_mode=config.debug_mode,
        save=save_plots, show=show_plots,
    )
    if rayleigh_plot is not None:
        plot_paths.append(rayleigh_plot)
        plots_made = True

    best_period = summary_df.get("RT_best_period", pd.Series([np.nan])).iloc[0]
    if np.isfinite(best_period) and best_period > 0:
        _, diagnostics = compute_corrected_phases(
            flare_times, obs_times, float(best_period),
            n_bins=config.phase_bins, rng=rng,
        )
        phase_plot = plot_phase_correction(
            tic_id, results_dir, diagnostics, float(best_period),
            save=save_plots, show=show_plots,
        )
        if phase_plot is not None:
            plot_paths.append(phase_plot)
            plots_made = True

    final_phase_plot = plot_final_flare_phase(
        phases, summary_df, results_dir, save=save_plots, show=show_plots,
    )
    if final_phase_plot is not None:
        plot_paths.append(final_phase_plot)
        plots_made = True

    print_run_summary(
        tic_id, input_data.paths["flare"], n_flares, time_span,
        summary_df, summary_file, plots_made, config,
    )
    return {
        "skipped": False,
        "skip_reason": "none",
        "input_data": input_data,
        "rayleigh": rayleigh,
        "minima_idx": minima_idx,
        "stellar_rotation_minima": stellar_rotation_minima,
        "stellar_rotation_family": stellar_rotation_family,
        "phases": phases,
        "summary": summary_df,
        "summary_file": summary_file,
        "plot_paths": plot_paths,
    }


__all__ = [
    "BASE_FONTSIZE",
    "BASE_LINEWIDTH",
    "SCALE",
    "FONTSIZE",
    "LINEWIDTH",
    "DEFAULT_FIGSIZE",
    "TWO_PANEL_FIGSIZE",
    "VERTICAL_TWO_PANEL_FIGSIZE",
    "InputData",
    "PeriodicityConfig",
    "PeriodicityInputError",
    "RayleighResult",
    "build_period_grid",
    "combi_period_finder",
    "compute_corrected_phases",
    "configure_matplotlib",
    "find_best_non_rotation_periods",
    "find_distinct_minima",
    "find_stellar_rotation_family",
    "find_stellar_rotation_minima",
    "finite_numeric_array",
    "load_periodicity_inputs",
    "parse_stellar_rotation_periods",
    "phase_align_min_avg_circular_distance",
    "phase_window_statistics",
    "plot_final_flare_phase",
    "plot_phase_correction",
    "plot_rayleigh_test",
    "rayleigh_test_from_phases",
    "read_csv_checked",
    "require_columns",
    "resolve_tic_base_path",
    "run_periodicity_workflow",
    "run_rayleigh_period_search",
    "stellar_rotation_period_family_combinations",
    "summarize_rayleigh_analysis",
]
