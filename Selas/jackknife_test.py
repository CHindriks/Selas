"""Minimal jackknife test utilities for SELAS.

This module implements a leave-one-out jackknife Rayleigh test and only runs
when the Rayleigh exceedance level is positive.

The saved jackknife dataframe is written to the target TIC's Results/Period_statistics folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

from periodicity_statistics import build_period_grid, compute_corrected_phases, rayleigh_test_from_phases


def first_existing_path(candidate_paths: list[Path | str]) -> Path:
    """Return the first path that exists, or raise a helpful error."""
    for path in candidate_paths:
        path = Path(path)
        if path.exists():
            return path

    message = "None of these files were found:\n" + "\n".join(
        f"  - {Path(path)}" for path in candidate_paths
    )
    raise FileNotFoundError(message)


def load_jackknife_inputs(
    tic_id: str | int,
    base_root: Path = Path("../Data/Selas-TIC-ids"),
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load flare and detrended time-series inputs for the jackknife test."""
    tic_id = str(tic_id)
    base_root = Path(base_root)
    candidate_dir = base_root / tic_id
    if candidate_dir.exists() and (candidate_dir / "Data").exists():
        base_dir = candidate_dir
    elif base_root.name == tic_id and (base_root / "Data").exists():
        base_dir = base_root
    else:
        base_dir = candidate_dir

    data_dir = base_dir / "Data"

    flare_path = first_existing_path([
        data_dir / f"{tic_id}_combined_final_flares.csv",
        data_dir / f"{tic_id}_flares.csv",
    ])
    ts_path = first_existing_path([
        data_dir / f"{tic_id}_detrended_timeseries_TOFFEE.csv",
        data_dir / f"{tic_id}_detrended.csv",
    ])

    flare_df = pd.read_csv(flare_path)
    ts_df = pd.read_csv(ts_path)

    if "tstart" not in flare_df.columns:
        raise ValueError(f"{flare_path} must contain a 'tstart' column.")
    if "time" not in ts_df.columns:
        raise ValueError(f"{ts_path} must contain a 'time' column.")

    flare_df = flare_df.copy()
    ts_df = ts_df.copy()

    if "flare_type" in flare_df.columns:
        flare_df = flare_df[flare_df["flare_type"] != "secondary"].copy()

    flare_df["tstart"] = pd.to_numeric(flare_df["tstart"], errors="coerce")
    ts_df["time"] = pd.to_numeric(ts_df["time"], errors="coerce")

    flare_df = flare_df.dropna(subset=["tstart"]).sort_values("tstart").reset_index(drop=True)
    ts_df = ts_df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if len(flare_df) < 5:
        raise ValueError(f"Jackknife needs at least 5 flares after filtering; found {len(flare_df)}.")
    if len(ts_df) == 0:
        raise ValueError("The time-series dataframe is empty after filtering.")

    return flare_df, ts_df, base_dir


def find_distinct_minima(
    p_values: np.ndarray,
    prominence: float = 0.25,
    smooth_sigma: float = 0,
    basin_prominence: float = 0.1,
) -> np.ndarray:
    """Find distinct minima in a p-value curve by locating peaks in -log10(p)."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=int)

    valid = np.isfinite(p_values) & (p_values > 0)
    if not np.any(valid):
        return np.array([], dtype=int)

    safe = np.where(valid, p_values, np.nan)
    fallback = np.nanmax(safe)
    safe_filled = np.where(np.isfinite(safe), safe, fallback)
    y = -np.log10(np.clip(safe_filled, np.finfo(float).tiny, None))

    if smooth_sigma > 0:
        y = gaussian_filter1d(y, sigma=float(smooth_sigma))

    peak_indices, _ = find_peaks(y, prominence=prominence)
    if peak_indices.size == 0:
        return np.array([], dtype=int)

    refine_half = max(20, int(5 * smooth_sigma))
    n = len(p_values)
    refined = []

    for idx in peak_indices:
        lo = max(0, int(idx) - refine_half)
        hi = min(n, int(idx) + refine_half + 1)
        window = safe[lo:hi]
        if not np.all(np.isnan(window)):
            refined.append(lo + int(np.nanargmin(window)))

    return np.array(sorted(set(refined)), dtype=int)


def make_period_grid(
    T_obs: float,
    min_period: float = 1.0,
    max_period: float = 12.0,
    phase_tol: float = 0.25,
    max_grid_points: int = 1_000_000,
) -> np.ndarray:
    """Build the adaptive Rayleigh trial period grid."""
    periods = []
    P = float(min_period)

    while P <= max_period:
        periods.append(P)
        if len(periods) >= max_grid_points:
            raise ValueError(
                f"Period grid exceeded max_grid_points={max_grid_points}. "
                "Increase phase_tol or narrow the period range."
            )
        step = phase_tol * (P * P) / T_obs
        P += max(step, 1e-12)

    return np.asarray(periods, dtype=float)


def precompute_time_correction(
    time: np.ndarray,
    periods: np.ndarray,
    n_bins: int = 51,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute uneven phase-coverage corrections for each trial period."""
    time = np.asarray(time, dtype=float)
    periods = np.asarray(periods, dtype=float)

    correction_C_unit = np.empty(len(periods), dtype=float)
    correction_S_unit = np.empty(len(periods), dtype=float)
    missing_total_unit = np.empty(len(periods), dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    left = bin_edges[:-1]
    right = bin_edges[1:]
    width = right - left

    mean_cos = (np.sin(2 * np.pi * right) - np.sin(2 * np.pi * left)) / (2 * np.pi * width)
    mean_sin = (-np.cos(2 * np.pi * right) + np.cos(2 * np.pi * left)) / (2 * np.pi * width)

    for j, P in enumerate(tqdm(periods, desc="Precomputing phase correction")):
        folded_time = (time % P) / P
        obs_counts, _ = np.histogram(folded_time, bins=bin_edges)
        missing_counts = obs_counts.max() - obs_counts
        correction_C_unit[j] = np.sum(missing_counts * mean_cos)
        correction_S_unit[j] = np.sum(missing_counts * mean_sin)
        missing_total_unit[j] = np.sum(missing_counts)

    return correction_C_unit, correction_S_unit, missing_total_unit


def rayleigh_with_precomputed_correction(
    flares: np.ndarray,
    periods: np.ndarray,
    correction_C_unit: np.ndarray,
    correction_S_unit: np.ndarray,
    missing_total_unit: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """Run the Rayleigh test with precomputed phase-coverage corrections."""
    flares = np.asarray(flares, dtype=float)
    periods = np.asarray(periods, dtype=float)
    p_values = np.empty(len(periods), dtype=float)

    if len(flares) == 0 or n_time <= 0:
        p_values[:] = np.nan
        return p_values

    flare_rate = len(flares) / n_time
    for j, P in enumerate(periods):
        phases = (flares % P) / P
        theta = 2.0 * np.pi * phases
        C = np.sum(np.cos(theta)) + flare_rate * correction_C_unit[j]
        S = np.sum(np.sin(theta)) + flare_rate * correction_S_unit[j]
        n_eff = len(flares) + flare_rate * missing_total_unit[j]
        if n_eff <= 0:
            p_values[j] = np.nan
        else:
            z = (C**2 + S**2) / n_eff
            p_values[j] = np.exp(-z)

    return p_values


def one_detection_k_sigma(k: float, N: int) -> float:
    """Return the p-value threshold for a one-detection k-sigma event."""
    s = (-k + np.sqrt(k**2 + 4)) / 2
    return (s**2) / N


def p_value_to_exceedance_sigma(p: float, N: int) -> float:
    """Convert a p-value into the corresponding k-sigma exceedance level."""
    pN = np.clip(float(p) * int(N), 1e-300, None)
    return (1 - pN) / np.sqrt(pN)


def summarize_period_search(
    tic_id: str | int,
    trial: int,
    p_values: np.ndarray,
    T: np.ndarray,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Summarize one jackknife trial into a single-row dataframe."""
    p_values = np.asarray(p_values, dtype=float)
    T = np.asarray(T, dtype=float)
    valid = np.isfinite(p_values)

    if len(T) == 0 or len(p_values) == 0 or not np.any(valid):
        return None

    minima_idx = find_distinct_minima(
        p_values,
        prominence=0.25,
        smooth_sigma=0,
        basin_prominence=0.1,
    )

    gridsize = len(T)
    one_detection_one_sigma = one_detection_k_sigma(1, gridsize)
    one_detection_two_sigma = one_detection_k_sigma(2, gridsize)
    one_detection_three_sigma = one_detection_k_sigma(3, gridsize)
    one_detection_four_sigma = one_detection_k_sigma(4, gridsize)
    one_detection_five_sigma = one_detection_k_sigma(5, gridsize)

    best_idx = int(np.nanargmin(p_values))
    min_p_value = float(p_values[best_idx])
    best_period = float(T[best_idx])
    best_periods = T[minima_idx] if len(minima_idx) > 0 else np.array([])
    best_pvals = p_values[minima_idx] if len(minima_idx) > 0 else np.array([])

    data = {
        "TIC_id": [str(tic_id)],
        "trial": [trial],
        "gridsize": [gridsize],
        "min_p_value": [min_p_value],
        "best_period": [best_period],
        "n_minima_found": [len(minima_idx)],
        "best_periods": [best_periods.tolist()],
        "best_period_pvals": [best_pvals.tolist()],
        "n_points_exceed_threshold": [int(np.sum(p_values < (1 / gridsize)))],
        "n_points_exceed_1sigma": [int(np.sum(p_values < one_detection_one_sigma))],
        "n_points_exceed_2sigma": [int(np.sum(p_values < one_detection_two_sigma))],
        "n_points_exceed_3sigma": [int(np.sum(p_values < one_detection_three_sigma))],
        "n_points_exceed_4sigma": [int(np.sum(p_values < one_detection_four_sigma))],
        "n_points_exceed_5sigma": [int(np.sum(p_values < one_detection_five_sigma))],
        "RT_exceedance_sigma": [float(p_value_to_exceedance_sigma(min_p_value, gridsize))],
    }

    if extra_columns is not None:
        data.update(extra_columns)

    return pd.DataFrame(data)


def run_jackknife_test(
    tic_id: str | int,
    base_root: Path = Path("../Data/Selas-TIC-ids"),
    n_jobs: int = 20,
    min_period: float = 1.0,
    max_period: float = 12.0,
    phase_tol: float = 0.25,
    n_bins: int = 51,
    save_csv: bool = True,
    show_progress: bool = False,
    periods: np.ndarray | None = None,
    period_seeds: np.ndarray | None = None,
    phase_bins: int | None = None,
) -> pd.DataFrame:
    """Run the leave-one-out jackknife Rayleigh test and save the results."""
    flare_df, ts_df, base_dir = load_jackknife_inputs(tic_id, base_root=base_root)
    time = ts_df["time"].to_numpy(dtype=float)
    total_observing_time_with_gaps = float(np.max(time) - np.min(time))
    flare_times = flare_df["tstart"].to_numpy(dtype=float)

    if periods is None:
        T = build_period_grid(
            total_observing_time_with_gaps,
            min_period=min_period,
            max_period=max_period,
            phase_tol=phase_tol,
            max_grid_points=1_000_000,
        )
    else:
        T = np.asarray(periods, dtype=float)

    n_trials = len(flare_df)

    # Configure parallel execution like `run_rayleigh_period_search`:
    # - interpret `n_jobs` as an int
    # - enable parallel only when joblib is available and `n_jobs != 1`
    # - force serial when `show_progress` is requested so `tqdm` is useful
    n_jobs = int(n_jobs) if n_jobs is not None else 20
    use_parallel = n_jobs != 1 and Parallel is not None and delayed is not None
    if show_progress:
        n_jobs_effective = 20
        use_parallel = False
    else:
        n_jobs_effective = n_jobs if use_parallel else 20

    # If the Rayleigh search supplied per-period RNG seeds, use them so the
    # jackknife phase-corrections reproduce the exact simulated flares used
    # during the original Rayleigh analysis.
    provided_period_seeds = None if period_seeds is None else np.asarray(period_seeds, dtype=np.uint32)
    if provided_period_seeds is not None and provided_period_seeds.size != len(T):
        provided_period_seeds = None
    if phase_bins is None:
        phase_bins = int(n_bins)

    def run_one_trial(trial: int, seed: int | None = None) -> pd.DataFrame | None:
        flare_df_jk = flare_df.drop(flare_df.index[trial]).reset_index(drop=True)
        left_out_idx = int(flare_df.index[trial])
        flares_jk = flare_df_jk["tstart"].to_numpy(dtype=float)
        p_values = np.full(len(T), np.nan, dtype=float)
        for idx, trial_period in enumerate(T):
            # Use per-period seed if provided (matches Rayleigh), otherwise
            # fall back to an RNG seeded from the trial-specific seed.
            if provided_period_seeds is not None:
                worker_rng = np.random.default_rng(int(provided_period_seeds[idx]))
            else:
                worker_rng = np.random.default_rng(seed)
            phases, _ = compute_corrected_phases(
                flares_jk,
                time,
                float(trial_period),
                n_bins=phase_bins,
                rng=worker_rng,
            )
            _, p_values[idx], _ = rayleigh_test_from_phases(phases)

        return summarize_period_search(
            tic_id=tic_id,
            trial=trial,
            p_values=p_values,
            T=T,
            extra_columns={
                "left_out_flare_idx": [left_out_idx],
                "left_out_flare_tstart": [float(flare_df.loc[left_out_idx, "tstart"])],
                "n_flares_used": [len(flare_df_jk)],
            },
        )

    if n_jobs_effective == 20 or Parallel is None:
        all_dfs = [
            run_one_trial(trial)
            for trial in tqdm(range(n_trials), desc="Running jackknife trials")
        ]
    else:
        seed_rng = np.random.default_rng()
        trial_seeds = seed_rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=n_trials,
            dtype=np.uint32,
        )
        iterator = tqdm(range(n_trials), desc="Running jackknife trials") if tqdm is not None else range(n_trials)
        try:
            all_dfs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_one_trial)(trial, int(trial_seeds[trial])) for trial in iterator
            )
        except Exception:
            # Fall back to a serial run with progress reporting on error.
            all_dfs = [run_one_trial(trial) for trial in tqdm(range(n_trials), desc="Running jackknife trials (fallback)")]

    all_dfs = [df for df in all_dfs if df is not None]
    jackknife_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if save_csv:
        results_dir = base_dir / "Results" / "Period_statistics"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{tic_id}_jackknife_df.csv"
        jackknife_df.to_csv(output_path, index=False)

    return jackknife_df


def run_jackknife_if_rayleigh_exceedance(
    tic_id: str | int,
    best_p_value: float,
    gridsize: int,
    base_root: Path = Path("../Data/Selas-TIC-ids"),
    **jackknife_kwargs: Any,
) -> pd.DataFrame:
    """Run the jackknife test only when Rayleigh exceedance is positive."""
    exceedance = p_value_to_exceedance_sigma(best_p_value, gridsize)
    if exceedance <= 0.0:
        return pd.DataFrame()
    return run_jackknife_test(tic_id, base_root=base_root, **jackknife_kwargs)


__all__ = [
    "first_existing_path",
    "load_jackknife_inputs",
    "run_jackknife_test",
    "run_jackknife_if_rayleigh_exceedance",
    "one_detection_k_sigma",
    "p_value_to_exceedance_sigma",
]
