"""Reusable TESS light-curve detrending utilities.

The functions in this module prepare TESS light curves, fit segmented
polynomial baselines, optionally remove short-period sinusoidal residuals, and
produce diagnostic tables and plots. The implementation follows the notebook
logic it was extracted from, while exposing the workflow as reusable functions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import warnings
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

try:
    import lightkurve as lk
except ImportError:
    lk = None

try:
    from astropy.timeseries import LombScargle
    _ASTROPY_LS = True
except ImportError:
    try:
        from astropy.stats import LombScargle
        _ASTROPY_LS = True
    except ImportError:
        _ASTROPY_LS = False

C = {
    "data":    "#60A5FA",
    "trend1":  "#A3E635",
    "trend2":  "#F59E0B",
    "ma":      "#94A3B8",
    "flare":   "#F87171",
    "sigma1":  "#34D399",
    "sigma2":  "#FBBF24",
    "sigma3":  "#F97316",
    "zero":    "#1E293B",
}

@dataclass
class DetrendConfig:
    """Settings used by the detrending pipeline.

    Attributes
    ----------
    window_sizes : tuple of float
        Window sizes, in days, used for segmented polynomial fits. If a known
        stellar rotation period is attached, the pipeline automatically uses
        ``(0.2, 0.4, 0.6, 0.8)`` for periods below 1 day and
        ``(0.4, 0.6, 0.8, 1.0)`` for periods above 2 days.
    poly_deg : int
        Polynomial degree used for each segment fit.
    n_edge : int
        Number of fitted points at each segment edge to up-weight.
    edge_weight : float
        Multiplicative weight applied to edge points.
    min_fit_pts : int
        Minimum number of cadences required for a segment fit.
    flare_mask_sigma : float
        Sigma threshold used for the preliminary flare mask.
    flare_mask_window_d : float
        Rough detrending window, in days, used for the preliminary flare mask.
    second_pass_sigma : float
        Sigma threshold used for second-pass and final flare masks.
    final_flare_max_mask_fraction : float
        Maximum allowed fraction of cadences in the recomputed final flare mask.
        If the candidate mask exceeds this fraction, the code falls back to the
        previous/provisional mask to prevent final-mask explosions.
    rolling_window_pts : int
        Rolling window size, in cadences, for local noise estimates.
    smooth_sigma_cadences : int
        Gaussian smoothing width, in cadences, for the combined trend.
    center_final_residual : bool
        If True, subtract the moving-average median from the final residual.
    centering_ma_window : int
        Moving-average window used for residual centering diagnostics.
    centering_clip_abs : float or None
        Optional absolute clipping limit applied when estimating the centering offset.
        The default is None, so no pre-centering clipping is performed.
    apply_sinusoid_correction : bool
        If True, apply global Lomb-Scargle sinusoid correction when available.
    n_sinusoid_components : int
        Maximum number of global sinusoid components to remove.
    min_sinusoid_period_hr : float
        Minimum searched sinusoid period, in hours.
    max_sinusoid_period_hr : float
        Maximum searched sinusoid period, in hours.
    amp_limit_fraction : float
        Maximum sinusoid amplitude as a multiple of the local sigma estimate.
    apply_local_final_sinusoid_correction : bool
        If True, apply the final local sinusoid correction when available.
    local_sinusoid_window_d : float
        Local sinusoid fitting window, in days.
    local_sinusoid_step_fraction : float
        Step size as a fraction of ``local_sinusoid_window_d``.
    local_sinusoid_min_points : int
        Minimum number of cadences required in a local sinusoid window.
    local_sinusoid_min_cycles : float
        Minimum number of sinusoid cycles required in a local window.
    local_sinusoid_amp_sigma : float
        Minimum local sinusoid amplitude in units of robust sigma.
    local_sinusoid_min_improvement : float
        Minimum fractional robust-sigma improvement required for acceptance.
    local_sinusoid_clip_sigma : float
        Symmetric clipping threshold used during robust local sinusoid fitting.
    n_remove : int
        Number of cadences removed on each side of a detected large gap.
    gap_sigma : float
        Robust sigma multiplier used in the large-gap threshold.
    gap_min_factor : float
        Minimum large-gap threshold as a multiple of the median cadence.
    quality_mask : int
        TESS quality bit mask used by ``get_timeseries``.
    granulation_timescale_days : float or None
        Known or estimated granulation timescale, in days. When set, the
        pipeline enforces a minimum detrending window of
        ``granulation_min_window_factor * granulation_timescale_days`` so that
        the polynomial baseline cannot absorb granulation power. It also
        penalises segments whose window length falls close to the granulation
        timescale in ``score_segment``, and applies a peak-sharpness guard in
        ``multi_sinusoid_correction`` to prevent the periodogram from chasing
        the broad granulation power hump. Set to ``None`` (default) to disable
        all granulation-specific logic.
    granulation_min_window_factor : float
        Minimum window size expressed as a multiple of
        ``granulation_timescale_days``. Windows shorter than this multiple are
        removed from ``window_sizes`` before fitting. Default is 3.0.
    granulation_score_penalty : float
        Extra penalty subtracted from the segment score for windows whose
        length falls below ``granulation_min_window_factor *
        granulation_timescale_days``. The penalty scales linearly from this
        value (at zero window length) to zero (at the minimum safe window
        length). Default is 0.5.
    granulation_sinusoid_sharpness_min : float
        Minimum ratio of a periodogram peak to the median power in a local
        neighbourhood before the peak is accepted as a coherent sinusoid.
        Values below this threshold are interpreted as part of a broad
        granulation hump and rejected. Default is 1.5. Set to 0.0 to disable
        the sharpness guard.
    auto_estimate_granulation : bool
        If True and ``granulation_timescale_days`` is None, attempt to
        estimate the granulation timescale automatically from the ACF of the
        second-pass residuals after the first detrending pass. The estimate is
        stored in the run summary. Default is False.
    granulation_acf_max_lag_days : float
        Maximum lag, in days, searched when estimating the granulation
        timescale from the ACF. Default is 3.0.
    granulation_noise_window_pts : int
        Short rolling-window length, in cadences, used for the shot-noise
        floor when granulation is active. When ``granulation_timescale_days``
        is set (or estimated), ``local_sigma`` for flare detection uses this
        shorter window while segment scoring uses the standard
        ``rolling_window_pts``. Default is 30.
    """
    window_sizes: tuple[float, ...] = (0.4, 0.6, 0.8)
    poly_deg: int = 4
    n_edge: int = 4
    edge_weight: float = 8.0
    min_fit_pts: int = 15
    flare_mask_sigma: float = 5.0
    flare_mask_window_d: float = 2.0
    second_pass_sigma: float = 3.0
    final_flare_max_mask_fraction: float = 0.05
    rolling_window_pts: int = 100
    smooth_sigma_cadences: int = 5
    center_final_residual: bool = True
    centering_ma_window: int = 10
    centering_clip_abs: float | None = None
    apply_sinusoid_correction: bool = True
    n_sinusoid_components: int = 3
    min_sinusoid_period_hr: float = 0.07
    max_sinusoid_period_hr: float = 0.10
    amp_limit_fraction: float = 2.0
    sinusoid_min_improvement: float = 0.005

    # Optional long-period correction applied to the second-pass residuals.
    # This reuses the same global sinusoid remover, but on rotation-like
    # residual modulation instead of the short cadence-scale period range.
    apply_rotation_sinusoid_correction: bool = True
    rotation_sinusoid_min_period_hr: float = 6.0
    rotation_sinusoid_max_period_hr: float = 36.0
    rotation_sinusoid_n_components: int = 3
    rotation_sinusoid_min_improvement: float = 0.02
    rotation_amp_limit_fraction: float = 3.0

    # Known fast-rotator harmonic correction. If a reliable stellar rotation
    # period is supplied, fast rotators are corrected with a harmonic model at
    # P_rot, P_rot/2, P_rot/3, ... per continuous observing block. This is much
    # safer than allowing a free global periodogram to chase only one harmonic.
    known_rotation_period_days: float | None = None
    apply_fast_rotation_harmonic_correction: bool = True
    fast_rotation_max_period_days: float = 1.0
    fast_rotation_harmonics: tuple[float, ...] = (0.5, 1, 2, 3, 4, 5)
    fast_rotation_min_improvement: float = 0.02
    fast_rotation_min_points: int = 120
    fast_rotation_min_cycles: float = 1.5
    fast_rotation_clip_sigma: float = 4.0
    fast_rotation_amp_limit_fraction: float = 3.0
    run_generic_rotation_after_fast_harmonic: bool = False

    apply_local_final_sinusoid_correction: bool = True
    local_sinusoid_window_d: float = 2.0
    local_sinusoid_step_fraction: float = 0.5
    local_sinusoid_min_points: int = 60
    local_sinusoid_min_cycles: float = 1.0
    local_sinusoid_amp_sigma: float = 0.5
    local_sinusoid_min_improvement: float = 0.005
    local_sinusoid_clip_sigma: float = 5.0
    n_remove: int = 125
    gap_sigma: float = 8.0
    gap_min_factor: float = 20.0
    quality_mask: int = (1 << 1) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 13)

    # ------------------------------------------------------------------ #
    # Granulation handling                                                 #
    # ------------------------------------------------------------------ #
    granulation_timescale_days: float | None = None
    granulation_min_window_factor: float = 3.0
    granulation_score_penalty: float = 0.5
    granulation_sinusoid_sharpness_min: float = 1.5
    auto_estimate_granulation: bool = False
    granulation_acf_max_lag_days: float = 3.0
    granulation_noise_window_pts: int = 30

@dataclass
class DetrendResult:
    """Outputs produced by the detrending pipeline.

    Attributes
    ----------
    final_df : pandas.DataFrame
        Time-sorted table containing flux, trends, residuals, masks, and diagnostics.
    seg_stats_p1 : pandas.DataFrame
        First-pass per-segment fit-quality table.
    seg_stats_p2 : pandas.DataFrame
        Second-pass per-segment fit-quality table.
    local_sinusoid_stats : pandas.DataFrame
        Per-window statistics from the final local sinusoid correction.
    summary : dict
        Scalar run summary and diagnostic values.
    arrays : dict
        Local arrays saved from ``run_detrending`` for detailed diagnostics.
    """
    final_df: pd.DataFrame
    seg_stats_p1: pd.DataFrame
    seg_stats_p2: pd.DataFrame
    local_sinusoid_stats: pd.DataFrame
    summary: dict
    arrays: dict

def fix_time_axis(ax):
    """Suppress offset notation on the time axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose x-axis should be formatted.

    Returns
    -------
    None
    """
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))

def sigma_clipped_std(arr: np.ndarray, n_sigma: float = 3.0, max_iter: int = 20) -> float:
    """Compute an iterative sigma-clipped standard deviation.

    Parameters
    ----------
    arr : numpy.ndarray
        Array containing values to summarize. Non-finite values are ignored.
    n_sigma : float, optional
        Sigma threshold used at each clipping iteration.
    max_iter : int, optional
        Maximum number of clipping iterations.

    Returns
    -------
    float
        Standard deviation after clipping. Returns ``numpy.nan`` if fewer than two
        finite points are available.
    """
    r = arr[np.isfinite(arr)].copy()
    if len(r) < 2:
        return np.nan
    for _ in range(max_iter):
        med, std = np.median(r), np.std(r)
        if not np.isfinite(std) or std == 0:
            break
        keep = np.abs(r - med) <= n_sigma * std
        if keep.sum() == len(r) or keep.sum() < 2:
            break
        r = r[keep]
    return float(np.std(r))

def compute_rolling_local_sigma(
    residuals: np.ndarray,
    window_pts: int = 100,
    flare_mask: np.ndarray | None = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Estimate the local noise at each cadence with a rolling MAD.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residual flux values.
    window_pts : int, optional
        Rolling window length, in cadences.
    flare_mask : numpy.ndarray or None, optional
        Boolean mask of cadences to exclude from the rolling estimate.
    min_periods : int, optional
        Minimum number of values required in each rolling window.

    Returns
    -------
    numpy.ndarray
        Per-cadence local sigma estimate. Missing rolling estimates are filled with
        the nearest valid estimate or a sigma-clipped global fallback.
    """
    r = residuals.copy().astype(float)
    if flare_mask is not None:
        r[flare_mask] = np.nan
    s = pd.Series(r)
    rolling_med = s.rolling(window_pts, center=True, min_periods=min_periods).median()
    abs_dev = (s - rolling_med).abs()
    if flare_mask is not None:
        abs_dev.iloc[np.where(flare_mask)[0]] = np.nan
    rolling_mad = abs_dev.rolling(window_pts, center=True, min_periods=min_periods).median()
    sigma_local = (1.4826 * rolling_mad).to_numpy(dtype=float)
    sigma_local = pd.Series(sigma_local).ffill().bfill().to_numpy(dtype=float)
    finite = sigma_local[np.isfinite(sigma_local)]
    fallback = float(np.nanmedian(finite)) if len(finite) else sigma_clipped_std(residuals)
    return np.where(np.isfinite(sigma_local), sigma_local, fallback)

def combine_lc_series(per_lc: dict) -> pd.DataFrame:
    """Stack per-sector light curves into one sorted table.

    Parameters
    ----------
    per_lc : dict
        Dictionary keyed by light-curve index. Each value must contain ``time``,
        ``flux``, and ``flux_err`` arrays.

    Returns
    -------
    pandas.DataFrame
        Concatenated table with non-finite flux and flux-error rows removed and rows
        sorted by time.
    """
    frames = []
    for lc_idx, data in per_lc.items():
        df = pd.DataFrame({
            "lc_index": lc_idx,
            "time": np.asarray(data["time"], dtype=float),
            "flux": np.asarray(data["flux"], dtype=float),
            "flux_err": np.asarray(data["flux_err"], dtype=float),
        })
        frames.append(df[np.isfinite(df["flux"]) & np.isfinite(df["flux_err"])].copy())
    return pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)

def get_timeseries(base_path: str | Path, tic_id: int, quality_mask: int | None = None) -> pd.DataFrame:
    """Download TESS 120 s SPOC light curves and save a CSV file.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Output directory. A ``lightkurve_cache`` subdirectory is created inside it.
    tic_id : int
        TIC identifier.
    quality_mask : int or None, optional
        TESS quality mask. If None, ``DetrendConfig().quality_mask`` is used.

    Returns
    -------
    pandas.DataFrame
        Combined, quality-filtered light curve table.

    Raises
    ------
    ImportError
        If ``lightkurve`` is not installed.
    RuntimeError
        If no matching or usable TESS SPOC light curves are found.
    """
    if lk is None:
        raise ImportError("lightkurve is required for get_timeseries().")
    qmask = DetrendConfig().quality_mask if quality_mask is None else quality_mask
    outdir = Path(base_path)
    download_dir = outdir / "Data/lightkurve_cache"
    outdir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    lc_table = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
    if len(lc_table) == 0:
        raise RuntimeError(f"No TESS 120 s SPOC light curves found for TIC {tic_id}")
    lcc = lc_table.download_all(download_dir=str(download_dir))
    per_lc = {}
    for i, lc in enumerate(lcc):
        def _v(attr):
            """Return one light-curve field as a float array."""
            obj = getattr(lc, attr)
            return np.asarray(obj.value if hasattr(obj, "value") else obj, float)
        try:
            quality = _v("quality").astype(int)
            good = (quality & qmask) == 0
            per_lc[i] = {
                "time": _v("time")[good],
                "flux": _v("flux")[good],
                "flux_err": _v("flux_err")[good],
            }
        except (AttributeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Skipping light curve {i}: it could not be read correctly ({exc}).",
                RuntimeWarning,
            )
    if not per_lc:
        raise RuntimeError(f"No usable light curves were downloaded for TIC {tic_id}.")
    combined = combine_lc_series(per_lc)
    if combined.empty:
        raise RuntimeError(f"All downloaded light curves for TIC {tic_id} were empty after filtering.")
    combined.to_csv(outdir / f"Data/{tic_id}_timeseries.csv", index=False)
    return combined

def prepare_arrays(ts_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sorted numeric arrays from a light-curve table.

    Invalid time and flux rows are dropped. Invalid ``flux_err`` values are replaced
    with the median positive finite flux-error value.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Table containing ``time``, ``flux``, and ``flux_err`` columns.

    Returns
    -------
    time : numpy.ndarray
        Sorted finite time values.
    flux : numpy.ndarray
        Flux values corresponding to ``time``.
    flux_err : numpy.ndarray
        Positive finite flux-error values corresponding to ``time``.

    Raises
    ------
    ValueError
        If required columns are missing, no valid time/flux rows remain, or no
        positive finite ``flux_err`` values are available.
    """
    req = {"time", "flux", "flux_err"}
    missing = req.difference(ts_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = ts_df.loc[:, ["time", "flux", "flux_err"]].copy()
    before = len(df)
    df = df[np.isfinite(df["time"]) & np.isfinite(df["flux"])].sort_values("time")
    dropped = before - len(df)
    if dropped:
        warnings.warn(
            f"Dropped {dropped} row(s) with invalid time or flux values.",
            RuntimeWarning,
        )
    if df.empty:
        raise ValueError("No valid rows remain after removing invalid time or flux values.")
    time = df["time"].to_numpy(dtype=float)
    flux = df["flux"].to_numpy(dtype=float)
    flux_err = df["flux_err"].to_numpy(dtype=float)
    bad = ~(np.isfinite(flux_err) & (flux_err > 0))
    if bad.any():
        good = flux_err[~bad]
        if len(good) == 0:
            raise ValueError("flux_err contains no positive finite values.")
        replacement = float(np.nanmedian(good))
        flux_err[bad] = replacement
        warnings.warn(
            f"Replaced {int(bad.sum())} invalid flux_err value(s) with the median value {replacement:.6g}.",
            RuntimeWarning,
        )
    return time, flux, flux_err

def remove_gap_edges(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_remove: int = 125,
    gap_sigma: float = 8.0,
    gap_min_factor: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove cadences around large gaps in a time series.

    Parameters
    ----------
    time : numpy.ndarray
        Time values.
    flux : numpy.ndarray
        Flux values.
    flux_err : numpy.ndarray
        Flux-error values.
    n_remove : int, optional
        Number of cadences to remove on both sides of each detected gap edge.
    gap_sigma : float, optional
        Robust sigma multiplier used in the gap threshold.
    gap_min_factor : float, optional
        Minimum threshold as a multiple of the median cadence.

    Returns
    -------
    time : numpy.ndarray
        Gap-cleaned, time-sorted time values.
    flux : numpy.ndarray
        Gap-cleaned flux values.
    flux_err : numpy.ndarray
        Gap-cleaned flux-error values.
    info : dict
        Gap-cleaning diagnostics, including cadence, threshold, gap indices, keep
        mask, and number of removed cadences.

    Raises
    ------
    ValueError
        If the input arrays have different lengths, fewer than two cadences are
        available, or all cadences would be removed.
    """
    if not (len(time) == len(flux) == len(flux_err)):
        raise ValueError("time, flux, and flux_err must have the same length before gap cleaning.")
    if len(time) < 2:
        raise ValueError("At least two cadences are needed for gap cleaning.")
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    dt = np.diff(time)
    cadence_days = float(np.nanmedian(dt))
    mad_dt = float(np.nanmedian(np.abs(dt - cadence_days)))
    robust_sigma = 1.4826 * mad_dt
    gap_threshold = max(gap_min_factor * cadence_days, cadence_days + gap_sigma * robust_sigma)
    gap_indices = np.where(dt > gap_threshold)[0]
    keep = np.ones(len(time), dtype=bool)
    for gi in gap_indices:
        center = gi + 1
        keep[max(0, center - n_remove): min(len(time), center + n_remove + 1)] = False
    info = {
        "cadence_days": cadence_days,
        "gap_threshold": gap_threshold,
        "gap_indices": gap_indices,
        "keep_mask": keep,
        "n_removed": int((~keep).sum()),
    }
    if not keep.any():
        raise ValueError("Gap cleaning would remove every cadence.")
    return time[keep], flux[keep], flux_err[keep], info

def multi_sinusoid_correction(
    times,
    residuals,
    flux_err=None,
    flare_mask=None,
    sigma_local=None,
    n_components=3,
    min_period_hr=0.07,
    max_period_hr=0.10,
    amp_limit_frac=2.0,
    min_improvement=0.005,
    label="Sinusoidal correction",
    return_stats=False,
    peak_sharpness_min=0.0,
):
    """Remove dominant sinusoidal components from residuals.

    Components are accepted greedily and only kept when they produce a minimum
    additional scatter improvement. This prevents the function from always
    removing exactly ``n_components`` weak or alias-like sinusoids.

    When ``peak_sharpness_min`` is greater than zero, each candidate peak is
    additionally tested against a local-neighbourhood median: the ratio of the
    peak power to the median power of the surrounding ±5 % of the frequency
    axis must exceed ``peak_sharpness_min``.  Broad power humps caused by
    granulation typically fail this test, preventing the correction from
    absorbing granulation power as spurious coherent sinusoids.
    """
    default_stats = {
        "applied": False,
        "n_components": 0,
        "periods_hr": [],
        "improvement": 0.0,
        "std_before": np.nan,
        "std_after": np.nan,
        "boundary_hit": False,
        "reject_reason": "",
    }

    def _finish(corrected, stats):
        return (corrected, stats) if return_stats else corrected

    if not _ASTROPY_LS:
        warnings.warn(
            f"{label} skipped because astropy LombScargle is unavailable.",
            RuntimeWarning,
        )
        default_stats["reject_reason"] = "astropy unavailable"
        return _finish(residuals, default_stats)

    times = np.asarray(times, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    if flare_mask is None:
        flare_mask = np.zeros(len(times), dtype=bool)
    else:
        flare_mask = np.asarray(flare_mask, dtype=bool)

    q_mask = ~flare_mask & np.isfinite(times) & np.isfinite(residuals)
    t_q, r_q = times[q_mask], residuals[q_mask]
    if len(t_q) < 30:
        default_stats["reject_reason"] = "too few points"
        return _finish(residuals, default_stats)

    if sigma_local is not None:
        sigma_arr = np.asarray(sigma_local, dtype=float)
        sigma_med = float(np.nanmedian(sigma_arr[np.isfinite(sigma_arr)]))
    else:
        sigma_med = 1.4826 * float(np.nanmedian(np.abs(r_q - np.nanmedian(r_q))))
    if not np.isfinite(sigma_med) or sigma_med <= 0:
        sigma_med = float(np.nanstd(r_q))
    amp_cap = amp_limit_frac * sigma_med if np.isfinite(sigma_med) else np.inf

    duration = float(t_q.max() - t_q.min())
    min_freq = 24.0 / max_period_hr
    max_freq = 24.0 / min_period_hr
    natural_min_freq = 1.0 / duration if duration > 0 else np.inf
    min_freq = max(min_freq, natural_min_freq)

    if min_freq >= max_freq:
        default_stats["reject_reason"] = "period range invalid for duration"
        return _finish(residuals, default_stats)

    try:
        ls = LombScargle(t_q, r_q)
        freqs, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=10,
        )
    except Exception as exc:
        warnings.warn(
            f"{label} skipped: Lomb-Scargle fitting failed ({exc}).",
            RuntimeWarning,
        )
        default_stats["reject_reason"] = "Lomb-Scargle failed"
        return _finish(residuals, default_stats)

    if len(freqs) == 0:
        default_stats["reject_reason"] = "empty periodogram"
        return _finish(residuals, default_stats)

    def _design(t_arr, selected_freqs):
        cols = [np.ones(len(t_arr))]
        for f in selected_freqs:
            cols += [np.sin(2 * np.pi * f * t_arr), np.cos(2 * np.pi * f * t_arr)]
        return np.column_stack(cols)

    def _fit_correct(selected_freqs):
        X_q = _design(t_q, selected_freqs)
        try:
            fe_q = np.asarray(flux_err, dtype=float)[q_mask] if flux_err is not None else None
            if fe_q is not None and np.all(fe_q > 0) and np.all(np.isfinite(fe_q)):
                w = 1.0 / fe_q**2
                coeffs = np.linalg.lstsq((X_q.T * w) @ X_q, (X_q.T * w) @ r_q, rcond=None)[0]
            else:
                coeffs = np.linalg.lstsq(X_q, r_q, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None, None

        model = np.full(len(times), float(coeffs[0]))
        for k, f in enumerate(selected_freqs):
            A, B = float(coeffs[1 + 2*k]), float(coeffs[2 + 2*k])
            amp = np.sqrt(A**2 + B**2)
            if amp > amp_cap > 0:
                A *= amp_cap / amp
                B *= amp_cap / amp
            model += A * np.sin(2 * np.pi * f * times) + B * np.cos(2 * np.pi * f * times)
        return residuals - model, model

    std_before = float(np.nanstd(residuals[q_mask]))
    if not np.isfinite(std_before) or std_before <= 0:
        default_stats["reject_reason"] = "invalid initial scatter"
        return _finish(residuals, default_stats)

    selected = []
    best_corrected = residuals.copy()
    best_std = std_before
    max_candidates = min(len(freqs), max(50, 25 * int(max(n_components, 1))))

    for idx in np.argsort(power)[::-1][:max_candidates]:
        f = float(freqs[idx])
        # Avoid selecting nearly duplicate frequencies.
        if any(abs(f - sf) < 0.10 * max(f, sf) for sf in selected):
            continue
        if len(selected) >= n_components:
            break

        # Granulation sharpness guard: reject peaks that sit on a broad power
        # hump (ratio of peak power to local-neighbourhood median is too low).
        if peak_sharpness_min > 0 and len(freqs) >= 5:
            half_width = max(3, len(freqs) // 20)  # ±5 % of frequency axis
            lo_i = max(0, idx - half_width)
            hi_i = min(len(power), idx + half_width + 1)
            neighbourhood = np.concatenate([power[lo_i:idx], power[idx + 1:hi_i]])
            if len(neighbourhood) > 0:
                local_med = float(np.nanmedian(neighbourhood))
                if local_med > 0:
                    sharpness = float(power[idx]) / local_med
                    if sharpness < peak_sharpness_min:
                        continue  # broad hump – skip this candidate

        trial_selected = selected + [f]
        trial_corrected, _ = _fit_correct(trial_selected)
        if trial_corrected is None:
            continue

        trial_std = float(np.nanstd(trial_corrected[q_mask]))
        if not np.isfinite(trial_std):
            continue

        total_improvement = (std_before - trial_std) / std_before
        marginal_improvement = (best_std - trial_std) / std_before

        # Accept the first component based on total improvement, and later
        # components based on marginal improvement. This avoids always removing
        # the configured maximum number of components.
        if not selected:
            accept = total_improvement >= min_improvement and trial_std < best_std
        else:
            accept = marginal_improvement >= min_improvement and trial_std < best_std

        if accept:
            selected = trial_selected
            best_corrected = trial_corrected
            best_std = trial_std

    improvement = (std_before - best_std) / std_before
    if not selected or improvement < min_improvement:
        print(f"  {label} discarded (improvement too small: {100 * max(improvement, 0.0):.2f}%).")
        default_stats.update({
            "reject_reason": "improvement too small",
            "std_before": std_before,
            "std_after": best_std,
            "improvement": float(max(improvement, 0.0)),
        })
        return _finish(residuals, default_stats)

    if best_std > std_before:
        print(f"  {label} discarded (would increase noise).")
        default_stats.update({
            "reject_reason": "would increase noise",
            "std_before": std_before,
            "std_after": best_std,
        })
        return _finish(residuals, default_stats)

    periods = [24.0 / f for f in selected]
    periods_hr = [f"{p:.3f} hr" for p in periods]
    boundary_hit = any(
        (p <= min_period_hr * 1.02) or (p >= max_period_hr * 0.98)
        for p in periods
    )
    boundary_note = " [boundary hit]" if boundary_hit else ""
    print(
        f"  {label}: removed {len(selected)} sinusoidal component(s): "
        f"periods = {periods_hr}; scatter improvement = {100 * improvement:.2f}%{boundary_note}"
    )
    stats = {
        "applied": True,
        "n_components": int(len(selected)),
        "periods_hr": [float(p) for p in periods],
        "improvement": float(improvement),
        "std_before": float(std_before),
        "std_after": float(best_std),
        "boundary_hit": bool(boundary_hit),
        "reject_reason": "",
    }
    return _finish(best_corrected, stats)


def _as_float_or_none(value):
    """Return a finite float or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _find_file_upwards_and_downwards(filename: str, roots: list[Path]) -> Path | None:
    """Find ``filename`` near one of ``roots`` without raising on permission errors."""
    checked: list[Path] = []
    for root in roots:
        if root is None:
            continue
        try:
            root = Path(root).expanduser().resolve()
        except Exception:
            continue
        candidates = []
        if root.is_file():
            candidates.append(root.parent)
        else:
            candidates.append(root)
        candidates.extend(list(candidates[0].parents))
        for base in candidates:
            if base in checked:
                continue
            checked.append(base)
            direct = base / filename
            if direct.exists():
                return direct
            try:
                # Limit recursive searching to likely project/data folders; this keeps
                # automatic discovery useful without making ordinary detrending slow.
                matches = list(base.glob(f"**/{filename}"))
            except Exception:
                matches = []
            if matches:
                return matches[0]
    return None


def load_known_rotation_period_days(base_path: str | Path | None, tic_id: int | str | None) -> float | None:
    """Load a saved stellar-rotation period for a TIC if a local CSV exists.

    The usual file is ``<base_path>/<tic_id>_df_stellar_rotation.csv`` and it
    must contain a ``stellar_rotation_period`` column in days. For convenience,
    this function also searches nearby parent/project folders when ``base_path``
    points to a project root rather than the exact TIC data directory. Missing
    files, missing columns, and invalid values return ``None`` instead of
    raising.
    """
    if tic_id is None:
        return None
    tic = str(tic_id)
    filename = f"{tic}_df_stellar_rotation.csv"

    roots: list[Path] = []
    if base_path is not None:
        try:
            roots.append(Path(base_path))
        except Exception:
            pass
    roots.append(Path.cwd())

    path = _find_file_upwards_and_downwards(filename, roots)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "stellar_rotation_period" not in df.columns:
        return None
    return _as_float_or_none(df.loc[0, "stellar_rotation_period"])


def _infer_detrending_context_from_caller() -> dict:
    """Infer TIC/base-path metadata from the notebook/script that called us.

    This keeps the public call simple::

        result, gap_info = detrend_dataframe(ts_df, clean_gaps=True)

    while still allowing the detrending code to find a previously saved
    ``<TIC>_df_stellar_rotation.csv`` for fast-rotator harmonic correction.
    The function looks for common notebook variables such as ``TIC_ID``,
    ``tic_id``, ``BASE_PATH``, ``base_path``, and ``OUTPUT_CSV`` in the caller's
    globals/locals. Nothing here is required; failure simply returns an empty
    context and detrending proceeds normally.
    """
    ctx: dict = {}
    try:
        frame = inspect.currentframe()
        # current -> helper, caller -> detrend_dataframe, caller's caller -> user code
        user_frame = frame.f_back.f_back if frame and frame.f_back and frame.f_back.f_back else None
        namespace = {}
        if user_frame is not None:
            namespace.update(user_frame.f_globals)
            namespace.update(user_frame.f_locals)

        for name in ("TIC_ID", "tic_id", "tic", "TIC"):
            value = namespace.get(name)
            if value is not None:
                try:
                    ctx["tic_id"] = str(int(value))
                except Exception:
                    ctx["tic_id"] = str(value)
                break

        for name in ("BASE_PATH", "base_path", "DATA_PATH", "data_path"):
            value = namespace.get(name)
            if value is not None:
                ctx["base_path"] = value
                break

        output_csv = namespace.get("OUTPUT_CSV") or namespace.get("output_csv")
        if output_csv is not None:
            try:
                out_path = Path(output_csv)
                ctx.setdefault("base_path", out_path.parent)
                if "tic_id" not in ctx:
                    stem = out_path.stem
                    digits = "".join(ch for ch in stem.split("_")[0] if ch.isdigit())
                    if digits:
                        ctx["tic_id"] = digits
            except Exception:
                pass
    except Exception:
        return {}
    return ctx

def attach_rotation_metadata_to_config(
    config: DetrendConfig | None = None,
    base_path: str | Path | None = None,
    tic_id: int | str | None = None,
    stellar_rotation_df: pd.DataFrame | None = None,
    known_rotation_period_days: float | None = None,
) -> DetrendConfig:
    """Return a config copy with a known stellar rotation period attached.

    Priority is explicit ``known_rotation_period_days``, then an in-memory
    ``stellar_rotation_df`` with a ``stellar_rotation_period`` column, then a
    local saved ``<tic_id>_df_stellar_rotation.csv`` under ``base_path``.
    """
    cfg = replace(config) if config is not None else DetrendConfig()

    period = _as_float_or_none(known_rotation_period_days)
    if period is None and stellar_rotation_df is not None:
        try:
            if not stellar_rotation_df.empty and "stellar_rotation_period" in stellar_rotation_df.columns:
                period = _as_float_or_none(stellar_rotation_df.iloc[0]["stellar_rotation_period"])
        except Exception:
            period = None
    if period is None:
        period = load_known_rotation_period_days(base_path, tic_id)

    if period is not None:
        cfg.known_rotation_period_days = period
        cfg.window_sizes = rotation_dependent_window_sizes(period, cfg.window_sizes)
    return cfg


def rotation_dependent_window_sizes(
    stellar_rotation_period_days: float | None,
    default_window_sizes: tuple[float, ...] = (0.4, 0.6, 0.8),
) -> tuple[float, ...]:
    """Choose detrending window sizes from the stellar rotation period.

    Parameters
    ----------
    stellar_rotation_period_days : float or None
        Known stellar rotation period in days.
    default_window_sizes : tuple of float, optional
        Window sizes to keep when no period is available, or when the period is
        between 1 and 2 days inclusive.

    Returns
    -------
    tuple of float
        Rotation-dependent window sizes in days.
    """
    period = _as_float_or_none(stellar_rotation_period_days)
    if period is None:
        return tuple(default_window_sizes)
    if period < 1.0:
        return (0.2, 0.4, 0.6, 0.8)
    if period > 2.0:
        return (0.4, 0.6, 0.8, 1.0)
    return tuple(default_window_sizes)


# ======================================================================= #
# Granulation helpers                                                       #
# ======================================================================= #

def granulation_safe_window_sizes(
    granulation_timescale_days: float,
    candidate_window_sizes: tuple[float, ...],
    min_window_factor: float = 3.0,
) -> tuple[float, ...]:
    """Return only window sizes that are safe given the granulation timescale.

    Windows shorter than ``min_window_factor * granulation_timescale_days``
    risk absorbing granulation power into the polynomial baseline instead of
    preserving it as astrophysical signal.  If all candidate windows are
    shorter than the safe limit, a set of three windows starting at the safe
    minimum is generated automatically so that detrending can still proceed.

    Parameters
    ----------
    granulation_timescale_days : float
        Estimated or known granulation timescale in days.
    candidate_window_sizes : tuple of float
        Current candidate window sizes, in days.
    min_window_factor : float, optional
        Minimum window expressed as a multiple of the granulation timescale.
        Default is 3.0.

    Returns
    -------
    tuple of float
        Filtered (and possibly extended) window sizes, all >= the safe minimum.
    """
    tau = float(granulation_timescale_days)
    if not np.isfinite(tau) or tau <= 0:
        return tuple(candidate_window_sizes)
    min_safe = min_window_factor * tau
    safe = tuple(w for w in candidate_window_sizes if w >= min_safe)
    if not safe:
        # Auto-generate three windows above the safe floor.
        safe = (min_safe, min_safe * 1.5, min_safe * 2.0)
        warnings.warn(
            f"All candidate window sizes are shorter than the granulation-safe "
            f"minimum of {min_safe:.3f} d ({min_window_factor}× τ_gran = "
            f"{tau:.3f} d).  Using auto-generated windows: "
            f"{[f'{w:.3f}' for w in safe]} d.",
            RuntimeWarning,
        )
    return safe


def estimate_granulation_timescale(
    residuals: np.ndarray,
    time: np.ndarray,
    flare_mask: np.ndarray | None = None,
    max_lag_days: float = 3.0,
) -> float | None:
    """Estimate the granulation timescale from the ACF of quiescent residuals.

    The granulation timescale is approximated as the lag at which the
    normalised autocorrelation function of the quiescent residuals first
    crosses zero.  A positive first-zero-crossing indicates coherent correlated
    noise on that timescale.

    Parameters
    ----------
    residuals : numpy.ndarray
        Detrended residual flux values (e.g. second-pass residuals).
    time : numpy.ndarray
        Time values in days, same length as ``residuals``.
    flare_mask : numpy.ndarray or None, optional
        Boolean mask; True cadences are excluded before computing the ACF.
    max_lag_days : float, optional
        Maximum lag in days to search for the first zero-crossing.

    Returns
    -------
    float or None
        Estimated granulation timescale in days, or ``None`` if it cannot be
        determined reliably.
    """
    r = np.asarray(residuals, dtype=float).copy()
    t = np.asarray(time, dtype=float)
    if flare_mask is not None:
        r[np.asarray(flare_mask, dtype=bool)] = np.nan

    finite = np.isfinite(r) & np.isfinite(t)
    if finite.sum() < 30:
        return None

    r_q = r[finite] - np.nanmean(r[finite])
    dt = float(np.nanmedian(np.diff(t[finite])))
    if not np.isfinite(dt) or dt <= 0:
        return None

    max_lag_pts = max(1, int(max_lag_days / dt))
    n = len(r_q)
    max_lag_pts = min(max_lag_pts, n - 1)

    # Full ACF via numpy correlate (unbiased normalisation).
    norm = float(np.dot(r_q, r_q))
    if norm <= 0:
        return None
    full = np.correlate(r_q, r_q, mode="full")
    acf = full[n - 1: n - 1 + max_lag_pts + 1] / norm

    # Find first zero-crossing (positive → negative or from positive domain).
    sign_changes = np.where(np.diff(np.sign(acf)))[0]
    if len(sign_changes) == 0:
        return None
    # Use only crossings that start from the positive side.
    pos_crossings = sign_changes[acf[sign_changes] > 0]
    if len(pos_crossings) == 0:
        return None
    first_crossing_pts = int(pos_crossings[0])
    timescale = first_crossing_pts * dt
    return float(timescale) if timescale > 0 else None


def split_continuous_blocks(times: np.ndarray, min_points: int = 1, gap_factor: float = 10.0) -> list[np.ndarray]:
    """Return index arrays for continuous observing blocks split by large gaps."""
    times = np.asarray(times, dtype=float)
    finite = np.isfinite(times)
    if finite.sum() == 0:
        return []
    order = np.argsort(times)
    t = times[order]
    dt = np.diff(t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(finite_dt) == 0:
        return [order[finite[order]]] if finite.sum() >= min_points else []
    cadence = float(np.nanmedian(finite_dt))
    mad = float(np.nanmedian(np.abs(finite_dt - cadence)))
    robust_sigma = 1.4826 * mad
    gap_threshold = max(gap_factor * cadence, cadence + 8.0 * robust_sigma)
    split_after = np.where(dt > gap_threshold)[0]
    starts = np.r_[0, split_after + 1]
    stops = np.r_[split_after + 1, len(order)]
    blocks = []
    for a, b in zip(starts, stops):
        idx = order[a:b]
        idx = idx[np.isfinite(times[idx])]
        if len(idx) >= min_points:
            blocks.append(idx)
    return blocks


def known_rotation_harmonic_correction(
    times: np.ndarray,
    residuals: np.ndarray,
    rotation_period_days: float | None,
    flux_err: np.ndarray | None = None,
    flare_mask: np.ndarray | None = None,
    sigma_local: np.ndarray | None = None,
    harmonics: tuple[float, ...] = (0.5, 1, 2, 3, 4, 5),
    fast_rotation_max_period_days: float = 1.0,
    min_improvement: float = 0.02,
    min_points: int = 120,
    min_cycles: float = 1.5,
    clip_sigma: float = 4.0,
    amp_limit_frac: float = 3.0,
    label: str = "Known-rotation harmonic correction",
    return_stats: bool = False,
):
    """Subtract a known fast-rotator harmonic model from residuals.

    For fast rotators, the residual modulation often appears at P_rot and at
    harmonics such as P_rot/2 and P_rot/3. This function fits those harmonics
    simultaneously, separately in each continuous observing block, while
    excluding flare-masked points and using robust symmetric clipping. No
    constant term is subtracted; residual centering remains a separate stage.
    """
    default_stats = {
        "applied": False,
        "method": "known_fast_rotation_harmonics",
        "n_components": 0,
        "periods_hr": [],
        "improvement": 0.0,
        "std_before": np.nan,
        "std_after": np.nan,
        "boundary_hit": False,
        "reject_reason": "",
        "known_rotation_period_days": np.nan,
        "known_rotation_period_hr": np.nan,
        "harmonics": tuple(),
        "n_blocks_total": 0,
        "n_blocks_accepted": 0,
    }

    def _finish(corrected, model, stats):
        return (corrected, model, stats) if return_stats else corrected

    times = np.asarray(times, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    period_days = _as_float_or_none(rotation_period_days)
    if period_days is None or period_days <= 0:
        default_stats["reject_reason"] = "no known rotation period"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    default_stats["known_rotation_period_days"] = float(period_days)
    default_stats["known_rotation_period_hr"] = float(24.0 * period_days)

    if period_days > fast_rotation_max_period_days:
        default_stats["reject_reason"] = "not a fast rotator"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    harmonics = tuple(float(h) for h in harmonics if float(h) > 0)
    if not harmonics:
        default_stats["reject_reason"] = "no harmonics configured"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    if flare_mask is None:
        flare_mask = np.zeros(len(times), dtype=bool)
    else:
        flare_mask = np.asarray(flare_mask, dtype=bool)

    if sigma_local is not None:
        sigma_arr = np.asarray(sigma_local, dtype=float)
        sigma_med = float(np.nanmedian(sigma_arr[np.isfinite(sigma_arr)]))
    else:
        q_tmp = ~flare_mask & np.isfinite(residuals)
        sigma_med = 1.4826 * float(np.nanmedian(np.abs(residuals[q_tmp] - np.nanmedian(residuals[q_tmp])))) if q_tmp.sum() else np.nan
    if not np.isfinite(sigma_med) or sigma_med <= 0:
        sigma_med = sigma_clipped_std(residuals[~flare_mask & np.isfinite(residuals)])
    amp_cap = amp_limit_frac * sigma_med if np.isfinite(sigma_med) and sigma_med > 0 else np.inf

    q_global = ~flare_mask & np.isfinite(times) & np.isfinite(residuals)
    if q_global.sum() < min_points:
        default_stats["reject_reason"] = "too few quiescent points"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    std_before = float(np.nanstd(residuals[q_global]))
    if not np.isfinite(std_before) or std_before <= 0:
        default_stats["reject_reason"] = "invalid initial scatter"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    freqs = np.asarray([h / period_days for h in harmonics], dtype=float)  # cycles/day

    def _design(t_arr):
        t0 = float(np.nanmedian(t_arr))
        x = t_arr - t0
        cols = []
        for f in freqs:
            cols += [np.sin(2 * np.pi * f * x), np.cos(2 * np.pi * f * x)]
        return np.column_stack(cols) if cols else np.empty((len(t_arr), 0))

    model = np.zeros(len(times), dtype=float)
    blocks = split_continuous_blocks(times, min_points=min_points)
    accepted_blocks = 0

    for block in blocks:
        block = np.asarray(block, dtype=int)
        duration = float(np.nanmax(times[block]) - np.nanmin(times[block])) if len(block) else 0.0
        if duration < min_cycles * period_days:
            continue
        fit_mask = block[~flare_mask[block] & np.isfinite(residuals[block]) & np.isfinite(times[block])]
        if len(fit_mask) < min_points:
            continue

        t_fit = times[fit_mask]
        r_fit = residuals[fit_mask]
        keep = np.ones(len(fit_mask), dtype=bool)
        coeffs = None

        for _ in range(4):
            if keep.sum() < max(min_points, 2 * len(freqs) + 8):
                break
            X = _design(t_fit[keep])
            y = r_fit[keep]
            try:
                if flux_err is not None:
                    fe = np.asarray(flux_err, dtype=float)[fit_mask][keep]
                    if np.all(np.isfinite(fe)) and np.all(fe > 0):
                        w = 1.0 / np.maximum(fe, 1e-12) ** 2
                        coeffs = np.linalg.lstsq((X.T * w) @ X, (X.T * w) @ y, rcond=None)[0]
                    else:
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                else:
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                coeffs = None
                break

            pred_all = _design(t_fit) @ coeffs
            resid_fit = r_fit - pred_all
            sig = 1.4826 * np.nanmedian(np.abs(resid_fit[keep] - np.nanmedian(resid_fit[keep])))
            if not np.isfinite(sig) or sig <= 0:
                sig = float(np.nanstd(resid_fit[keep]))
            if not np.isfinite(sig) or sig <= 0:
                break
            new_keep = np.abs(resid_fit) <= clip_sigma * sig
            if new_keep.sum() == keep.sum() and np.all(new_keep == keep):
                break
            keep = new_keep

        if coeffs is None:
            continue

        coeffs = np.asarray(coeffs, dtype=float).copy()
        # Cap each harmonic amplitude independently.
        for j in range(len(freqs)):
            A, B = coeffs[2*j], coeffs[2*j + 1]
            amp = float(np.hypot(A, B))
            if amp_cap > 0 and np.isfinite(amp_cap) and amp > amp_cap:
                coeffs[2*j] *= amp_cap / amp
                coeffs[2*j + 1] *= amp_cap / amp

        block_model = _design(times[block]) @ coeffs
        model[block] = block_model
        accepted_blocks += 1

    corrected = residuals - model
    std_after = float(np.nanstd(corrected[q_global]))
    improvement = (std_before - std_after) / std_before if np.isfinite(std_after) else 0.0

    default_stats.update({
        "std_before": float(std_before),
        "std_after": float(std_after) if np.isfinite(std_after) else np.nan,
        "improvement": float(max(improvement, 0.0)),
        "harmonics": tuple(harmonics),
        "periods_hr": [float(24.0 * period_days / h) for h in harmonics],
        "n_blocks_total": int(len(blocks)),
        "n_blocks_accepted": int(accepted_blocks),
    })

    if accepted_blocks == 0:
        print(f"  {label} discarded (no valid continuous blocks).")
        default_stats["reject_reason"] = "no valid continuous blocks"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    if not np.isfinite(std_after) or std_after >= std_before:
        print(f"  {label} discarded (would increase noise).")
        default_stats["reject_reason"] = "would increase noise"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    if improvement < min_improvement:
        print(f"  {label} discarded (improvement too small: {100 * max(improvement, 0.0):.2f}%).")
        default_stats["reject_reason"] = "improvement too small"
        return _finish(residuals, np.zeros(len(times), dtype=float), default_stats)

    default_stats["applied"] = True
    default_stats["n_components"] = int(len(harmonics))
    default_stats["reject_reason"] = ""
    period_text = [f"{p:.3f} hr" for p in default_stats["periods_hr"]]
    print(
        f"  {label}: P_rot = {24.0 * period_days:.3f} hr; "
        f"harmonics = {list(harmonics)}; periods = {period_text}; "
        f"blocks = {accepted_blocks}/{len(blocks)}; scatter improvement = {100 * improvement:.2f}%"
    )
    return _finish(corrected, model, default_stats)


def preliminary_flare_mask(
    time, flux, flux_err,
    sigma_thresh=5, rough_window_days=2.0, poly_deg=3, rolling_pts=100,
) -> np.ndarray:
    """Build a preliminary flare mask with a single rough detrend.

    Parameters
    ----------
    time : array-like
        Time values, in days.
    flux : array-like
        Flux values.
    flux_err : array-like
        Flux-error values.
    sigma_thresh : float, optional
        Positive-residual threshold in units of effective sigma.
    rough_window_days : float, optional
        Rough polynomial-fit segment length, in days.
    poly_deg : int, optional
        Polynomial degree used in each rough segment.
    rolling_pts : int, optional
        Rolling window length, in cadences, for the local sigma estimate.

    Returns
    -------
    numpy.ndarray
        Boolean array that is True for likely flare cadences.
    """
    seg_id   = np.floor((time - time.min()) / rough_window_days).astype(int)
    baseline = np.full(len(time), np.nan)

    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        if len(idx) < poly_deg + 2:
            continue
        t_loc = time[idx] - time[idx].mean()
        w_    = 1.0 / np.maximum(flux_err[idx], 1e-15)**2
        try:
            coeffs = np.polyfit(t_loc, flux[idx], deg=poly_deg, w=np.sqrt(w_))
            baseline[idx] = np.polyval(coeffs, t_loc)
        except np.linalg.LinAlgError:
            continue

    residuals = flux - baseline
    finite    = np.isfinite(residuals)
    res_clean = np.where(finite, residuals, 0.0)

    sigma_local  = compute_rolling_local_sigma(res_clean, window_pts=rolling_pts)
    sigma_total  = np.sqrt(sigma_local**2 + flux_err**2)
    sc_floor     = sigma_clipped_std(res_clean[finite]) if finite.sum() > 10 else 0.0
    eff_sigma    = np.maximum(sigma_total, sc_floor)

    mask = (res_clean > sigma_thresh * eff_sigma) & finite
    print(f"Preliminary flare mask: {mask.sum():,} cadences flagged "
          f"({100 * mask.sum() / len(time):.2f}% of {len(time):,})")
    return mask
def fit_window_grid(
    time, flux, flux_err, flare_mask,
    window_size, shift=0.0,
    poly_deg=3, n_edge=4, edge_weight=8.0, min_fit_pts=15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a segmented polynomial baseline for one grid.

    Parameters
    ----------
    time : array-like
        Time values, in days.
    flux : array-like
        Flux values to fit.
    flux_err : array-like
        Positive flux-error values used as fit weights.
    flare_mask : array-like
        Boolean mask of cadences excluded from the preferred segment fit.
    window_size : float
        Segment width, in days.
    shift : float, optional
        Time offset for the window grid start, in days. Use ``0.0`` for normal
        windows and ``0.5 * window_size`` for shifted windows.
    poly_deg : int, optional
        Polynomial degree used in each segment.
    n_edge : int, optional
        Number of fitted points at each segment edge to up-weight.
    edge_weight : float, optional
        Multiplicative edge-point weight.
    min_fit_pts : int, optional
        Minimum number of unmasked cadences required before falling back to all
        points in the segment.

    Returns
    -------
    baseline : numpy.ndarray
        Fitted trend at every cadence. Cadences in unfitted segments are NaN.
    seg_id : numpy.ndarray
        Integer segment index at every cadence.
    metrics : dict
        Per-segment fit-quality statistics.
    """
    # Build segment grid starting from time.min() + shift
    t_offset = time - (time.min() + shift)
    seg_id   = np.floor(t_offset / window_size).astype(int)

    baseline = np.full(len(time), np.nan)
    metrics  = {}

    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        if len(idx) < poly_deg + 2:
            continue

        t_seg, f_seg = time[idx], flux[idx]
        fe_seg = np.maximum(flux_err[idx], 1e-15)
        fm_seg = flare_mask[idx]

        t_local   = t_seg - t_seg.mean()
        n_total   = len(idx)
        n_masked  = int(fm_seg.sum())
        frac_mask = n_masked / n_total

        fit_idx = np.where(~fm_seg)[0]
        if len(fit_idx) < min_fit_pts:
            fit_idx = np.arange(n_total)   # fall back to all points

        t_fit  = t_local[fit_idx]
        f_fit  = f_seg[fit_idx]
        fe_fit = fe_seg[fit_idx]

        weights = 1.0 / fe_fit**2
        n_e = min(n_edge, len(t_fit) // 2)
        if n_e > 0:
            weights[:n_e]  *= edge_weight
            weights[-n_e:] *= edge_weight

        try:
            coeffs = np.polyfit(t_fit, f_fit, deg=poly_deg, w=np.sqrt(weights))
        except (np.linalg.LinAlgError, ValueError):
            continue

        baseline[idx] = np.polyval(coeffs, t_local)

        # Fit-quality statistics on non-flare points
        r_q  = f_fit - np.polyval(coeffs, t_fit)
        n_q  = len(r_q)
        k    = poly_deg + 1
        dof  = max(n_q - k, 1)

        ss_res = float(np.sum(r_q**2))
        ss_tot = float(np.sum((f_fit - f_fit.mean())**2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else np.nan
        r2_adj = (1.0 - (1.0 - r2) * (n_q - 1) / dof) if np.isfinite(r2) and dof > 0 else np.nan

        chi2     = float(np.sum((r_q / fe_fit)**2))
        red_chi2 = chi2 / dof
        rsd      = float(np.std(r_q))

        rss = max(ss_res, 1e-300)
        aic = n_q * np.log(rss / max(n_q, 1)) + 2.0 * k
        bic = n_q * np.log(rss / max(n_q, 1)) + k * np.log(max(n_q, 1))

        metrics[sid] = {
            "r2_adj":     r2_adj,
            "red_chi2":   red_chi2,
            "rsd":        rsd,
            "aic":        float(aic),
            "bic":        float(bic),
            "n_fit":      len(fit_idx),
            "n_total":    n_total,
            "n_masked":   n_masked,
            "frac_masked": frac_mask,
            "seg_start":  float(t_seg.min()),
            "seg_end":    float(t_seg.max()),
        }

    return baseline, seg_id, metrics


def run_all_windows(time, flux, flux_err, flare_mask, window_sizes,
                    poly_deg=3, n_edge=4, edge_weight=8.0, min_fit_pts=15):
    """Fit all window-size and grid-shift combinations.

    For each window size, two grids are evaluated: a normal grid beginning at
    ``time.min()`` and a shifted grid beginning at ``time.min() + 0.5 * window``.

    Parameters
    ----------
    time : array-like
        Time values, in days.
    flux : array-like
        Flux values to fit.
    flux_err : array-like
        Positive flux-error values used as fit weights.
    flare_mask : array-like
        Boolean mask of cadences excluded from preferred segment fits.
    window_sizes : iterable of float
        Segment widths, in days.
    poly_deg : int, optional
        Polynomial degree used in each segment.
    n_edge : int, optional
        Number of fitted points at each segment edge to up-weight.
    edge_weight : float, optional
        Multiplicative edge-point weight.
    min_fit_pts : int, optional
        Minimum number of unmasked cadences required before fallback.

    Returns
    -------
    dict
        Dictionary keyed by ``(window_size, shift)`` with baselines, segment ids,
        metrics, and grid labels.
    """
    results = {}
    print(f"{'Window':>8}  {'Grid':>8}  {'Segs':>5}  {'R2_adj':>8}  {'red_χ²':>8}  {'RSD':>10}")
    print("-" * 58)

    for w in window_sizes:
        for shift_frac, label in [(0.0, "normal"), (0.5, "shifted")]:
            shift = shift_frac * w
            bl, sids, mets = fit_window_grid(
                time, flux, flux_err, flare_mask,
                window_size=w, shift=shift,
                poly_deg=poly_deg, n_edge=n_edge,
                edge_weight=edge_weight, min_fit_pts=min_fit_pts,
            )
            results[(w, shift)] = {
                "baseline":    bl,
                "seg_ids":     sids,
                "metrics":     mets,
                "window_size": w,
                "shift":       shift,
                "label":       label,
            }

            vals = list(mets.values())
            if not vals:
                warnings.warn(
                    f"No valid segments were fitted for window {w:.2f} d ({label} grid).",
                    RuntimeWarning,
                )
            med_r2  = np.nanmedian([m["r2_adj"]   for m in vals]) if vals else np.nan
            med_rc  = np.nanmedian([m["red_chi2"]  for m in vals]) if vals else np.nan
            med_rsd = np.nanmedian([m["rsd"]       for m in vals]) if vals else np.nan
            print(f"{w:>8.2f}  {label:>8}  {len(mets):>5d}  "
                  f"{med_r2:>8.4f}  {med_rc:>8.3f}  {med_rsd:>10.5f}")

    print("All window grids fitted.")
    return results


def build_segment_stats_df(all_results: dict) -> pd.DataFrame:
    """Flatten per-segment metrics into one table.

    Parameters
    ----------
    all_results : dict
        Output from ``run_all_windows``.

    Returns
    -------
    pandas.DataFrame
        Table with one row per fitted segment and window grid.
    """
    rows = []
    for (w, shift), res in all_results.items():
        for sid, m in res["metrics"].items():
            rows.append({
                "window_size":  w,
                "shift":        shift,
                "window_label": res["label"],
                "seg_id":       sid,
                "seg_start":    m["seg_start"],
                "seg_end":      m["seg_end"],
                "r2_adj":       m["r2_adj"],
                "red_chi2":     m["red_chi2"],
                "rsd":          m["rsd"],
                "aic":          m["aic"],
                "bic":          m["bic"],
                "n_fit":        m["n_fit"],
                "n_total":      m["n_total"],
                "n_masked":     m["n_masked"],
                "frac_masked":  m["frac_masked"],
            })
    return pd.DataFrame(rows)


def score_segment(
    m: dict,
    min_fit_pts: int = 15,
    granulation_timescale_days: float | None = None,
    granulation_min_window_factor: float = 3.0,
    granulation_score_penalty: float = 0.5,
) -> float:
    """Compute a scalar quality score for one fitted segment.

    The score combines adjusted R², reduced chi-square distance from one,
    fraction of masked flare points, and the number of fitted points.

    When ``granulation_timescale_days`` is provided, an additional penalty is
    applied to segments whose window length falls below
    ``granulation_min_window_factor * granulation_timescale_days``.  The
    penalty scales linearly from ``granulation_score_penalty`` (at zero window
    length) to zero (at the minimum safe window length), steering the softmax
    blending away from windows that are too short to avoid fitting out
    granulation.

    Parameters
    ----------
    m : dict
        Segment metric dictionary.
    min_fit_pts : int, optional
        Minimum number of fitted points required for a finite score.
    granulation_timescale_days : float or None, optional
        Known or estimated granulation timescale in days.  If None, no
        granulation penalty is applied.
    granulation_min_window_factor : float, optional
        Minimum safe window expressed as a multiple of the granulation
        timescale.  Default is 3.0.
    granulation_score_penalty : float, optional
        Maximum penalty subtracted for windows that are too short.
        Default is 0.5.

    Returns
    -------
    float
        Segment score. Returns ``-numpy.inf`` for disqualified segments.
    """
    r2_adj     = m.get("r2_adj",     np.nan)
    red_chi2   = m.get("red_chi2",   np.nan)
    n_fit      = m.get("n_fit",      0)
    frac_mask  = m.get("frac_masked", 1.0)

    # Hard disqualifiers
    if not np.isfinite(r2_adj):
        return -np.inf
    if n_fit < min_fit_pts:
        return -np.inf
    if frac_mask > 0.85:
        return -np.inf

    rc = np.clip(red_chi2, 1e-6, 1e6) if np.isfinite(red_chi2) else 1e6

    score = (
        r2_adj                             # [0, 1]: higher is better
        - 0.15 * abs(np.log(rc))           # penalise chi² far from 1
        - 0.30 * frac_mask                 # penalise high flare fraction
        + 0.05 * np.log1p(n_fit / 50.0)    # small bonus for more data
    )

    # Granulation window-length penalty.
    tau = _as_float_or_none(granulation_timescale_days)
    if tau is not None and tau > 0 and np.isfinite(granulation_score_penalty) and granulation_score_penalty > 0:
        min_safe = granulation_min_window_factor * tau
        seg_start = m.get("seg_start", np.nan)
        seg_end   = m.get("seg_end",   np.nan)
        if np.isfinite(seg_start) and np.isfinite(seg_end):
            window_len = float(seg_end - seg_start)
            if window_len < min_safe and min_safe > 0:
                # Linear penalty: full at window_len=0, zero at window_len=min_safe.
                frac_short = max(0.0, 1.0 - window_len / min_safe)
                score -= granulation_score_penalty * frac_short

    return float(score)


def _window_fit_column_suffix(window_size: float, label: str) -> str:
    """Return a safe column suffix for one window/grid combination."""
    w_txt = f"{float(window_size):g}".replace("-", "m").replace(".", "p")
    return f"w{w_txt}_{label}"


def add_window_fit_columns(
    final_df: pd.DataFrame,
    sort_idx: np.ndarray,
    all_results: dict,
    prefix: str,
    min_fit_pts: int = 15,
    granulation_timescale_days: float | None = None,
    granulation_min_window_factor: float = 3.0,
    granulation_score_penalty: float = 0.5,
) -> None:
    """Add every fitted window/grid baseline and score to ``final_df``.

    The columns are added in-place. For every window size and grid shift, three
    columns are added: the fitted baseline, the corresponding segment score, and
    the segment id that produced that fit.
    """
    for (_window_size, _shift), res in all_results.items():
        suffix = _window_fit_column_suffix(res["window_size"], res["label"])
        baseline = np.asarray(res["baseline"], dtype=float)
        seg_ids = np.asarray(res["seg_ids"])
        seg_score_map = {
            sid: score_segment(
                m,
                min_fit_pts=min_fit_pts,
                granulation_timescale_days=granulation_timescale_days,
                granulation_min_window_factor=granulation_min_window_factor,
                granulation_score_penalty=granulation_score_penalty,
            )
            for sid, m in res["metrics"].items()
        }
        scores = np.array([
            seg_score_map.get(int(sid), -np.inf) if np.isfinite(baseline[i]) else np.nan
            for i, sid in enumerate(seg_ids)
        ], dtype=float)

        final_df[f"{prefix}_fit_{suffix}"] = baseline[sort_idx]
        final_df[f"{prefix}_score_{suffix}"] = scores[sort_idx]
        final_df[f"{prefix}_segment_id_{suffix}"] = seg_ids[sort_idx]


def assemble_combined_trend(
    time: np.ndarray,
    all_results: dict,
    smooth_sigma_cadences: int = 5,
    min_fit_pts: int = 15,
    granulation_timescale_days: float | None = None,
    granulation_min_window_factor: float = 3.0,
    granulation_score_penalty: float = 0.5,
) -> tuple:
    """Build a blended per-cadence baseline from all window grids.

    At each cadence, valid segment fits are weighted by ``exp(score)`` using scores
    from ``score_segment``. This softmax-style weighting smooths transitions between
    competing fits.

    Parameters
    ----------
    time : numpy.ndarray
        Time values corresponding to the fitted baselines.
    all_results : dict
        Output from ``run_all_windows``.
    smooth_sigma_cadences : int, optional
        Gaussian smoothing width, in cadences. Set to zero to skip smoothing.
    min_fit_pts : int, optional
        Minimum number of fitted points required for segment scoring.
    granulation_timescale_days : float or None, optional
        Granulation timescale in days forwarded to ``score_segment`` for
        window-length penalty computation.  None disables the penalty.
    granulation_min_window_factor : float, optional
        Forwarded to ``score_segment``.  Default is 3.0.
    granulation_score_penalty : float, optional
        Forwarded to ``score_segment``.  Default is 0.5.

    Returns
    -------
    trend : numpy.ndarray
        Smoothed blended baseline.
    best_window_size : numpy.ndarray
        Window size of the highest-scoring candidate per cadence.
    best_shift : numpy.ndarray
        Grid shift of the highest-scoring candidate per cadence.
    best_seg_id : numpy.ndarray
        Segment id of the highest-scoring candidate per cadence.
    best_score : numpy.ndarray
        Score of the highest-scoring candidate per cadence.

    Raises
    ------
    RuntimeError
        If no finite trend can be built from the fitted window grids.
    """
    n         = len(time)
    keys      = list(all_results.keys())
    n_combos  = len(keys)

    bl_stack = np.full((n_combos, n), np.nan)
    sc_stack = np.full((n_combos, n), -np.inf)

    for ki, key in enumerate(keys):
        res  = all_results[key]
        bl   = res["baseline"]
        sids = res["seg_ids"]
        mets = res["metrics"]

        # Precompute segment scores (with optional granulation penalty)
        seg_score_map = {
            sid: score_segment(
                m,
                min_fit_pts,
                granulation_timescale_days=granulation_timescale_days,
                granulation_min_window_factor=granulation_min_window_factor,
                granulation_score_penalty=granulation_score_penalty,
            )
            for sid, m in mets.items()
        }

        bl_stack[ki] = bl
        for i in range(n):
            sid = int(sids[i])
            sc  = seg_score_map.get(sid, -np.inf)
            sc_stack[ki, i] = sc if np.isfinite(bl[i]) else -np.inf

    # Softmax-style weights: shift by per-cadence max before exponentiation
    valid  = np.isfinite(bl_stack) & (sc_stack > -1e5)
    sc_max = np.where(valid, sc_stack, -np.inf).max(axis=0, keepdims=True)
    sc_sh  = np.where(valid, sc_stack - sc_max, -np.inf)
    weights = np.where(sc_sh > -50, np.exp(sc_sh), 0.0)
    weights = np.where(valid, weights, 0.0)

    w_sum = weights.sum(axis=0)
    trend = np.where(w_sum > 0,
                     np.nansum(bl_stack * weights, axis=0) / w_sum,
                     np.nan)

    if not np.isfinite(trend).any():
        raise RuntimeError("No valid trend could be built from the fitted window grids.")

    # Fill residual NaNs by linear interpolation
    fin = np.isfinite(trend)
    if fin.sum() > 2 and (~fin).any():
        trend = np.interp(np.arange(n), np.where(fin)[0], trend[fin])

    # Light Gaussian smoothing to suppress window-boundary ringing
    if smooth_sigma_cadences > 0 and np.all(np.isfinite(trend)):
        trend = gaussian_filter1d(trend.astype(float), sigma=smooth_sigma_cadences)

    # Track the single best candidate at each cadence (for diagnostics)
    best_ki           = np.argmax(sc_stack, axis=0)
    best_window_size  = np.array([keys[best_ki[i]][0] for i in range(n)])
    best_shift        = np.array([keys[best_ki[i]][1] for i in range(n)])
    best_seg_id       = np.array([int(all_results[keys[best_ki[i]]]["seg_ids"][i]) for i in range(n)])
    best_score        = sc_stack[best_ki, np.arange(n)]

    return trend, best_window_size, best_shift, best_seg_id, best_score



def _robust_best_local_sinusoid(
    times: np.ndarray,
    residuals: np.ndarray,
    flux_err: np.ndarray | None = None,
    min_period_hr: float = 0.07,
    max_period_hr: float = 0.10,
    min_cycles: float = 1.5,
    amp_sigma: float = 1.5,
    min_improvement: float = 0.03,
    clip_sigma: float = 4.0,
) -> tuple[np.ndarray | None, dict]:
    """Fit one robust local sinusoid to a residual window.

    The initial search uses all finite points, including provisionally flare-masked
    points, so broad sinusoid crests can be recovered. Iterative symmetric clipping
    is then applied around the fitted sinusoid to reject sharp excursions before the
    final fit is accepted.

    Parameters
    ----------
    times : numpy.ndarray
        Local time values, in days.
    residuals : numpy.ndarray
        Local residual flux values.
    flux_err : numpy.ndarray or None, optional
        Local flux-error values used as least-squares weights when valid.
    min_period_hr : float, optional
        Minimum searched period, in hours.
    max_period_hr : float, optional
        Maximum searched period, in hours.
    min_cycles : float, optional
        Minimum number of cycles required in the local window.
    amp_sigma : float, optional
        Minimum accepted amplitude in units of robust pre-fit sigma.
    min_improvement : float, optional
        Minimum fractional robust-sigma improvement required for acceptance.
    clip_sigma : float, optional
        Symmetric clipping threshold used during robust fitting.

    Returns
    -------
    model : numpy.ndarray or None
        Accepted local sinusoid model, or None if no fit is accepted.
    info : dict
        Fit diagnostics and acceptance flag.
    """
    n = len(times)
    info = {
        "accepted": False,
        "period_hr": np.nan,
        "amplitude": np.nan,
        "sigma_before": np.nan,
        "sigma_after": np.nan,
        "improvement": np.nan,
        "n_fit": 0,
    }
    if not _ASTROPY_LS or n < 8:
        return None, info

    finite = np.isfinite(times) & np.isfinite(residuals)
    if finite.sum() < 8:
        return None, info

    t = times[finite]
    y = residuals[finite]
    duration = float(t.max() - t.min())
    if not np.isfinite(duration) or duration <= 0:
        return None, info

    min_freq = max(24.0 / max_period_hr, min_cycles / duration)
    max_freq = 24.0 / min_period_hr
    if min_freq >= max_freq:
        return None, info

    try:
        ls = LombScargle(t, y)
        freqs, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=10,
        )
    except Exception:
        return None, info
    if len(freqs) == 0 or not np.isfinite(power).any():
        return None, info

    f = float(freqs[np.nanargmax(power)])
    X = np.column_stack([
        np.ones(len(t)),
        np.sin(2 * np.pi * f * t),
        np.cos(2 * np.pi * f * t),
    ])

    keep = np.ones(len(t), dtype=bool)
    coeffs = None
    for _ in range(8):
        if keep.sum() < 8:
            return None, info
        Xk, yk = X[keep], y[keep]
        try:
            if flux_err is not None:
                fe = np.asarray(flux_err, dtype=float)[finite][keep]
                if np.all(np.isfinite(fe)) and np.all(fe > 0):
                    w = 1.0 / fe**2
                    coeffs = np.linalg.lstsq((Xk.T * w) @ Xk, (Xk.T * w) @ yk, rcond=None)[0]
                else:
                    coeffs = np.linalg.lstsq(Xk, yk, rcond=None)[0]
            else:
                coeffs = np.linalg.lstsq(Xk, yk, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None, info

        model = X @ coeffs
        err = y - model
        sigma = 1.4826 * np.nanmedian(np.abs(err - np.nanmedian(err)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = sigma_clipped_std(err)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_keep = np.abs(err - np.nanmedian(err)) <= clip_sigma * sigma
        if new_keep.sum() == keep.sum() and np.all(new_keep == keep):
            break
        keep = new_keep

    if coeffs is None or keep.sum() < 8:
        return None, info

    full_model = X @ coeffs
    amp = float(np.hypot(coeffs[1], coeffs[2]))
    sigma_before = sigma_clipped_std(y[keep])
    sigma_after = sigma_clipped_std((y - full_model)[keep])
    if not np.isfinite(sigma_before) or sigma_before <= 0 or not np.isfinite(sigma_after):
        return None, info
    improvement = float((sigma_before - sigma_after) / sigma_before)
    period_hr = float(24.0 / f)

    info.update({
        "period_hr": period_hr,
        "amplitude": amp,
        "sigma_before": sigma_before,
        "sigma_after": sigma_after,
        "improvement": improvement,
        "n_fit": int(keep.sum()),
    })

    accepted = (
        amp >= amp_sigma * sigma_before
        and improvement >= min_improvement
        and keep.sum() >= 8
    )
    info["accepted"] = bool(accepted)
    if not accepted:
        return None, info

    model = np.full(n, np.nan)
    model[finite] = full_model
    return model, info


def local_sinusoid_correction(
    times: np.ndarray,
    residuals: np.ndarray,
    flux_err: np.ndarray | None = None,
    provisional_flare_mask: np.ndarray | None = None,
    window_days: float = 2.0,
    step_fraction: float = 0.5,
    min_points: int = 60,
    min_period_hr: float = 0.07,
    max_period_hr: float = 0.10,
    min_cycles: float = 1.5,
    amp_sigma: float = 1.5,
    min_improvement: float = 0.03,
    clip_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Remove smooth periodic residual structure in local windows.

    The provisional flare mask is not used as a hard exclusion in the sinusoid
    search. This allows smooth sinusoid peaks that were initially misclassified as
    possible flares to be recovered. The robust fit rejects sharp deviations
    symmetrically around the sinusoid, and the caller should recompute the flare
    mask after correction.

    Parameters
    ----------
    times : numpy.ndarray
        Time values, in days.
    residuals : numpy.ndarray
        Residual flux values.
    flux_err : numpy.ndarray or None, optional
        Flux-error values used as least-squares weights when valid.
    provisional_flare_mask : numpy.ndarray or None, optional
        Boolean provisional flare mask used for diagnostics only.
    window_days : float, optional
        Local fitting window size, in days.
    step_fraction : float, optional
        Window step as a fraction of ``window_days``.
    min_points : int, optional
        Minimum finite cadences required in a local window.
    min_period_hr : float, optional
        Minimum searched period, in hours.
    max_period_hr : float, optional
        Maximum searched period, in hours.
    min_cycles : float, optional
        Minimum number of cycles required in a local window.
    amp_sigma : float, optional
        Minimum accepted amplitude in units of robust pre-fit sigma.
    min_improvement : float, optional
        Minimum fractional robust-sigma improvement required for acceptance.
    clip_sigma : float, optional
        Symmetric clipping threshold used during robust fitting.

    Returns
    -------
    corrected : numpy.ndarray
        Residuals after subtracting the blended local sinusoid model.
    model_unsorted : numpy.ndarray
        Local sinusoid model in the original input order.
    stats : pandas.DataFrame
        Per-window local sinusoid diagnostics.

    Raises
    ------
    ValueError
        If ``window_days`` is not positive, ``step_fraction`` is not in ``(0, 1]``,
        or input arrays do not have matching lengths.
    """
    times = np.asarray(times, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if provisional_flare_mask is None:
        provisional_flare_mask = np.zeros(len(times), dtype=bool)
    else:
        provisional_flare_mask = np.asarray(provisional_flare_mask, dtype=bool)

    if window_days <= 0 or not (0 < step_fraction <= 1):
        raise ValueError("window_days must be positive and step_fraction must lie in (0, 1].")
    if len(times) != len(residuals) or len(times) != len(provisional_flare_mask):
        raise ValueError("times, residuals, and provisional_flare_mask must have the same length.")

    order = np.argsort(times)
    t = times[order]
    r = residuals[order]
    fe = None if flux_err is None else np.asarray(flux_err, dtype=float)[order]

    model_sum = np.zeros(len(t), dtype=float)
    weight_sum = np.zeros(len(t), dtype=float)
    rows = []
    start = float(t.min())
    stop = float(t.max())
    step = window_days * step_fraction

    left = start
    wid = 0
    while left <= stop:
        right = left + window_days
        idx = np.where((t >= left) & (t <= right) & np.isfinite(r))[0]
        row = {
            "window_id": wid,
            "window_start": left,
            "window_end": right,
            "n_points": int(len(idx)),
            "n_provisional_masked": int(provisional_flare_mask[order][idx].sum()) if len(idx) else 0,
            "accepted": False,
            "period_hr": np.nan,
            "amplitude": np.nan,
            "sigma_before": np.nan,
            "sigma_after": np.nan,
            "improvement": np.nan,
            "n_fit": 0,
        }
        if len(idx) >= min_points:
            local_model, info = _robust_best_local_sinusoid(
                t[idx], r[idx], None if fe is None else fe[idx],
                min_period_hr=min_period_hr,
                max_period_hr=max_period_hr,
                min_cycles=min_cycles,
                amp_sigma=amp_sigma,
                min_improvement=min_improvement,
                clip_sigma=clip_sigma,
            )
            row.update(info)
            if local_model is not None:
                x = (t[idx] - left) / window_days
                taper = np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 2
                taper = np.maximum(taper, 1e-6)
                model_sum[idx] += np.nan_to_num(local_model, nan=0.0) * taper
                weight_sum[idx] += taper
        rows.append(row)
        left += step
        wid += 1

    local_model = np.zeros_like(model_sum, dtype=float)
    np.divide(model_sum, weight_sum, out=local_model, where=weight_sum > 0)
    corrected = residuals.copy()
    corrected[order] = r - local_model
    model_unsorted = np.zeros(len(times), dtype=float)
    model_unsorted[order] = local_model
    return corrected, model_unsorted, pd.DataFrame(rows)


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average.

    Parameters
    ----------
    arr : numpy.ndarray
        Values to smooth.
    window : int
        Rolling window length, in cadences.

    Returns
    -------
    numpy.ndarray
        Centred moving-average values.

    Raises
    ------
    ValueError
        If ``window`` is smaller than one.
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def below_zero_run_stats(arr: np.ndarray) -> dict:
    """Summarize how often and how long an array is below zero.

    Parameters
    ----------
    arr : numpy.ndarray
        Input values. Non-finite values are ignored for the denominator and never
        counted as below zero.

    Returns
    -------
    dict
        Fraction below zero, number of below-zero cadences, number of runs, longest
        run, and median run length.
    """
    x = np.asarray(arr, dtype=float)
    finite = np.isfinite(x)
    below = finite & (x < 0)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {
            "fraction_below_zero": np.nan,
            "n_below_zero": 0,
            "n_runs_below_zero": 0,
            "longest_run_below_zero": 0,
            "median_run_below_zero": np.nan,
        }

    run_lengths = []
    current = 0
    for is_below in below:
        if is_below:
            current += 1
        elif current:
            run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)

    return {
        "fraction_below_zero": float(below.sum() / n_finite),
        "n_below_zero": int(below.sum()),
        "n_runs_below_zero": int(len(run_lengths)),
        "longest_run_below_zero": int(max(run_lengths) if run_lengths else 0),
        "median_run_below_zero": float(np.median(run_lengths)) if run_lengths else np.nan,
    }


def center_residual_by_moving_average(
    residuals: np.ndarray,
    ma_window: int = 10,
    clip_abs: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Center residuals by subtracting the moving-average median.

    The correction is applied to the residual array itself, not only to the
    moving-average diagnostic. Because moving averages are linear, subtracting this
    offset from the residual shifts every recomputed moving average by the same
    amount.

    By default this function does not clip the moving-average values before
    estimating the offset. This avoids changing the centering population and keeps
    the reported moving-average median after centering close to zero.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residual flux values to center.
    ma_window : int, optional
        Moving-average window length, in cadences.
    clip_abs : float or None, optional
        Optional absolute clipping limit used when estimating the median offset. If
        None or non-positive, no absolute clipping is applied. If a positive value
        is supplied, the same cadence mask is used for the before and after
        diagnostics.

    Returns
    -------
    centered : numpy.ndarray
        Centered residual values.
    diagnostics : dict
        Centering diagnostics and below-zero run statistics before and after
        centering.
    """
    residuals = np.asarray(residuals, dtype=float)
    ma = moving_average(residuals, ma_window)

    # Build the centering sample once. With the default clip_abs=None this is all
    # finite moving-average values. If clipping is explicitly requested, keep the
    # same cadence mask for the before/after diagnostics so the median comparison
    # remains meaningful.
    center_mask = np.isfinite(ma)
    if clip_abs is not None and np.isfinite(clip_abs) and clip_abs > 0 and center_mask.any():
        center_mask &= np.abs(ma) < clip_abs

    if not center_mask.any():
        offset = 0.0
        centered = residuals.copy()
    else:
        offset = float(np.nanmedian(ma[center_mask]))
        centered = residuals - offset

    centered_ma = moving_average(centered, ma_window)
    before_sample = ma[center_mask]
    after_sample = centered_ma[center_mask]

    diagnostics = {
        "centering_applied": bool(center_mask.any()),
        "centering_ma_window": int(ma_window),
        "centering_clip_abs": None if clip_abs is None else float(clip_abs),
        "residual_offset_subtracted": float(offset),
        "ma_median_before_centering": float(np.nanmedian(before_sample)) if len(before_sample) else np.nan,
        "ma_median_after_centering": float(np.nanmedian(after_sample)) if len(after_sample) else np.nan,
        "n_centering_points": int(center_mask.sum()),
    }
    diagnostics.update({f"ma_before_{k}": v for k, v in below_zero_run_stats(ma).items()})
    diagnostics.update({f"ma_after_{k}": v for k, v in below_zero_run_stats(centered_ma).items()})
    return centered, diagnostics


def build_safe_final_flare_mask(
    residuals: np.ndarray,
    sigma_thresh: float = 3.0,
    previous_mask: np.ndarray | None = None,
    max_mask_fraction: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Build a positive-outlier flare mask with a guard against mask explosion.

    The final flare mask is recomputed after the periodic corrections and optional
    residual centering. In pathological cases a one-sided threshold can mark nearly
    the entire light curve as flaring, which then corrupts the later quiescent-noise
    estimate. This helper rejects such masks and falls back to the previous mask.

    Parameters
    ----------
    residuals : numpy.ndarray
        Final residual flux values.
    sigma_thresh : float, optional
        Positive-outlier threshold in robust sigma units.
    previous_mask : numpy.ndarray or None, optional
        Previous/provisional flare mask to use as the fallback and as the first
        quiescent sample for estimating the robust noise.
    max_mask_fraction : float, optional
        Maximum allowed fraction of finite cadences in the candidate final mask.
        If the candidate exceeds this fraction, ``previous_mask`` is returned.

    Returns
    -------
    final_mask : numpy.ndarray
        Safe final flare mask.
    diagnostics : dict
        Diagnostics describing whether the recomputed mask was accepted or rejected.
    """
    residuals = np.asarray(residuals, dtype=float)
    finite = np.isfinite(residuals)

    if previous_mask is None:
        previous_mask = np.zeros(len(residuals), dtype=bool)
    else:
        previous_mask = np.asarray(previous_mask, dtype=bool)
        if len(previous_mask) != len(residuals):
            raise ValueError("previous_mask must have the same length as residuals.")

    n_finite = int(finite.sum())
    if n_finite == 0:
        return previous_mask.copy(), {
            "final_mask_recomputed": False,
            "final_mask_reject_reason": "no finite residuals",
            "final_mask_candidate_fraction": np.nan,
            "final_mask_fraction": float(previous_mask.mean()) if len(previous_mask) else np.nan,
            "final_mask_center": np.nan,
            "final_mask_sigma": np.nan,
            "final_mask_candidate_count": 0,
        }

    # Estimate the noise from points not already suspected as flares. If too few
    # remain, use all finite residuals rather than estimating from a tiny sample.
    q = finite & ~previous_mask
    if q.sum() < 20:
        q = finite

    center = float(np.nanmedian(residuals[q]))
    mad = float(np.nanmedian(np.abs(residuals[q] - center)))
    robust_sigma = 1.4826 * mad

    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = sigma_clipped_std(residuals[q])

    previous_fraction = float((previous_mask & finite).sum() / n_finite)
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        return previous_mask.copy(), {
            "final_mask_recomputed": False,
            "final_mask_reject_reason": "invalid sigma",
            "final_mask_candidate_fraction": np.nan,
            "final_mask_fraction": previous_fraction,
            "final_mask_center": center,
            "final_mask_sigma": robust_sigma,
            "final_mask_candidate_count": 0,
        }

    candidate = finite & (residuals > center + sigma_thresh * robust_sigma)
    candidate_count = int(candidate.sum())
    candidate_fraction = float(candidate_count / n_finite)

    # Safety guard: flares should not be almost the whole light curve. If this
    # happens, keep the previous/provisional mask so the downstream noise estimate
    # still has a meaningful quiescent sample.
    max_mask_fraction = float(max_mask_fraction)
    if np.isfinite(max_mask_fraction) and max_mask_fraction > 0 and candidate_fraction > max_mask_fraction:
        return previous_mask.copy(), {
            "final_mask_recomputed": False,
            "final_mask_reject_reason": f"candidate mask fraction too high: {candidate_fraction:.4f}",
            "final_mask_candidate_fraction": candidate_fraction,
            "final_mask_fraction": previous_fraction,
            "final_mask_center": center,
            "final_mask_sigma": robust_sigma,
            "final_mask_candidate_count": candidate_count,
        }

    return candidate, {
        "final_mask_recomputed": True,
        "final_mask_reject_reason": "",
        "final_mask_candidate_fraction": candidate_fraction,
        "final_mask_fraction": candidate_fraction,
        "final_mask_center": center,
        "final_mask_sigma": robust_sigma,
        "final_mask_candidate_count": candidate_count,
    }

def run_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    config: DetrendConfig | None = None,
    external_flare_mask: np.ndarray | None = None,
) -> DetrendResult:
    """Run the full two-pass detrending pipeline.

    Parameters
    ----------
    time : numpy.ndarray
        Time values, in days.
    flux : numpy.ndarray
        Flux values.
    flux_err : numpy.ndarray
        Positive finite flux-error values.
    config : DetrendConfig or None, optional
        Pipeline configuration. If None, the default ``DetrendConfig`` is used.
    external_flare_mask : numpy.ndarray or None, optional
        Optional externally supplied boolean mask. True values are included in all
        flare masks used for fitting/noise estimates, but the original behavior is
        unchanged when this is None.

    Returns
    -------
    DetrendResult
        Final detrended table, segment statistics, local sinusoid statistics,
        summary diagnostics, and intermediate arrays.

    Raises
    ------
    ValueError
        If array lengths differ, there are too few cadences, non-finite values are
        present in required inputs, flux errors are not positive and finite, window
        sizes are invalid, or the polynomial degree is negative.
    RuntimeError
        If no valid combined trend can be built during either pass.
    """
    cfg = config or DetrendConfig()
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if not (len(time) == len(flux) == len(flux_err)):
        raise ValueError("time, flux, and flux_err must have the same length.")
    if len(time) < max(cfg.min_fit_pts, cfg.poly_deg + 2):
        raise ValueError("Not enough cadences to detrend.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")
    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values.")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("flux_err must contain only positive finite values.")
    if len(cfg.window_sizes) == 0 or any(w <= 0 for w in cfg.window_sizes):
        raise ValueError("window_sizes must contain at least one positive value.")
    if cfg.poly_deg < 0:
        raise ValueError("poly_deg must be zero or greater.")
    if cfg.min_fit_pts < cfg.poly_deg + 2:
        warnings.warn(
            "min_fit_pts is small for the polynomial degree; some fits may be unstable.",
            RuntimeWarning,
        )
    if not _ASTROPY_LS and cfg.apply_sinusoid_correction:
        warnings.warn(
            "Astropy LombScargle is unavailable, so sinusoidal correction will be skipped.",
            RuntimeWarning,
        )

    if external_flare_mask is None:
        external_flare_mask_arr = np.zeros(len(time), dtype=bool)
    else:
        external_flare_mask_arr = np.asarray(external_flare_mask, dtype=bool)
        if len(external_flare_mask_arr) != len(time):
            raise ValueError("external_flare_mask must have the same length as time, flux, and flux_err.")

    # ------------------------------------------------------------------ #
    # Granulation bootstrap                                                #
    # ------------------------------------------------------------------ #
    # Resolve the effective granulation timescale.  Priority:
    #   1. Explicit cfg.granulation_timescale_days (user-supplied).
    #   2. Auto-estimation from a rough first-pass ACF when
    #      cfg.auto_estimate_granulation is True.
    #   3. None  →  all granulation logic is disabled.
    gran_tau: float | None = _as_float_or_none(cfg.granulation_timescale_days)
    gran_tau_source: str = "user-supplied" if gran_tau is not None else "none"
    gran_tau_estimated: float | None = None  # stored for summary

    # Enforce granulation-safe window sizes from the user-supplied timescale
    # before the first pass.  Auto-estimation may tighten them further after
    # the first pass residuals are available.
    effective_window_sizes = tuple(cfg.window_sizes)
    if gran_tau is not None:
        effective_window_sizes = granulation_safe_window_sizes(
            gran_tau,
            effective_window_sizes,
            min_window_factor=cfg.granulation_min_window_factor,
        )
        print(
            f"Granulation: τ_gran = {gran_tau:.3f} d (user-supplied). "
            f"Safe window sizes: {list(effective_window_sizes)} d."
        )

    preliminary_flare_mask_internal = preliminary_flare_mask(
        time, flux, flux_err,
        sigma_thresh=cfg.flare_mask_sigma,
        rough_window_days=cfg.flare_mask_window_d,
        poly_deg=cfg.poly_deg,
        rolling_pts=cfg.rolling_window_pts,
    )
    flare_mask_prelim = preliminary_flare_mask_internal | external_flare_mask_arr

    all_results_p1 = run_all_windows(
        time, flux, flux_err, flare_mask_prelim, effective_window_sizes,
        poly_deg=cfg.poly_deg, n_edge=cfg.n_edge, edge_weight=cfg.edge_weight,
        min_fit_pts=cfg.min_fit_pts,
    )
    seg_stats_p1 = build_segment_stats_df(all_results_p1)
    fp_trend, fp_best_window, fp_best_shift, fp_best_seg, fp_best_score = assemble_combined_trend(
        time, all_results_p1, smooth_sigma_cadences=cfg.smooth_sigma_cadences,
        min_fit_pts=cfg.min_fit_pts,
        granulation_timescale_days=gran_tau,
        granulation_min_window_factor=cfg.granulation_min_window_factor,
        granulation_score_penalty=cfg.granulation_score_penalty,
    )
    fp_residuals = flux - fp_trend

    # Auto-estimate granulation timescale from first-pass residuals when
    # requested and no user-supplied value exists.
    if gran_tau is None and cfg.auto_estimate_granulation:
        gran_tau_estimated = estimate_granulation_timescale(
            fp_residuals, time, flare_mask=flare_mask_prelim,
            max_lag_days=cfg.granulation_acf_max_lag_days,
        )
        if gran_tau_estimated is not None:
            gran_tau = gran_tau_estimated
            gran_tau_source = "auto-estimated (ACF)"
            # Tighten window sizes for the second pass with the estimated τ.
            effective_window_sizes = granulation_safe_window_sizes(
                gran_tau,
                effective_window_sizes,
                min_window_factor=cfg.granulation_min_window_factor,
            )
            print(
                f"Granulation: τ_gran auto-estimated = {gran_tau:.3f} d. "
                f"Safe window sizes for second pass: {list(effective_window_sizes)} d."
            )
        else:
            print("Granulation: auto-estimation produced no reliable timescale estimate.")

    # Choose the short noise-floor window for flare detection when granulation
    # is active.  When gran_tau is None the standard rolling_window_pts is used
    # unchanged throughout.
    noise_window_pts = (
        cfg.granulation_noise_window_pts
        if gran_tau is not None
        else cfg.rolling_window_pts
    )

    if cfg.apply_sinusoid_correction and _ASTROPY_LS:
        sigma_prelim = compute_rolling_local_sigma(
            fp_residuals, window_pts=noise_window_pts, flare_mask=flare_mask_prelim
        )
        fp_residuals_corr, fp_sinusoid_stats = multi_sinusoid_correction(
            time, fp_residuals, flux_err=flux_err, flare_mask=flare_mask_prelim,
            sigma_local=sigma_prelim, n_components=cfg.n_sinusoid_components,
            min_period_hr=cfg.min_sinusoid_period_hr,
            max_period_hr=cfg.max_sinusoid_period_hr,
            amp_limit_frac=cfg.amp_limit_fraction,
            min_improvement=cfg.sinusoid_min_improvement,
            label="Short-period sinusoid correction (FP)",
            return_stats=True,
            peak_sharpness_min=cfg.granulation_sinusoid_sharpness_min if gran_tau is not None else 0.0,
        )
        fp_sinusoid_applied = bool(fp_sinusoid_stats.get("applied", False))
    else:
        fp_residuals_corr = fp_residuals.copy()
        fp_sinusoid_stats = {"applied": False, "n_components": 0, "periods_hr": [], "improvement": 0.0, "boundary_hit": False, "reject_reason": "disabled"}
        fp_sinusoid_applied = False
    fp_sinusoid_model = fp_residuals - fp_residuals_corr
    flux_p2_input = flux - fp_sinusoid_model
    fp_sigma_local = compute_rolling_local_sigma(
        fp_residuals_corr, window_pts=noise_window_pts, flare_mask=flare_mask_prelim
    )

    fp_global_std = sigma_clipped_std(fp_residuals_corr[~flare_mask_prelim & np.isfinite(fp_residuals_corr)])
    flare_mask_p2 = flare_mask_prelim | (
        np.isfinite(fp_residuals_corr) & (fp_residuals_corr > cfg.second_pass_sigma * fp_global_std)
    )
    all_results_p2 = run_all_windows(
        time, flux_p2_input, flux_err, flare_mask_p2, effective_window_sizes,
        poly_deg=cfg.poly_deg, n_edge=cfg.n_edge, edge_weight=cfg.edge_weight,
        min_fit_pts=cfg.min_fit_pts,
    )
    seg_stats_p2 = build_segment_stats_df(all_results_p2)
    sp_trend, sp_best_window, sp_best_shift, sp_best_seg, sp_best_score = assemble_combined_trend(
        time, all_results_p2, smooth_sigma_cadences=cfg.smooth_sigma_cadences,
        min_fit_pts=cfg.min_fit_pts,
        granulation_timescale_days=gran_tau,
        granulation_min_window_factor=cfg.granulation_min_window_factor,
        granulation_score_penalty=cfg.granulation_score_penalty,
    )
    sp_residuals = flux_p2_input - sp_trend
    if cfg.apply_sinusoid_correction and _ASTROPY_LS:
        sigma_p2_prelim = compute_rolling_local_sigma(
            sp_residuals, window_pts=noise_window_pts, flare_mask=flare_mask_p2
        )
        sp_residuals_corr, sp_sinusoid_stats = multi_sinusoid_correction(
            time, sp_residuals, flux_err=flux_err, flare_mask=flare_mask_p2,
            sigma_local=sigma_p2_prelim, n_components=cfg.n_sinusoid_components,
            min_period_hr=cfg.min_sinusoid_period_hr,
            max_period_hr=cfg.max_sinusoid_period_hr,
            amp_limit_frac=cfg.amp_limit_fraction,
            min_improvement=cfg.sinusoid_min_improvement,
            label="Short-period sinusoid correction (SP)",
            return_stats=True,
            peak_sharpness_min=cfg.granulation_sinusoid_sharpness_min if gran_tau is not None else 0.0,
        )
        sp_sinusoid_applied = bool(sp_sinusoid_stats.get("applied", False))
    else:
        sp_residuals_corr = sp_residuals.copy()
        sp_sinusoid_stats = {"applied": False, "n_components": 0, "periods_hr": [], "improvement": 0.0, "boundary_hit": False, "reject_reason": "disabled"}
        sp_sinusoid_applied = False
    sp_sinusoid_model = sp_residuals - sp_residuals_corr

    if cfg.apply_rotation_sinusoid_correction and _ASTROPY_LS:
        sigma_rotation = compute_rolling_local_sigma(
            sp_residuals_corr, window_pts=noise_window_pts, flare_mask=flare_mask_p2
        )

        rotation_residuals_corr = sp_residuals_corr.copy()
        rotation_sinusoid_model = np.zeros(len(time), dtype=float)
        # Initialise with "not_run" so that if neither branch fires the
        # summary reports an accurate reason rather than the stale "disabled".
        rotation_sinusoid_stats = {
            "applied": False, "method": "generic_periodogram", "n_components": 0,
            "periods_hr": [], "improvement": 0.0, "boundary_hit": False,
            "reject_reason": "not_run",
        }

        tried_fast_harmonic = (
            cfg.apply_fast_rotation_harmonic_correction
            and cfg.known_rotation_period_days is not None
            and np.isfinite(float(cfg.known_rotation_period_days))
            and float(cfg.known_rotation_period_days) <= cfg.fast_rotation_max_period_days
        )

        if tried_fast_harmonic:
            (
                rotation_residuals_corr,
                rotation_sinusoid_model,
                rotation_sinusoid_stats,
            ) = known_rotation_harmonic_correction(
                time, sp_residuals_corr, cfg.known_rotation_period_days,
                flux_err=flux_err, flare_mask=flare_mask_p2, sigma_local=sigma_rotation,
                harmonics=cfg.fast_rotation_harmonics,
                fast_rotation_max_period_days=cfg.fast_rotation_max_period_days,
                min_improvement=cfg.fast_rotation_min_improvement,
                min_points=cfg.fast_rotation_min_points,
                min_cycles=cfg.fast_rotation_min_cycles,
                clip_sigma=cfg.fast_rotation_clip_sigma,
                amp_limit_frac=cfg.fast_rotation_amp_limit_fraction,
                label="Known fast-rotation harmonic correction",
                return_stats=True,
            )

        should_run_generic_rotation = (
            not tried_fast_harmonic
            or not rotation_sinusoid_stats.get("applied", False)
            or cfg.run_generic_rotation_after_fast_harmonic
        )

        if should_run_generic_rotation:
            generic_input = rotation_residuals_corr.copy()
            generic_corr, generic_stats = multi_sinusoid_correction(
                time, generic_input, flux_err=flux_err, flare_mask=flare_mask_p2,
                sigma_local=sigma_rotation, n_components=cfg.rotation_sinusoid_n_components,
                min_period_hr=cfg.rotation_sinusoid_min_period_hr,
                max_period_hr=cfg.rotation_sinusoid_max_period_hr,
                amp_limit_frac=cfg.rotation_amp_limit_fraction,
                min_improvement=cfg.rotation_sinusoid_min_improvement,
                label="Long-period residual sinusoid correction",
                return_stats=True,
            )
            generic_model = generic_input - generic_corr
            if generic_stats.get("applied", False):
                if rotation_sinusoid_stats.get("applied", False):
                    # Both fast-harmonic and generic corrections applied —
                    # merge their stats.
                    prev_stats = rotation_sinusoid_stats.copy()
                    rotation_sinusoid_stats = generic_stats.copy()
                    rotation_sinusoid_stats["method"] = "known_harmonics_plus_generic_periodogram"
                    rotation_sinusoid_stats["known_harmonic_stats"] = prev_stats
                    rotation_sinusoid_stats["n_components"] = int(prev_stats.get("n_components", 0)) + int(generic_stats.get("n_components", 0))
                    rotation_sinusoid_stats["periods_hr"] = list(prev_stats.get("periods_hr", [])) + list(generic_stats.get("periods_hr", []))
                    rotation_sinusoid_stats["improvement"] = 1.0 - (1.0 - float(prev_stats.get("improvement", 0.0))) * (1.0 - float(generic_stats.get("improvement", 0.0)))
                else:
                    rotation_sinusoid_stats = generic_stats.copy()
                    rotation_sinusoid_stats["method"] = "generic_periodogram"
                rotation_residuals_corr = generic_corr
                rotation_sinusoid_model = rotation_sinusoid_model + generic_model
            else:
                # Generic correction ran but found nothing significant.
                # Propagate the actual reject reason from generic_stats so the
                # summary never shows the stale initialisation value.
                if not rotation_sinusoid_stats.get("applied", False):
                    rotation_sinusoid_stats["reject_reason"] = generic_stats.get(
                        "reject_reason", "improvement too small"
                    )
                    rotation_sinusoid_stats["method"] = "generic_periodogram"
                    rotation_sinusoid_stats["std_before"] = generic_stats.get("std_before", np.nan)
                    rotation_sinusoid_stats["std_after"] = generic_stats.get("std_after", np.nan)
                    rotation_sinusoid_stats["improvement"] = float(generic_stats.get("improvement", 0.0))

        rotation_sinusoid_applied = bool(rotation_sinusoid_stats.get("applied", False))
    else:
        rotation_residuals_corr = sp_residuals_corr.copy()
        rotation_sinusoid_model = np.zeros(len(time), dtype=float)
        rotation_sinusoid_stats = {"applied": False, "method": "disabled", "n_components": 0, "periods_hr": [], "improvement": 0.0, "boundary_hit": False, "reject_reason": "disabled"}
        rotation_sinusoid_applied = False

    if cfg.apply_local_final_sinusoid_correction and _ASTROPY_LS:
        final_residual, local_sinusoid_model, local_sinusoid_stats = local_sinusoid_correction(
            time,
            rotation_residuals_corr,
            flux_err=flux_err,
            provisional_flare_mask=flare_mask_p2,
            window_days=cfg.local_sinusoid_window_d,
            step_fraction=cfg.local_sinusoid_step_fraction,
            min_points=cfg.local_sinusoid_min_points,
            min_period_hr=cfg.min_sinusoid_period_hr,
            max_period_hr=cfg.max_sinusoid_period_hr,
            min_cycles=cfg.local_sinusoid_min_cycles,
            amp_sigma=cfg.local_sinusoid_amp_sigma,
            min_improvement=cfg.local_sinusoid_min_improvement,
            clip_sigma=cfg.local_sinusoid_clip_sigma,
        )
        local_sinusoid_applied = bool(local_sinusoid_stats["accepted"].any()) if not local_sinusoid_stats.empty else False
    else:
        final_residual = rotation_residuals_corr.copy()
        local_sinusoid_model = np.zeros(len(time), dtype=float)
        local_sinusoid_stats = pd.DataFrame()
        local_sinusoid_applied = False

    # Rebuild the final flare mask after the final periodic correction. This
    # releases points that were provisionally flagged only because they sat on
    # broad sinusoid crests, while true sharp positive outliers should remain.
    # The helper guards against final-mask explosions that would otherwise mark
    # nearly the full light curve and corrupt the downstream noise estimate.
    final_flare_mask, final_mask_diagnostics = build_safe_final_flare_mask(
        final_residual,
        sigma_thresh=cfg.second_pass_sigma,
        previous_mask=flare_mask_p2,
        max_mask_fraction=cfg.final_flare_max_mask_fraction,
    )
    final_flare_mask = final_flare_mask | external_flare_mask_arr

    if cfg.center_final_residual:
        final_residual, centering_diagnostics = center_residual_by_moving_average(
            final_residual,
            ma_window=cfg.centering_ma_window,
            clip_abs=cfg.centering_clip_abs,
        )
        # Recompute the final flare mask after centering. The threshold width is
        # unchanged, but the residual zero-point is now corrected. Keep the same
        # safety guard to avoid exploding the mask after the centering shift.
        final_flare_mask, final_mask_diagnostics = build_safe_final_flare_mask(
            final_residual,
            sigma_thresh=cfg.second_pass_sigma,
            previous_mask=flare_mask_p2,
            max_mask_fraction=cfg.final_flare_max_mask_fraction,
        )
        final_flare_mask = final_flare_mask | external_flare_mask_arr
    else:
        ma_uncentered = moving_average(final_residual, cfg.centering_ma_window)
        centering_diagnostics = {
            "centering_applied": False,
            "centering_ma_window": int(cfg.centering_ma_window),
            "centering_clip_abs": None if cfg.centering_clip_abs is None else float(cfg.centering_clip_abs),
            "residual_offset_subtracted": 0.0,
            "ma_median_before_centering": float(np.nanmedian(ma_uncentered[np.isfinite(ma_uncentered)])),
            "ma_median_after_centering": float(np.nanmedian(ma_uncentered[np.isfinite(ma_uncentered)])),
        }
        centering_diagnostics.update({f"ma_before_{k}": v for k, v in below_zero_run_stats(ma_uncentered).items()})
        centering_diagnostics.update({f"ma_after_{k}": v for k, v in below_zero_run_stats(ma_uncentered).items()})

    sp_sigma_local = compute_rolling_local_sigma(
        final_residual, window_pts=cfg.rolling_window_pts, flare_mask=final_flare_mask
    )
    # When granulation is active, also compute a short-window sigma for flare
    # detection.  If no granulation timescale is set the two are identical.
    if gran_tau is not None:
        sp_sigma_local_short = compute_rolling_local_sigma(
            final_residual, window_pts=noise_window_pts, flare_mask=final_flare_mask
        )
    else:
        sp_sigma_local_short = sp_sigma_local
    sp_sigma_total = np.sqrt(sp_sigma_local**2 + flux_err**2)
    q_sp = final_residual[np.isfinite(final_residual) & ~final_flare_mask]
    sp_sc_std = sigma_clipped_std(q_sp) if len(q_sp) > 10 else np.nan

    sort_idx = np.argsort(time)
    final_df = pd.DataFrame({
        "time": time[sort_idx],
        "flux": flux[sort_idx],
        "flux_err": flux_err[sort_idx],
        "first_pass_trend": fp_trend[sort_idx],
        "first_pass_residual": fp_residuals[sort_idx],
        "first_pass_residual_corr": fp_residuals_corr[sort_idx],
        "first_pass_sinusoid_model": fp_sinusoid_model[sort_idx],
        "second_pass_input_flux": flux_p2_input[sort_idx],
        "second_pass_trend": sp_trend[sort_idx],
        "second_pass_residual": sp_residuals[sort_idx],
        "second_pass_residual_corr": sp_residuals_corr[sort_idx],
        "second_pass_sinusoid_model": sp_sinusoid_model[sort_idx],
        "rotation_sinusoid_model": rotation_sinusoid_model[sort_idx],
        "rotation_residual_corr": rotation_residuals_corr[sort_idx],
        "local_sinusoid_model": local_sinusoid_model[sort_idx],
        "final_residual": final_residual[sort_idx],
        "local_sigma": sp_sigma_local[sort_idx],
        "granulation_noise_sigma_local": sp_sigma_local_short[sort_idx],
        "total_sigma": sp_sigma_total[sort_idx],
        "external_flare_mask": external_flare_mask_arr[sort_idx],
        "flare_mask_prelim_internal": preliminary_flare_mask_internal[sort_idx],
        "flare_mask_prelim": flare_mask_prelim[sort_idx],
        "provisional_flare_mask": flare_mask_p2[sort_idx],
        "final_flare_mask": final_flare_mask[sort_idx],
        "selected_window_size": sp_best_window[sort_idx],
        "selected_shift": sp_best_shift[sort_idx],
        "selected_segment_id": sp_best_seg[sort_idx],
        "selected_segment_score": sp_best_score[sort_idx],
    })
    add_window_fit_columns(
        final_df, sort_idx, all_results_p1, prefix="first_pass", min_fit_pts=cfg.min_fit_pts,
        granulation_timescale_days=gran_tau,
        granulation_min_window_factor=cfg.granulation_min_window_factor,
        granulation_score_penalty=cfg.granulation_score_penalty,
    )
    add_window_fit_columns(
        final_df, sort_idx, all_results_p2, prefix="second_pass", min_fit_pts=cfg.min_fit_pts,
        granulation_timescale_days=gran_tau,
        granulation_min_window_factor=cfg.granulation_min_window_factor,
        granulation_score_penalty=cfg.granulation_score_penalty,
    )

    for w in (2, 5, 10, 20, 50):
        final_df[f"ma_{w}"] = moving_average(final_df["final_residual"].to_numpy(), w)

    cadence_days = float(np.nanmedian(np.diff(time)))
    summary = {
        "span_days": float(time.max() - time.min()),
        "n_cadences": int(len(time)),
        "cadence_min": cadence_days * 24 * 60,
        "window_sizes": tuple(cfg.window_sizes),
        "poly_deg": cfg.poly_deg,
        "smooth_sigma_cadences": cfg.smooth_sigma_cadences,
        "final_flare_max_mask_fraction": float(cfg.final_flare_max_mask_fraction),
        "min_sinusoid_period_hr": float(cfg.min_sinusoid_period_hr),
        "max_sinusoid_period_hr": float(cfg.max_sinusoid_period_hr),
        "sinusoid_min_improvement": float(cfg.sinusoid_min_improvement),
        "fp_sinusoid_applied": fp_sinusoid_applied,
        "sp_sinusoid_applied": sp_sinusoid_applied,
        "fp_sinusoid_n_components": int(fp_sinusoid_stats.get("n_components", 0)),
        "sp_sinusoid_n_components": int(sp_sinusoid_stats.get("n_components", 0)),
        "fp_sinusoid_improvement": float(fp_sinusoid_stats.get("improvement", 0.0)),
        "sp_sinusoid_improvement": float(sp_sinusoid_stats.get("improvement", 0.0)),
        "fp_sinusoid_periods_hr": tuple(fp_sinusoid_stats.get("periods_hr", [])),
        "sp_sinusoid_periods_hr": tuple(sp_sinusoid_stats.get("periods_hr", [])),
        "fp_sinusoid_boundary_hit": bool(fp_sinusoid_stats.get("boundary_hit", False)),
        "sp_sinusoid_boundary_hit": bool(sp_sinusoid_stats.get("boundary_hit", False)),
        "rotation_sinusoid_applied": rotation_sinusoid_applied,
        "rotation_sinusoid_min_period_hr": float(cfg.rotation_sinusoid_min_period_hr),
        "rotation_sinusoid_max_period_hr": float(cfg.rotation_sinusoid_max_period_hr),
        "rotation_sinusoid_n_components": int(rotation_sinusoid_stats.get("n_components", 0)),
        "rotation_sinusoid_improvement": float(rotation_sinusoid_stats.get("improvement", 0.0)),
        "rotation_sinusoid_periods_hr": tuple(rotation_sinusoid_stats.get("periods_hr", [])),
        "rotation_sinusoid_boundary_hit": bool(rotation_sinusoid_stats.get("boundary_hit", False)),
        "rotation_sinusoid_reject_reason": str(rotation_sinusoid_stats.get("reject_reason", "")),
        "rotation_correction_method": str(rotation_sinusoid_stats.get("method", "")),
        "known_rotation_period_days": (None if cfg.known_rotation_period_days is None else float(cfg.known_rotation_period_days)),
        "known_rotation_period_hr": (None if cfg.known_rotation_period_days is None else float(24.0 * cfg.known_rotation_period_days)),
        "fast_rotation_harmonics": tuple(rotation_sinusoid_stats.get("harmonics", cfg.fast_rotation_harmonics)),
        "fast_rotation_blocks_accepted": int(rotation_sinusoid_stats.get("n_blocks_accepted", 0)),
        "fast_rotation_blocks_total": int(rotation_sinusoid_stats.get("n_blocks_total", 0)),
        "local_sinusoid_applied": local_sinusoid_applied,
        "n_external_flare_masked": int(external_flare_mask_arr.sum()),
        "granulation_timescale_days": gran_tau,
        "granulation_timescale_source": gran_tau_source,
        "granulation_timescale_estimated_days": gran_tau_estimated,
        "granulation_effective_window_sizes": tuple(effective_window_sizes),
        "granulation_noise_window_pts": int(noise_window_pts),
        "n_prelim_internal_masked": int(preliminary_flare_mask_internal.sum()),
        "n_prelim_masked": int(flare_mask_prelim.sum()),
        "n_provisional_masked": int(flare_mask_p2.sum()),
        "n_final_masked": int(final_flare_mask.sum()),
        "n_local_sinusoid_windows": int(local_sinusoid_stats["accepted"].sum()) if not local_sinusoid_stats.empty else 0,
        "median_local_sigma": float(np.nanmedian(sp_sigma_local)),
        "sigma_clipped_global_std": float(sp_sc_std),
        "median_total_sigma": float(np.nanmedian(sp_sigma_total)),
        **final_mask_diagnostics,
        **centering_diagnostics,
    }
    arrays = locals().copy()
    return DetrendResult(
        final_df=final_df,
        seg_stats_p1=seg_stats_p1,
        seg_stats_p2=seg_stats_p2,
        local_sinusoid_stats=local_sinusoid_stats,
        summary=summary,
        arrays=arrays,
    )

def _prepare_external_flare_mask_for_timeseries(
    ts_df: pd.DataFrame,
    external_flare_mask: np.ndarray | pd.Series | None,
) -> np.ndarray | None:
    """Align an external flare mask with ``prepare_arrays(ts_df)`` output.

    The user-facing mask is supplied against the input DataFrame rows. This helper
    applies the same finite-time/finite-flux filtering and time sorting used by
    ``prepare_arrays`` so the mask stays aligned with the detrending arrays.
    """
    if external_flare_mask is None:
        return None
    mask = np.asarray(external_flare_mask, dtype=bool)
    if len(mask) != len(ts_df):
        raise ValueError("external_flare_mask must have the same length as ts_df.")
    req = {"time", "flux"}
    missing = req.difference(ts_df.columns)
    if missing:
        raise ValueError(f"Missing columns needed to align external_flare_mask: {sorted(missing)}")
    work = ts_df.loc[:, ["time", "flux"]].copy()
    work["_external_flare_mask"] = mask
    work = work[np.isfinite(work["time"]) & np.isfinite(work["flux"])].sort_values("time")
    return work["_external_flare_mask"].to_numpy(dtype=bool)

def detrend_dataframe(
    ts_df: pd.DataFrame,
    config: DetrendConfig | None = None,
    clean_gaps: bool = True,
    tic_id: int | str | None = None,
    base_path: str | Path | None = None,
    stellar_rotation_df: pd.DataFrame | None = None,
    known_rotation_period_days: float | None = None,
    external_flare_mask: np.ndarray | pd.Series | None = None,
) -> tuple[DetrendResult, dict | None]:
    """Prepare a light-curve table and run detrending.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Table containing ``time``, ``flux``, and ``flux_err`` columns.
    config : DetrendConfig or None, optional
        Pipeline configuration. If None, the default ``DetrendConfig`` is used.
    clean_gaps : bool, optional
        If True, remove cadences around detected large gaps before detrending.
    external_flare_mask : array-like or None, optional
        Boolean mask aligned with ``ts_df`` rows. True points are excluded from
        detrending fits/noise estimates in addition to the internally computed
        flare masks. Omit this argument to keep the original behavior.

    Returns
    -------
    result : DetrendResult
        Full detrending result.
    gap_info : dict or None
        Gap-cleaning diagnostics if cleaning was attempted successfully; otherwise
        None.

    Raises
    ------
    ValueError
        Propagated from input preparation or detrending when required inputs are
        invalid.
    """
    # Keep the simple notebook API working: when tic_id/base_path are not
    # supplied explicitly, try to infer them from common caller variables such
    # as TIC_ID, BASE_PATH, and OUTPUT_CSV. This lets the fast-rotator harmonic
    # correction automatically use a saved <TIC>_df_stellar_rotation.csv without
    # requiring extra arguments in the user-facing detrend_dataframe call.
    inferred_context = _infer_detrending_context_from_caller()
    if tic_id is None:
        tic_id = inferred_context.get("tic_id")
    if base_path is None:
        base_path = inferred_context.get("base_path")

    cfg = attach_rotation_metadata_to_config(
        config,
        base_path=base_path,
        tic_id=tic_id,
        stellar_rotation_df=stellar_rotation_df,
        known_rotation_period_days=known_rotation_period_days,
    )
    time, flux, flux_err = prepare_arrays(ts_df)
    external_flare_mask_arr = _prepare_external_flare_mask_for_timeseries(ts_df, external_flare_mask)
    gap_info = None
    if clean_gaps:
        try:
            time, flux, flux_err, gap_info = remove_gap_edges(
                time, flux, flux_err, n_remove=cfg.n_remove,
                gap_sigma=cfg.gap_sigma, gap_min_factor=cfg.gap_min_factor,
            )
            if external_flare_mask_arr is not None:
                external_flare_mask_arr = external_flare_mask_arr[np.asarray(gap_info["keep_mask"], dtype=bool)]
        except ValueError as exc:
            warnings.warn(
                f"Gap cleaning skipped: {exc} Continuing with the uncleaned series.",
                RuntimeWarning,
            )
    return run_detrending(time, flux, flux_err, cfg, external_flare_mask=external_flare_mask_arr), gap_info

def save_result(result: DetrendResult, path: str | Path) -> Path:
    """Save the final detrended table as a CSV file.

    Parameters
    ----------
    result : DetrendResult
        Detrending result to save.
    path : str or pathlib.Path
        Output CSV path.

    Returns
    -------
    pathlib.Path
        Output path.

    Raises
    ------
    TypeError
        If ``result`` is not a ``DetrendResult``.
    """
    if not isinstance(result, DetrendResult):
        raise TypeError("result must be a DetrendResult.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.final_df.to_csv(path, index=False)
    return path

def print_summary(result: DetrendResult, tic_id: int | None = None) -> None:
    """Print a short summary of a detrending result.

    Parameters
    ----------
    result : DetrendResult
        Detrending result to summarize.
    tic_id : int or None, optional
        TIC identifier to include in the title.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``result`` is not a ``DetrendResult``.
    """
    if not isinstance(result, DetrendResult):
        raise TypeError("result must be a DetrendResult.")
    s = result.summary
    title = f"TIC {tic_id}" if tic_id is not None else "Detrending summary"
    print("=" * 65)
    print(f"  {title}  |  Span: {s['span_days']:.2f} d  |  Cadences: {s['n_cadences']:,}")
    print("=" * 65)
    print(f"  Cadence                     : {s['cadence_min']:.2f} min")
    print(f"  Window sizes evaluated      : {list(s['window_sizes'])} d")
    print(f"  Polynomial degree           : {s['poly_deg']}")
    print(f"  Trend smoothing             : {s['smooth_sigma_cadences']} cadences")
    # Granulation summary
    gran_tau_print = s.get('granulation_timescale_days')
    if gran_tau_print is not None:
        gran_src = s.get('granulation_timescale_source', '')
        gran_est = s.get('granulation_timescale_estimated_days')
        gran_wins = list(s.get('granulation_effective_window_sizes', []))
        gran_nw = s.get('granulation_noise_window_pts', '')
        print(f"  Granulation τ               : {gran_tau_print:.3f} d ({gran_src})")
        if gran_est is not None:
            print(f"  Granulation τ (ACF est.)    : {gran_est:.3f} d")
        print(f"  Granulation-safe windows    : {gran_wins} d")
        print(f"  Granulation noise window    : {gran_nw} cadences")
    else:
        print("  Granulation handling        : disabled (no timescale set)")
    print(f"  Short sinusoid period range : {s.get('min_sinusoid_period_hr', np.nan):.3f}–{s.get('max_sinusoid_period_hr', np.nan):.3f} hr")
    print(f"  Short sinusoid correction FP: {'applied' if s['fp_sinusoid_applied'] else 'not applied'} "
          f"({s.get('fp_sinusoid_n_components', 0)} comp, {100 * s.get('fp_sinusoid_improvement', 0.0):.2f}% improvement)")
    print(f"  Short sinusoid correction SP: {'applied' if s['sp_sinusoid_applied'] else 'not applied'} "
          f"({s.get('sp_sinusoid_n_components', 0)} comp, {100 * s.get('sp_sinusoid_improvement', 0.0):.2f}% improvement)")
    if s.get('fp_sinusoid_boundary_hit') or s.get('sp_sinusoid_boundary_hit'):
        print("  Short sinusoid warning      : selected period near search boundary")
    print(f"  Rotation sinusoid range     : {s.get('rotation_sinusoid_min_period_hr', np.nan):.2f}–{s.get('rotation_sinusoid_max_period_hr', np.nan):.2f} hr")
    method = s.get('rotation_correction_method', '')
    method_txt = f", {method}" if method else ""
    print(f"  Rotation sinusoid correction: {'applied' if s.get('rotation_sinusoid_applied') else 'not applied'} "
          f"({s.get('rotation_sinusoid_n_components', 0)} comp, {100 * s.get('rotation_sinusoid_improvement', 0.0):.2f}% improvement{method_txt})")
    if s.get('known_rotation_period_hr') is not None:
        print(f"  Known stellar rotation P    : {s.get('known_rotation_period_hr'):.3f} hr")
    if s.get('fast_rotation_blocks_total', 0):
        print(f"  Fast-rotation blocks used   : {s.get('fast_rotation_blocks_accepted', 0)} / {s.get('fast_rotation_blocks_total', 0)}")
    if s.get('rotation_sinusoid_applied') and s.get('rotation_sinusoid_periods_hr'):
        rot_periods = [f"{p:.3f} hr" for p in s.get('rotation_sinusoid_periods_hr', [])]
        print(f"  Rotation periods            : {rot_periods}")
    elif s.get('rotation_sinusoid_reject_reason'):
        print(f"  Rotation reject reason      : {s.get('rotation_sinusoid_reject_reason')}")
    if s.get('rotation_sinusoid_boundary_hit'):
        print("  Rotation sinusoid warning   : selected period near search boundary")
    print(f"  Local final sinusoid stage   : {'applied' if s['local_sinusoid_applied'] else 'not applied'}")
    print(f"  Accepted local windows       : {s['n_local_sinusoid_windows']:,}")
    print(f"  Preliminary flare mask       : {s['n_prelim_masked']:,}")
    print(f"  Provisional flare mask       : {s['n_provisional_masked']:,}")
    print(f"  Recomputed final flare mask  : {s['n_final_masked']:,}")
    if 'final_mask_recomputed' in s:
        status = 'accepted' if s.get('final_mask_recomputed') else 'fallback used'
        print(f"  Final flare mask guard       : {status}")
        if s.get('final_mask_reject_reason'):
            print(f"    Reason                    : {s['final_mask_reject_reason']}")
        if np.isfinite(s.get('final_mask_candidate_fraction', np.nan)):
            print(f"    Candidate fraction        : {100 * s['final_mask_candidate_fraction']:.3f}%")
    print("  Noise floor (second-pass residuals, quiescent):")
    print(f"    Rolling MAD sigma (median): {s['median_local_sigma']:.6f}")
    print(f"    Sigma-clipped global std  : {s['sigma_clipped_global_std']:.6f}")
    print(f"    Total sigma (median)      : {s['median_total_sigma']:.6f}")
    print("  Residual centering:")
    print(f"    Applied                   : {'yes' if s.get('centering_applied') else 'no'}")
    print(f"    MA window                 : {s.get('centering_ma_window', np.nan)} cadences")
    print(f"    Offset subtracted         : {s.get('residual_offset_subtracted', np.nan):.6f}")
    print(f"    MA median before / after  : {s.get('ma_median_before_centering', np.nan):.6f} / {s.get('ma_median_after_centering', np.nan):.6f}")
    print(f"    Fraction below zero       : {s.get('ma_before_fraction_below_zero', np.nan):.3f} / {s.get('ma_after_fraction_below_zero', np.nan):.3f}")
    print(f"    Longest run below zero    : {s.get('ma_before_longest_run_below_zero', 0)} / {s.get('ma_after_longest_run_below_zero', 0)} cadences")
    print("=" * 65)

def _x_range(df: pd.DataFrame, x_range):
    """Return the requested x-range or the full time range.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing a ``time`` column.
    x_range : tuple or None
        Requested ``(min, max)`` range. If None, the table time span is used.

    Returns
    -------
    tuple
        Two-element x-axis range.
    """
    return (float(df["time"].min()), float(df["time"].max())) if x_range is None else x_range


def _visible_mask(df: pd.DataFrame, x_range) -> np.ndarray:
    """Select rows inside the current x-axis range.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing a ``time`` column.
    x_range : tuple or None
        Requested ``(min, max)`` range. If None, the table time span is used.

    Returns
    -------
    numpy.ndarray
        Boolean mask of rows inside the range and with finite time values.
    """
    t = df["time"].to_numpy(dtype=float)
    lo, hi = _x_range(df, x_range)
    return np.isfinite(t) & (t >= lo) & (t <= hi)


def _robust_limits(values, lower_pct: float = 1.0, upper_pct: float = 99.0, pad_frac: float = 0.08):
    """Return robust y-limits with padding.

    Parameters
    ----------
    values : array-like
        Values used to estimate the limits.
    lower_pct : float, optional
        Lower percentile.
    upper_pct : float, optional
        Upper percentile.
    pad_frac : float, optional
        Fractional padding added to the percentile span.

    Returns
    -------
    tuple or None
        ``(lo, hi)`` limits, or None if no finite limits can be estimated.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    lo, hi = np.nanpercentile(vals, [lower_pct, upper_pct])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0)
        return float(lo - pad), float(hi + pad)
    pad = (hi - lo) * pad_frac
    return float(lo - pad), float(hi + pad)


def _set_robust_ylim(
    ax,
    values,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    pad_frac: float = 0.08,
    include_zero: bool = False,
):
    """Apply robust y-limits to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to update.
    values : array-like
        Values used to estimate the limits.
    lower_pct : float, optional
        Lower percentile.
    upper_pct : float, optional
        Upper percentile.
    pad_frac : float, optional
        Fractional padding added to the percentile span.
    include_zero : bool, optional
        If True, expand limits to include zero.

    Returns
    -------
    None
    """
    lim = _robust_limits(values, lower_pct=lower_pct, upper_pct=upper_pct, pad_frac=pad_frac)
    if lim is None:
        return
    lo, hi = lim
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    ax.set_ylim(lo, hi)

def plot_trends(
    result: DetrendResult,
    x_range=None,
    tic_id: int | None = None,
    robust_ylim: bool = True,
    y_percentiles: tuple[float, float] = (1.0, 99.0),
):
    """Plot the light curve and fitted trends.

    Parameters
    ----------
    result : DetrendResult
        Detrending result to plot.
    x_range : tuple or None, optional
        Time range shown on the x-axis. If None, the full range is shown.
    tic_id : int or None, optional
        TIC identifier to include in the title.
    robust_ylim : bool, optional
        If True, use percentile-based y-axis limits.
    y_percentiles : tuple of float, optional
        Lower and upper percentiles used for robust y-axis limits.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        Created axis.
    """
    df = result.final_df
    xr = _x_range(df, x_range)
    fig, ax = plt.subplots(figsize=(16, 5))
    title = f"TIC {tic_id} — Original light curve with detrending fits" if tic_id is not None else "Original light curve with detrending fits"
    ax.set_title(title)
    mk = df["final_flare_mask"].to_numpy(bool)
    ax.scatter(df.loc[~mk, "time"], df.loc[~mk, "flux"], s=2, alpha=0.35, color=C["data"], label="Quiescent flux")
    ax.scatter(df.loc[mk, "time"], df.loc[mk, "flux"], s=10, alpha=0.75, color=C["flare"], label="Flare-masked")
    ax.plot(df["time"], df["first_pass_trend"], color=C["trend1"], lw=1.5, label="First-pass trend")
    ax.plot(df["time"], df["second_pass_trend"], color=C["trend2"], lw=1.5, label="Second-pass trend")
    ax.set_xlim(*xr)
    if robust_ylim:
        vis = _visible_mask(df, x_range)
        y_parts = [
            df.loc[vis & ~mk, "flux"].to_numpy(dtype=float),
            df.loc[vis, "first_pass_trend"].to_numpy(dtype=float),
            df.loc[vis, "second_pass_trend"].to_numpy(dtype=float),
        ]
        _set_robust_ylim(
            ax,
            np.concatenate(y_parts),
            lower_pct=y_percentiles[0],
            upper_pct=y_percentiles[1],
        )
    ax.set_xlabel("Time [BTJD days]")
    ax.set_ylabel("Flux [e⁻ s⁻¹]")
    ax.legend()
    fix_time_axis(ax)
    fig.tight_layout()
    return fig, ax

def plot_residuals(
    result: DetrendResult,
    x_range=None,
    tic_id: int | None = None,
    robust_ylim: bool = True,
    y_percentiles: tuple[float, float] = (1.0, 99.0),
    sigma_ylim: float = 5.0,
):
    """Plot final residuals and local noise bands.

    Parameters
    ----------
    result : DetrendResult
        Detrending result to plot.
    x_range : tuple or None, optional
        Time range shown on the x-axis. If None, the full range is shown.
    tic_id : int or None, optional
        TIC identifier to include in the title.
    robust_ylim : bool, optional
        If True, use percentile-based y-axis limits expanded to include local sigma.
    y_percentiles : tuple of float, optional
        Lower and upper percentiles used for robust y-axis limits.
    sigma_ylim : float, optional
        Minimum y-axis half-range in units of the visible median local sigma.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        Created axis.
    """
    df = result.final_df
    xr = _x_range(df, x_range)
    fig, ax = plt.subplots(figsize=(16, 5))
    title = f"TIC {tic_id} — Final residuals" if tic_id is not None else "Final residuals"
    ax.set_title(title)
    t = df["time"].to_numpy()
    rs = df["final_residual"].to_numpy()
    sl = df["local_sigma"].to_numpy()
    mk = df["final_flare_mask"].to_numpy(bool)
    ax.fill_between(t, -3*sl, 3*sl, color=C["sigma3"], alpha=0.12, label="±3σ")
    ax.fill_between(t, -2*sl, 2*sl, color=C["sigma2"], alpha=0.17, label="±2σ")
    ax.fill_between(t, -sl, sl, color=C["sigma1"], alpha=0.22, label="±1σ")
    ax.scatter(t[~mk], rs[~mk], s=2, alpha=0.4, color=C["data"], label="Quiescent residuals")
    ax.scatter(t[mk], rs[mk], s=8, alpha=0.7, color=C["flare"], label="Flare-masked")
    ax.plot(t, df["ma_10"], color=C["ma"], lw=1.4, label="10-cadence moving avg")
    ax.axhline(0, color=C["zero"], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(*xr)
    if robust_ylim:
        vis = _visible_mask(df, x_range)
        quiet = vis & ~mk & np.isfinite(rs)
        values = [rs[quiet], df.loc[vis, "ma_10"].to_numpy(dtype=float)]
        lim = _robust_limits(
            np.concatenate(values),
            lower_pct=y_percentiles[0],
            upper_pct=y_percentiles[1],
        )
        finite_sigma = sl[vis & np.isfinite(sl)]
        sigma_med = float(np.nanmedian(finite_sigma)) if len(finite_sigma) else np.nan
        if lim is not None:
            lo, hi = lim
            if np.isfinite(sigma_med) and sigma_med > 0:
                lo = min(lo, -sigma_ylim * sigma_med)
                hi = max(hi, sigma_ylim * sigma_med)
            ax.set_ylim(min(lo, 0.0), max(hi, 0.0))
    ax.set_xlabel("Time [BTJD days]")
    ax.set_ylabel("Residual flux [e⁻ s⁻¹]")
    ax.legend()
    fix_time_axis(ax)
    fig.tight_layout()
    return fig, ax

def plot_selected_windows(result: DetrendResult, x_range=None, tic_id: int | None = None):
    """Plot selected window size and shift over time.

    Parameters
    ----------
    result : DetrendResult
        Detrending result to plot.
    x_range : tuple or None, optional
        Time range shown on the x-axis. If None, the full range is shown.
    tic_id : int or None, optional
        TIC identifier to include in the title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    axes : numpy.ndarray
        Created axes.
    """
    df = result.final_df
    xr = _x_range(df, x_range)
    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    title = f"TIC {tic_id} — Selected window size and shift" if tic_id is not None else "Selected window size and shift"
    fig.suptitle(title)
    axes[0].scatter(df["time"], df["selected_window_size"], s=1, alpha=0.4, color=C["data"])
    axes[0].set_ylabel("Window size [days]")
    axes[1].scatter(df["time"], df["selected_shift"], s=1, alpha=0.4, color=C["trend2"])
    axes[1].set_ylabel("Shift [days]"); axes[1].set_xlabel("Time [BTJD days]")
    for ax in axes:
        ax.set_xlim(*xr); fix_time_axis(ax)
    fig.tight_layout()
    return fig, axes

def plot_segment_quality(result: DetrendResult):
    """Plot second-pass segment-quality summaries.

    Parameters
    ----------
    result : DetrendResult
        Detrending result to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    axes : numpy.ndarray
        Created axes.
    """
    df = result.seg_stats_p2
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Segment quality: normal vs shifted windows (second pass)")
    metrics_cfg = [
        ("r2_adj", "Adjusted R²", None),
        ("red_chi2", "Reduced χ²", 1.0),
        ("rsd", "Residual σ", None),
    ]
    colors = {"normal": "#60A5FA", "shifted": "crimson"}
    for ax, (col, ylabel, ref) in zip(axes, metrics_cfg):
        for label, grp in df.groupby("window_label"):
            summary = grp.groupby("window_size")[col].median()
            x = np.arange(len(summary))
            ax.bar(x + (0.2 if label == "shifted" else -0.2), summary.values, width=0.35,
                   color=colors[label], alpha=0.85, label=label, edgecolor="white")
        if ref is not None:
            ax.axhline(ref, color="#EF4444", lw=1.2, ls="--", label=f"ideal={ref}")
        vals = sorted(df["window_size"].unique())
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels([f"{w:.2f}d" for w in vals])
        ax.set_xlabel("Window size"); ax.set_ylabel(ylabel); ax.set_title(ylabel); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes
