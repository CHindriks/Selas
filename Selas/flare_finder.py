"""Reusable two-pass flare finder with optional plotting and CSV output.

This module is a refactor of the original exploratory notebook into functions that
can be imported from a smaller, easier notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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
THESIS_SPAN_ALPHA = 0.20


def _apply_thesis_plot_style() -> None:
    """Apply thesis-style Matplotlib defaults used by the reference figures."""

    font_family = "DejaVu Serif"
    font_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf"),
        Path("/usr/share/fonts/dejavu-serif-fonts/DejaVuSerifCondensed.ttf"),
    )
    for font_path in font_candidates:
        if font_path.exists():
            try:
                font_family = fm.FontProperties(fname=str(font_path)).get_name()
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


_apply_thesis_plot_style()


def _safe_filename_token(value: object) -> str:
    """Return a filesystem-safe token for plot filenames."""

    token = "plot" if value is None else str(value)
    keep = []
    for char in token:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "plot"


def _default_plot_prefix(prefix: Optional[str] = None, tic_id: Optional[int | str] = None) -> str:
    """Build a stable plot filename prefix."""

    if prefix is not None:
        return _safe_filename_token(prefix)
    if tic_id is not None:
        return _safe_filename_token(f"TIC_{tic_id}")
    return "flare_finder"


def _resolve_plot_dir(
    plot_dir: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
) -> Path:
    """Resolve the directory used for thesis-style PDF figures."""

    if plot_dir is not None:
        return Path(plot_dir)
    if out_dir is not None:
        return Path(out_dir) / "Figures"
    return Path("flare_finder_figures")


def _save_thesis_figure(fig, path: str | Path) -> Path:
    """Save a figure as an individual thesis-style PDF and close it."""

    output_path = Path(path).with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _apply_scalar_formatter(ax, format_x: bool = True, format_y: bool = True) -> None:
    """Disable scientific and offset notation on linear numerical axes."""

    if format_x and ax.get_xscale() == "linear":
        formatter_x = ScalarFormatter(useOffset=False)
        formatter_x.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter_x)

    if format_y and ax.get_yscale() == "linear":
        formatter_y = ScalarFormatter(useOffset=False)
        formatter_y.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter_y)


def _style_axis(
    ax,
    panel_label: Optional[str] = None,
    format_x: bool = True,
    format_y: bool = True,
) -> None:
    """Apply common thesis-style ticks, spines, formatting, and panel labels."""

    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH)

    ax.tick_params(axis="both", which="major", width=LINEWIDTH, length=7 * SCALE, direction="out")
    ax.tick_params(axis="both", which="minor", width=LINEWIDTH * 0.8, length=4 * SCALE, direction="out")
    ax.minorticks_on()
    _apply_scalar_formatter(ax, format_x=format_x, format_y=format_y)

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


def _finish_thesis_figure(
    fig,
    ax,
    path: Optional[str | Path] = None,
    show: bool = False,
    legend_loc: Optional[str] = None,
) -> Optional[Path]:
    """Finalize spacing and either save the figure as PDF or show it."""

    if legend_loc is not None:
        _legend_if_needed(ax, loc=legend_loc)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.96)
    if path is not None:
        return _save_thesis_figure(fig, path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return None


@dataclass
class FlareFinderConfig:
    """Settings for the two-pass flare finder.

    The defaults mirror the exploratory notebook as closely as possible. Pass an
    instance of this class to :func:`run_two_pass_flare_finder` instead of editing
    the detection routines directly.

    Attributes
    ----------
    lower_sigma : float
        Sigma multiplier for the point-wise residual threshold.
    moving_average_sigma : float
        Sigma multiplier for the moving-average threshold.
    n_consecutive_points : int
        Minimum number of high points required for an initial candidate.
    max_below_threshold : int
        Maximum cadence gap allowed while grouping high points.
    min_gap_distance_points : int
        Minimum distance from large observing gaps.
    temporary_combined_bump_gap_points : int
        Maximum gap used to build temporary combined flare windows.
    strong_flare_peak_sigma : float
        Peak threshold used when masking strong pass-1 flares for noise estimation.
    strong_flare_mask_padding_points : int
        Padding added around strong flares in the second-pass noise mask.
    residual_col_for_noise : str
        Residual column used to recompute local noise.
    flux_err_col : str
        Flux-error column optionally combined with the cleaned local scatter.
    local_noise_window_days : float or None
        Rolling-window size in days. Used only when ``local_noise_window_points`` is
        not set.
    local_noise_window_points : int or None
        Rolling-window size in cadences. Takes precedence over days.
    include_flux_err_in_detection_sigma : bool
        If True, use quadrature-combined flux error and local scatter for detection.
    unique_flare_merge_gap_points : int
        Maximum gap used to merge pass-1 and pass-2 detections.
    apply_ma10_residual_filter : bool
        Apply the final moving-average/residual post-filter.
    ma10_filter_sigma : float
        Moving-average sigma threshold used by the post-filter.
    residual_filter_sigma : float
        Residual sigma threshold used by the post-filter.
    split_multi_peak_flares : bool
        Split complex flare windows into multiple sub-flares when possible.
    multi_peak_detection_column : str
        Column used to detect multi-peak structure.
    multi_peak_datapoint_column : str
        Column used to assign the final datapoint peak.
    multi_peak_min_peak_sigma : float
        Minimum sigma threshold for a sub-peak.
    multi_peak_min_separation_points : int
        Minimum separation between sub-peaks in cadences.
    multi_peak_valley_fraction : float
        Valley-depth threshold relative to neighboring peaks.
    multi_peak_valley_sigma : float
        Valley-depth threshold in sigma units.
    multi_peak_min_segment_points : int
        Minimum number of cadences allowed in a split segment.
    multi_peak_correspondence_radius_points : int
        Search radius used to map a moving-average peak to a datapoint peak.
    multi_peak_plot_padding_days : float
        Plot padding around multi-peak candidates in days.
    multi_peak_max_plots : int or None
        Maximum number of diagnostic plots to draw.
    """

    lower_sigma: float = 1.8
    moving_average_sigma: float = 1.5
    n_consecutive_points: int = 3
    max_below_threshold: int = 4
    min_gap_distance_points: int = 50
    temporary_combined_bump_gap_points: int = 5

    strong_flare_peak_sigma: float = 5.0
    strong_flare_mask_padding_points: int = 30

    residual_col_for_noise: str = "final_residual"
    flux_err_col: str = "flux_err"
    local_noise_window_days: Optional[float] = None
    local_noise_window_points: Optional[int] = None
    include_flux_err_in_detection_sigma: bool = False

    unique_flare_merge_gap_points: int = 5

    apply_ma10_residual_filter: bool = True
    ma10_filter_sigma: float = 2.0
    residual_filter_sigma: float = 4.0

    split_multi_peak_flares: bool = True
    multi_peak_detection_column: str = "ma_10"
    multi_peak_datapoint_column: str = "final_residual"
    multi_peak_min_peak_sigma: float = 2.0
    multi_peak_min_separation_points: int = 5
    multi_peak_valley_fraction: float = 0.50
    multi_peak_valley_sigma: float = 1.5
    multi_peak_min_segment_points: int = 2
    multi_peak_correspondence_radius_points: int = 5
    multi_peak_plot_padding_days: float = 0.25
    multi_peak_max_plots: Optional[int] = None


def load_time_series(csv_path: str | Path) -> pd.DataFrame:
    """Load a detrended time-series table from CSV.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file containing the light-curve table.

    Returns
    -------
    pandas.DataFrame
        Loaded time-series table.

    Raises
    ------
    FileNotFoundError
        Raised by :func:`pandas.read_csv` when ``csv_path`` does not exist.
    pandas.errors.ParserError
        Raised by :func:`pandas.read_csv` when the file cannot be parsed as CSV.
    """

    return pd.read_csv(Path(csv_path))


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Check that a DataFrame contains all required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to validate.
    columns : list of str
        Column names that must be present in ``df``.
    context : str
        Short description of the calling step, used in error messages.

    Raises
    ------
    KeyError
        If one or more required columns are missing.
    """

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}. Available columns: {list(df.columns)}")


def _as_positions(index_like: Any, n: int) -> np.ndarray:
    """Convert integer-like values to valid row positions.

    Parameters
    ----------
    index_like : array-like
        Values to convert to integer row positions.
    n : int
        Number of rows in the target table.

    Returns
    -------
    numpy.ndarray
        Integer positions clipped to the half-open interval ``[0, n)``.
    """

    arr = np.asarray(index_like, dtype=int)
    return arr[(arr >= 0) & (arr < n)]


def _global_sigma(ts_df: pd.DataFrame, sigma_col: str) -> float:
    """Return the representative global sigma for a time-series table.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    sigma_col : str
        Fallback sigma column used when ``total_sigma`` is absent.

    Returns
    -------
    float
        Median ``total_sigma`` when available, otherwise median ``sigma_col``.

    Raises
    ------
    KeyError
        If neither ``total_sigma`` nor ``sigma_col`` is available.
    """

    if "total_sigma" in ts_df.columns:
        return float(np.nanmedian(ts_df["total_sigma"]))
    return float(np.nanmedian(ts_df[sigma_col]))


def find_candidate_flares(
    ts_df: pd.DataFrame,
    sigma_col: str = "local_sigma",
    lower_sigma: float = 2.0,
    moving_average_sigma: float = 1.8,
    n_consecutive_points: int = 3,
    max_below_threshold: int = 4,
    min_gap_distance_points: int = 50,
    **kwargs: Any,
) -> pd.DataFrame:
    """Find initial flare candidates from residual thresholds.

    A cadence is marked high when both ``final_residual`` and ``ma_10`` are above
    sigma-scaled thresholds. Neighboring high cadences are grouped into segments,
    short segments are removed, and candidates too close to large observing gaps are
    discarded. The misspelled keyword ``n_consecucative_points`` is still accepted
    for backward compatibility with the original notebook.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table with residual, moving-average, sigma, and time columns.
    sigma_col : str, optional
        Column used as the sigma threshold.
    lower_sigma : float, optional
        Threshold multiplier for ``final_residual``.
    moving_average_sigma : float, optional
        Threshold multiplier for ``ma_10``.
    n_consecutive_points : int, optional
        Minimum number of grouped high points required for a candidate.
    max_below_threshold : int, optional
        Maximum allowed gap, in cadences, while grouping high points.
    min_gap_distance_points : int, optional
        Minimum distance from large observing gaps, in cadences.
    **kwargs : dict
        Backward-compatible keyword arguments from the notebook.

    Returns
    -------
    pandas.DataFrame
        Initial flare candidates. Empty when no candidates pass the cuts.

    Raises
    ------
    KeyError
        If required input columns are missing.
    """

    if "n_consecucative_points" in kwargs:
        n_consecutive_points = int(kwargs["n_consecucative_points"])

    _require_columns(ts_df, ["final_residual", "ma_10", sigma_col, "time"], "candidate flare detection")

    is_high = (
        (ts_df["final_residual"] > lower_sigma * ts_df[sigma_col])
        & (ts_df["ma_10"] > moving_average_sigma * ts_df[sigma_col])
    )

    high_pos = np.flatnonzero(is_high.to_numpy())
    if len(high_pos) == 0:
        return pd.DataFrame()

    segment_id = pd.Series(high_pos).diff().fillna(1).gt(max_below_threshold).cumsum()
    valid_segments = (
        pd.DataFrame({"pos": high_pos, "segment_id": segment_id})
        .groupby("segment_id")
        .filter(lambda g: len(g) >= n_consecutive_points)
    )

    if valid_segments.empty:
        return pd.DataFrame()

    flares_df = (
        valid_segments.groupby("segment_id")["pos"]
        .agg(start_pos="min", end_pos="max", n_points_above_threshold="count")
        .reset_index(drop=True)
    )

    dt = ts_df["time"].diff()
    typical_dt = dt.median()
    if pd.notna(typical_dt) and typical_dt > 0:
        missing_points = np.round(dt / typical_dt).astype("Int64") - 1
        gap_after_pos = np.flatnonzero((missing_points >= 10).fillna(False).to_numpy())
    else:
        gap_after_pos = np.array([], dtype=int)

    def far_from_gaps(start_pos: int, end_pos: int, gap_positions: np.ndarray, min_distance: int) -> bool:
        for g in gap_positions:
            left_gap_edge = g - 1
            right_gap_edge = g
            if start_pos < right_gap_edge + min_distance and end_pos > left_gap_edge - min_distance:
                return False
        return True

    flares_df = flares_df[
        flares_df.apply(
            lambda row: far_from_gaps(
                int(row["start_pos"]),
                int(row["end_pos"]),
                gap_after_pos,
                min_distance=min_gap_distance_points,
            ),
            axis=1,
        )
    ].copy()

    if flares_df.empty:
        return flares_df

    flares_df["start_idx"] = ts_df.index[flares_df["start_pos"]]
    flares_df["end_idx"] = ts_df.index[flares_df["end_pos"]]
    flares_df["start_time"] = ts_df["time"].iloc[flares_df["start_pos"]].to_numpy()
    flares_df["end_time"] = ts_df["time"].iloc[flares_df["end_pos"]].to_numpy()

    flares_df["peak_pos"] = flares_df.apply(
        lambda row: ts_df["final_residual"].iloc[int(row["start_pos"]): int(row["end_pos"]) + 1].idxmax(),
        axis=1,
    )
    flares_df["peak_pos"] = ts_df.index.get_indexer(flares_df["peak_pos"])
    flares_df["peak_idx"] = ts_df.index[flares_df["peak_pos"]]
    flares_df["t_peak"] = ts_df["time"].iloc[flares_df["peak_pos"]].to_numpy()
    flares_df["peak_final_residual"] = ts_df["final_residual"].iloc[flares_df["peak_pos"]].to_numpy()

    return flares_df.reset_index(drop=True)


def expand_flare_windows(ts_df: pd.DataFrame, flares_df: pd.DataFrame) -> pd.DataFrame:
    """Expand candidate windows while the moving average is positive.

    Expansion proceeds left and right at the same time. Ownership tracking prevents
    neighboring candidates from claiming the same cadence.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table containing ``time`` and ``ma_10``.
    flares_df : pandas.DataFrame
        Candidate flare table with ``start_pos`` and ``end_pos`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``flares_df`` with expanded window columns added.

    Raises
    ------
    KeyError
        If required time-series columns are missing.
    """

    _require_columns(ts_df, ["time", "ma_10"], "flare window expansion")
    flares_df = flares_df.sort_values("start_pos").reset_index(drop=True).copy()
    if flares_df.empty:
        return flares_df

    ma10 = ts_df["ma_10"].to_numpy()
    n = len(ts_df)
    left_edges = flares_df["start_pos"].astype(int).to_numpy().copy()
    right_edges = flares_df["end_pos"].astype(int).to_numpy().copy()

    owner = np.full(n, -1, dtype=int)
    assert len(left_edges) == len(right_edges), "Window edge arrays must match."
    for flare_idx, (left, right) in enumerate(zip(left_edges, right_edges)):
        owner[left:right + 1] = flare_idx

    changed = True
    while changed:
        changed = False
        left_proposals: dict[int, int] = {}
        right_proposals: dict[int, int] = {}

        for flare_idx in range(len(flares_df)):
            left_candidate = left_edges[flare_idx] - 1
            right_candidate = right_edges[flare_idx] + 1

            if left_candidate >= 0 and ma10[left_candidate] > 0 and owner[left_candidate] == -1:
                left_proposals[flare_idx] = left_candidate
            if right_candidate < n and ma10[right_candidate] > 0 and owner[right_candidate] == -1:
                right_proposals[flare_idx] = right_candidate

        all_proposals = list(left_proposals.values()) + list(right_proposals.values())
        if not all_proposals:
            break
        proposal_counts = pd.Series(all_proposals).value_counts()

        for flare_idx, candidate in left_proposals.items():
            if proposal_counts[candidate] == 1:
                left_edges[flare_idx] = candidate
                owner[candidate] = flare_idx
                changed = True
        for flare_idx, candidate in right_proposals.items():
            if proposal_counts[candidate] == 1:
                right_edges[flare_idx] = candidate
                owner[candidate] = flare_idx
                changed = True

    flares_df["new_start_pos"] = left_edges
    flares_df["new_end_pos"] = right_edges
    flares_df["new_start_time"] = ts_df["time"].iloc[flares_df["new_start_pos"]].to_numpy()
    flares_df["new_end_time"] = ts_df["time"].iloc[flares_df["new_end_pos"]].to_numpy()
    flares_df["duration"] = flares_df["new_end_time"] - flares_df["new_start_time"]
    return flares_df


def add_temporary_combined_flares(
    ts_df: pd.DataFrame,
    flares_df: pd.DataFrame,
    bump_gap_points: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add temporary combined rows for touching expanded windows.

    Temporary combined rows let later filters test whether a chain of nearby
    candidates behaves better as one flare. These rows are removed later unless the
    combined flare survives the filters.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table used to compute combined peaks and times.
    flares_df : pandas.DataFrame
        Expanded candidate table.
    bump_gap_points : int, optional
        Maximum gap, in cadences, for candidates to belong to the same chain.

    Returns
    -------
    tuple of pandas.DataFrame
        Updated flare table and a table describing temporary combined windows.
    """

    flares_df = flares_df.sort_values("new_start_pos").reset_index(drop=True).copy()
    if flares_df.empty:
        flares_df["is_combined_flare"] = []
        return flares_df, pd.DataFrame()

    flares_df["is_combined_flare"] = False
    flares_df["combined_group_id"] = pd.NA
    flares_df["combined_group_size"] = 1

    gap_to_next = flares_df["new_start_pos"].shift(-1) - flares_df["new_end_pos"] - 1
    continues_into_next = gap_to_next.le(bump_gap_points).fillna(False)
    chain_start = ~continues_into_next.shift(1, fill_value=False)
    flares_df["_bump_chain_id"] = chain_start.cumsum() - 1

    combined_rows = []
    combined_windows = []

    for chain_id, group in flares_df.groupby("_bump_chain_id", sort=True):
        if len(group) < 2:
            continue

        new_start_pos = int(group["new_start_pos"].min())
        new_end_pos = int(group["new_end_pos"].max())
        start_pos = int(group["start_pos"].min())
        end_pos = int(group["end_pos"].max())
        peak_idx = ts_df["final_residual"].iloc[new_start_pos:new_end_pos + 1].idxmax()
        peak_pos = int(ts_df.index.get_indexer([peak_idx])[0])
        combined_group_id = int(chain_id)

        combined_rows.append({
            "start_pos": start_pos,
            "end_pos": end_pos,
            "n_points_above_threshold": int(group["n_points_above_threshold"].sum()),
            "start_idx": ts_df.index[start_pos],
            "end_idx": ts_df.index[end_pos],
            "start_time": ts_df["time"].iloc[start_pos],
            "end_time": ts_df["time"].iloc[end_pos],
            "peak_pos": peak_pos,
            "peak_idx": ts_df.index[peak_pos],
            "t_peak": ts_df["time"].iloc[peak_pos],
            "peak_final_residual": ts_df["final_residual"].iloc[peak_pos],
            "new_start_pos": new_start_pos,
            "new_end_pos": new_end_pos,
            "new_start_time": ts_df["time"].iloc[new_start_pos],
            "new_end_time": ts_df["time"].iloc[new_end_pos],
            "duration": ts_df["time"].iloc[new_end_pos] - ts_df["time"].iloc[new_start_pos],
            "is_combined_flare": True,
            "combined_group_id": combined_group_id,
            "combined_group_size": int(len(group)),
            "_bump_chain_id": combined_group_id,
        })
        combined_windows.append({
            "combined_group_id": combined_group_id,
            "new_start_pos": new_start_pos,
            "new_end_pos": new_end_pos,
            "new_start_time": ts_df["time"].iloc[new_start_pos],
            "new_end_time": ts_df["time"].iloc[new_end_pos],
            "n_component_flares": int(len(group)),
        })
        flares_df.loc[group.index, "combined_group_id"] = combined_group_id
        flares_df.loc[group.index, "combined_group_size"] = int(len(group))

    if combined_rows:
        flares_df = pd.concat([flares_df, pd.DataFrame(combined_rows)], ignore_index=True, sort=False)
        flares_df = flares_df.sort_values(["new_start_pos", "is_combined_flare"]).reset_index(drop=True)

    return flares_df.drop(columns=["_bump_chain_id"], errors="ignore"), pd.DataFrame(combined_windows)


def filter_by_duration_points_relation(
    ts_df: pd.DataFrame,
    flares_df: pd.DataFrame,
    sigma_col: str = "local_sigma",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[float], Optional[float], Optional[float]]:
    """Filter candidates by the duration--high-points relation.

    The relation between expanded duration and number of threshold-crossing points
    is fit in log space. Very strong peaks are rescued, matching the original
    notebook behavior.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table used for the global sigma estimate.
    flares_df : pandas.DataFrame
        Candidate table with duration and high-point counts.
    sigma_col : str, optional
        Sigma column used when ``total_sigma`` is absent.

    Returns
    -------
    tuple
        Full candidate table, included candidates, excluded candidates,
        peak-rescued candidates, fit slope, fit intercept, and relation scatter.
    """

    flares_df = flares_df.copy()
    if flares_df.empty or len(flares_df) < 3:
        flares_df["included"] = True
        return flares_df, flares_df.copy(), pd.DataFrame(), pd.DataFrame(), None, None, None

    x = flares_df["duration"].to_numpy(dtype=float) * 24
    y = flares_df["n_points_above_threshold"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 3 or len(np.unique(x[valid])) < 2:
        flares_df["included"] = True
        return flares_df, flares_df.copy(), pd.DataFrame(), pd.DataFrame(), None, None, None

    def linear_log_fit(x_value: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x_value + b

    log_y = np.log10(y[valid])
    denom = log_y.max() - log_y.min()
    y_scaled = np.zeros_like(log_y) if denom == 0 else (log_y - log_y.min()) / denom
    weights = 1 + 50 * y_scaled ** 3

    try:
        a_fit, b_fit = curve_fit(linear_log_fit, x[valid], log_y, sigma=1 / weights, absolute_sigma=False)[0]
    except Exception:
        flares_df["included"] = True
        return flares_df, flares_df.copy(), pd.DataFrame(), pd.DataFrame(), None, None, None

    flares_df["fit_y"] = 10 ** linear_log_fit(flares_df["duration"].to_numpy(dtype=float) * 24, a_fit, b_fit)
    flares_df["log_residual"] = np.log10(flares_df["n_points_above_threshold"]) - np.log10(flares_df["fit_y"])

    relation_sigma = float(np.nanstd(log_y - linear_log_fit(x[valid], a_fit, b_fit)))
    if not np.isfinite(relation_sigma) or relation_sigma == 0:
        flares_df["included"] = True
        return flares_df, flares_df.copy(), pd.DataFrame(), pd.DataFrame(), float(a_fit), float(b_fit), relation_sigma

    global_sigma = _global_sigma(ts_df, sigma_col)
    below_3sigma = flares_df["log_residual"] < -3 * relation_sigma
    strong_peak = flares_df["peak_final_residual"] > 5 * global_sigma

    flares_df["re_included_by_peak"] = below_3sigma & strong_peak
    flares_df["still_excluded"] = below_3sigma & ~strong_peak
    flares_df["included"] = ~flares_df["still_excluded"]

    return (
        flares_df,
        flares_df[flares_df["included"]].reset_index(drop=True),
        flares_df[below_3sigma].copy().reset_index(drop=True),
        flares_df[flares_df["re_included_by_peak"]].reset_index(drop=True),
        float(a_fit),
        float(b_fit),
        relation_sigma,
    )


def apply_late_peak_rule(
    ts_df: pd.DataFrame,
    flares_df_3sigma: pd.DataFrame,
    sigma_col: str = "local_sigma",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove candidates whose peak occurs too late in the window.

    Late peaks are kept when they are close to another flare or when the peak is
    very strong.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table used for the global sigma estimate.
    flares_df_3sigma : pandas.DataFrame
        Candidates that survived the duration/high-points filter.
    sigma_col : str, optional
        Sigma column used when ``total_sigma`` is absent.

    Returns
    -------
    tuple of pandas.DataFrame
        Final candidates after the rule, all late-peak candidates, and candidates
        excluded by the rule.
    """

    flares_df_3sigma = flares_df_3sigma.sort_values("new_start_pos").reset_index(drop=True).copy()
    if flares_df_3sigma.empty:
        return flares_df_3sigma.copy(), flares_df_3sigma.copy(), flares_df_3sigma.copy()

    peak_position_limit = 0.70
    close_gap_points = 20
    duration = flares_df_3sigma["new_end_time"] - flares_df_3sigma["new_start_time"]
    flares_df_3sigma["relative_peak_position"] = np.where(
        duration != 0,
        (flares_df_3sigma["t_peak"] - flares_df_3sigma["new_start_time"]) / duration,
        0,
    )
    peak_too_late = flares_df_3sigma["relative_peak_position"] > peak_position_limit
    gap_to_previous = flares_df_3sigma["new_start_pos"] - flares_df_3sigma["new_end_pos"].shift(1) - 1
    gap_to_next = flares_df_3sigma["new_start_pos"].shift(-1) - flares_df_3sigma["new_end_pos"] - 1
    close_to_other = ((gap_to_previous <= close_gap_points) | (gap_to_next <= close_gap_points)).fillna(False)
    strong_peak = flares_df_3sigma["peak_final_residual"] > 5 * _global_sigma(ts_df, sigma_col)

    flares_df_3sigma["peak_too_late"] = peak_too_late
    flares_df_3sigma["close_to_other_flare"] = close_to_other
    flares_df_3sigma["re_included_by_nearby_flare"] = peak_too_late & close_to_other
    flares_df_3sigma["re_included_by_strong_peak"] = peak_too_late & ~close_to_other & strong_peak
    flares_df_3sigma["excluded_by_peak_rule"] = peak_too_late & ~close_to_other & ~strong_peak

    late_peak_flares_df = flares_df_3sigma[peak_too_late].reset_index(drop=True)
    final_flares_df = flares_df_3sigma[~flares_df_3sigma["excluded_by_peak_rule"]].reset_index(drop=True)
    excluded_peak_rule_flares_df = flares_df_3sigma[flares_df_3sigma["excluded_by_peak_rule"]].reset_index(drop=True)
    return final_flares_df, late_peak_flares_df, excluded_peak_rule_flares_df


def remove_failed_temporary_combined_flares(
    final_flares_df: pd.DataFrame,
    combined_flare_windows_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int]:
    """Remove temporary combined rows and failed component rows.

    If a temporary combined flare fails later filters, component rows inside that
    failed combined window are removed as in the original notebook.

    Parameters
    ----------
    final_flares_df : pandas.DataFrame
        Candidate table after the late-peak rule.
    combined_flare_windows_df : pandas.DataFrame
        Temporary combined-window metadata.

    Returns
    -------
    tuple
        Cleaned flare table, number of component rows removed, and number of
        temporary combined rows removed.
    """

    final_flares_df = final_flares_df.copy()
    if final_flares_df.empty:
        return final_flares_df, 0, 0
    if "is_combined_flare" not in final_flares_df.columns:
        final_flares_df["is_combined_flare"] = False

    if not combined_flare_windows_df.empty:
        surviving_ids = set(final_flares_df.loc[final_flares_df["is_combined_flare"], "combined_group_id"].dropna().astype(int))
        all_ids = set(combined_flare_windows_df["combined_group_id"].astype(int))
        failed_ids = all_ids - surviving_ids
        remove_component = pd.Series(False, index=final_flares_df.index)
        for _, combined in combined_flare_windows_df[combined_flare_windows_df["combined_group_id"].isin(failed_ids)].iterrows():
            inside = (
                ~final_flares_df["is_combined_flare"]
                & (final_flares_df["new_start_pos"] >= combined["new_start_pos"])
                & (final_flares_df["new_end_pos"] <= combined["new_end_pos"])
            )
            remove_component |= inside
        n_removed_components = int(remove_component.sum())
        n_removed_combined = int(final_flares_df["is_combined_flare"].sum())
        final_flares_df = final_flares_df[~remove_component & ~final_flares_df["is_combined_flare"]].reset_index(drop=True)
    else:
        n_removed_components = 0
        n_removed_combined = int(final_flares_df["is_combined_flare"].sum())
        final_flares_df = final_flares_df[~final_flares_df["is_combined_flare"]].reset_index(drop=True)

    return final_flares_df, n_removed_components, n_removed_combined


def run_flare_finder(
    ts_df: pd.DataFrame,
    config: Optional[FlareFinderConfig] = None,
    sigma_col: str = "local_sigma",
    label: str = "pass",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one complete flare-finder pass using one sigma column.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    config : FlareFinderConfig or None, optional
        Detection settings. Defaults are used when omitted.
    sigma_col : str, optional
        Column used as the local sigma threshold.
    label : str, optional
        Label shown in progress messages.
    verbose : bool, optional
        Print a compact summary of the pass.

    Returns
    -------
    dict
        Intermediate and final DataFrames for this pass, plus relation-fit metadata.

    Raises
    ------
    KeyError
        If required columns are missing in called detection steps.
    """

    config = config or FlareFinderConfig()
    flares_df = find_candidate_flares(
        ts_df,
        sigma_col=sigma_col,
        lower_sigma=config.lower_sigma,
        moving_average_sigma=config.moving_average_sigma,
        n_consecutive_points=config.n_consecutive_points,
        max_below_threshold=config.max_below_threshold,
        min_gap_distance_points=config.min_gap_distance_points,
    )
    candidate_count = len(flares_df)

    if flares_df.empty:
        if verbose:
            print(f"{label}: no candidate flares found.")
        return {
            "candidate_flares_df": flares_df,
            "expanded_flares_df": flares_df,
            "combined_flare_windows_df": pd.DataFrame(),
            "flares_df_3sigma": flares_df,
            "excluded_flares_df": pd.DataFrame(),
            "re_included_flares_df": pd.DataFrame(),
            "late_peak_flares_df": pd.DataFrame(),
            "excluded_peak_rule_flares_df": pd.DataFrame(),
            "final_flares_df": flares_df,
            "relation_fit": (None, None, None),
        }

    flares_df = expand_flare_windows(ts_df, flares_df)
    flares_df, combined_flare_windows_df = add_temporary_combined_flares(
        ts_df,
        flares_df,
        bump_gap_points=config.temporary_combined_bump_gap_points,
    )
    (
        flares_df,
        flares_df_3sigma,
        excluded_flares_df,
        re_included_flares_df,
        a_fit,
        b_fit,
        relation_sigma,
    ) = filter_by_duration_points_relation(ts_df, flares_df, sigma_col=sigma_col)

    final_flares_df, late_peak_flares_df, excluded_peak_rule_flares_df = apply_late_peak_rule(
        ts_df,
        flares_df_3sigma,
        sigma_col=sigma_col,
    )
    final_flares_df, n_removed_components, n_removed_combined = remove_failed_temporary_combined_flares(
        final_flares_df,
        combined_flare_windows_df,
    )

    if verbose:
        print(f"{label}: candidate flares before expansion: {candidate_count}")
        print(f"{label}: after duration/n-points filter: {len(flares_df_3sigma)}")
        print(f"{label}: late-peak exclusions: {len(excluded_peak_rule_flares_df)}")
        print(f"{label}: temporary combined flares removed: {n_removed_combined}")
        print(f"{label}: component flares removed because combined flare failed: {n_removed_components}")
        print(f"{label}: final accepted individual flares: {len(final_flares_df)}")

    return {
        "candidate_flares_df": flares_df,
        "expanded_flares_df": flares_df,
        "combined_flare_windows_df": combined_flare_windows_df,
        "flares_df_3sigma": flares_df_3sigma,
        "excluded_flares_df": excluded_flares_df,
        "re_included_flares_df": re_included_flares_df,
        "late_peak_flares_df": late_peak_flares_df,
        "excluded_peak_rule_flares_df": excluded_peak_rule_flares_df,
        "final_flares_df": final_flares_df,
        "relation_fit": (a_fit, b_fit, relation_sigma),
    }


def choose_local_noise_window_points(ts_df: pd.DataFrame, config: Optional[FlareFinderConfig] = None) -> int:
    """Choose the rolling-window size for cleaned local noise.

    The function first uses ``config.local_noise_window_points``. If that is unset,
    it uses ``config.local_noise_window_days``. If neither is set, it uses the median
    ``selected_window_size`` column when available, otherwise 0.8 days.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table containing ``time`` and optionally ``selected_window_size``.
    config : FlareFinderConfig or None, optional
        Pipeline settings.

    Returns
    -------
    int
        Odd rolling-window size in cadences.

    Raises
    ------
    KeyError
        If ``time`` is missing and a point-based window was not supplied.
    """

    config = config or FlareFinderConfig()
    if config.local_noise_window_points is not None:
        return int(config.local_noise_window_points)

    _require_columns(ts_df, ["time"], "local noise window selection")
    dt = ts_df["time"].diff().median()
    if config.local_noise_window_days is not None:
        days = float(config.local_noise_window_days)
    elif "selected_window_size" in ts_df.columns and ts_df["selected_window_size"].notna().any():
        days = float(ts_df["selected_window_size"].median())
    else:
        days = 0.8

    if pd.isna(dt) or dt <= 0:
        return 101

    points = max(int(round(days / dt)), 11)
    if points % 2 == 0:
        points += 1
    return points


def build_strong_flare_noise_mask(
    ts_df: pd.DataFrame,
    first_pass_final_flares_df: pd.DataFrame,
    config: Optional[FlareFinderConfig] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build the mask used for second-pass local-noise estimation.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    first_pass_final_flares_df : pandas.DataFrame
        Accepted first-pass flares.
    config : FlareFinderConfig or None, optional
        Pipeline settings.

    Returns
    -------
    tuple
        Boolean keep mask and first-pass flares considered strong enough to mask.
    """

    config = config or FlareFinderConfig()
    keep_for_noise = np.ones(len(ts_df), dtype=bool)
    if first_pass_final_flares_df is None or first_pass_final_flares_df.empty:
        return keep_for_noise, pd.DataFrame()

    global_sigma = _global_sigma(ts_df, "local_sigma")
    strong_flares_df = first_pass_final_flares_df[
        first_pass_final_flares_df["peak_final_residual"] > config.strong_flare_peak_sigma * global_sigma
    ].copy()

    for _, flare in strong_flares_df.iterrows():
        start = max(0, int(flare["new_start_pos"]) - config.strong_flare_mask_padding_points)
        end = min(len(ts_df) - 1, int(flare["new_end_pos"]) + config.strong_flare_mask_padding_points)
        keep_for_noise[start:end + 1] = False

    return keep_for_noise, strong_flares_df


def rolling_std_with_masked_points(values: np.ndarray, keep_mask: np.ndarray, window_points: int) -> np.ndarray:
    """Compute centered rolling scatter after masking selected points.

    Masked values are linearly interpolated for the noise estimate only. The signal
    used for flare detection is not modified.

    Parameters
    ----------
    values : numpy.ndarray
        Residual values used for the scatter estimate.
    keep_mask : numpy.ndarray
        Boolean mask where True marks values kept for noise estimation.
    window_points : int
        Rolling-window size in cadences.

    Returns
    -------
    numpy.ndarray
        Local rolling standard deviation.
    """

    values_series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    cleaned = values_series.mask(~keep_mask).interpolate(method="linear", limit_direction="both")
    local_std = cleaned.rolling(window=window_points, center=True, min_periods=max(5, window_points // 4)).std()
    local_std = local_std.interpolate(method="linear", limit_direction="both")
    if local_std.isna().any():
        local_std = local_std.fillna(cleaned.std(skipna=True))
    return local_std.to_numpy()


def add_clean_local_sigma(
    ts_df: pd.DataFrame,
    first_pass_final_flares_df: pd.DataFrame,
    config: Optional[FlareFinderConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Add cleaned local-noise columns for the second pass.

    Strong first-pass flares are masked only while estimating noise. The returned
    DataFrame keeps the original residuals and moving average for detection.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Input time-series table.
    first_pass_final_flares_df : pandas.DataFrame
        Accepted first-pass flares.
    config : FlareFinderConfig or None, optional
        Pipeline settings.
    verbose : bool, optional
        Print summary information.

    Returns
    -------
    tuple
        Second-pass time-series table, masked strong flares, and noise-window size.

    Raises
    ------
    KeyError
        If required columns are missing.
    """

    config = config or FlareFinderConfig()
    _require_columns(ts_df, ["time", config.residual_col_for_noise, "local_sigma"], "clean local sigma")
    ts_df_clean = ts_df.copy()
    window_points = choose_local_noise_window_points(ts_df_clean, config)
    keep_for_noise, strong_flares_df = build_strong_flare_noise_mask(ts_df_clean, first_pass_final_flares_df, config)
    clean_local_std = rolling_std_with_masked_points(
        ts_df_clean[config.residual_col_for_noise].to_numpy(),
        keep_for_noise,
        window_points,
    )

    ts_df_clean["noise_keep_mask_second_pass"] = keep_for_noise
    ts_df_clean["local_std_clean"] = clean_local_std
    if config.flux_err_col in ts_df_clean.columns:
        ts_df_clean["local_sigma_clean_with_flux_err"] = np.sqrt(
            ts_df_clean[config.flux_err_col].to_numpy(dtype=float) ** 2 + ts_df_clean["local_std_clean"].to_numpy(dtype=float) ** 2
        )
    else:
        ts_df_clean["local_sigma_clean_with_flux_err"] = ts_df_clean["local_std_clean"]

    ts_df_clean["local_sigma_clean"] = (
        ts_df_clean["local_sigma_clean_with_flux_err"] if config.include_flux_err_in_detection_sigma else ts_df_clean["local_std_clean"]
    )

    if verbose:
        print(f"Second-pass local-noise window: {window_points} points")
        print(f"Strong first-pass flares masked for noise estimate: {len(strong_flares_df)}")
        print(f"Rows excluded only from noise estimate: {(~keep_for_noise).sum():,}")

    return ts_df_clean, strong_flares_df, window_points


def _empty_unique_flare_catalog() -> pd.DataFrame:
    """Return an empty unique-flare catalog with stable columns.

    Returns
    -------
    pandas.DataFrame
        Empty catalog using the same columns as non-empty results.
    """

    return pd.DataFrame(columns=[
        "unique_flare_id", "new_start_pos", "new_end_pos", "new_start_time", "new_end_time",
        "start_pos", "end_pos", "start_time", "end_time", "peak_pos", "peak_idx", "t_peak",
        "peak_final_residual", "duration", "n_points_above_threshold", "found_in_first_pass",
        "found_in_second_pass", "source_passes", "n_pass_detections_merged",
        "first_pass_detection_count", "second_pass_detection_count", "first_pass_peak_final_residual",
        "second_pass_peak_final_residual",
    ])


def make_unique_flare_catalog(
    ts_df: pd.DataFrame,
    first_pass_flares_df: pd.DataFrame,
    second_pass_flares_df: pd.DataFrame,
    merge_gap_points: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge first-pass and second-pass detections into one catalog.

    Detections are considered the same flare when their expanded windows overlap or
    are separated by no more than ``merge_gap_points`` cadences.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table used to recompute unified flare properties.
    first_pass_flares_df : pandas.DataFrame
        Accepted first-pass flares.
    second_pass_flares_df : pandas.DataFrame
        Accepted second-pass flares.
    merge_gap_points : int, optional
        Maximum gap, in cadences, for detections to be merged.

    Returns
    -------
    tuple of pandas.DataFrame
        Unique flare catalog and all pass detections with merge metadata.
    """

    pieces = []
    if first_pass_flares_df is not None and not first_pass_flares_df.empty:
        fp = first_pass_flares_df.copy()
        fp["source_pass"] = "first_pass"
        fp["source_sigma_col"] = "local_sigma"
        pieces.append(fp)
    if second_pass_flares_df is not None and not second_pass_flares_df.empty:
        sp = second_pass_flares_df.copy()
        sp["source_pass"] = "second_pass"
        sp["source_sigma_col"] = "local_sigma_clean"
        pieces.append(sp)

    if not pieces:
        return _empty_unique_flare_catalog(), pd.DataFrame()

    all_detections = pd.concat(pieces, ignore_index=True, sort=False)
    all_detections = all_detections.sort_values(["new_start_pos", "new_end_pos", "source_pass"]).reset_index(drop=True)

    group_ids = []
    current_group = -1
    current_group_end = None
    for _, row in all_detections.iterrows():
        start = int(row["new_start_pos"])
        end = int(row["new_end_pos"])
        if current_group_end is None or start > current_group_end + merge_gap_points:
            current_group += 1
            current_group_end = end
        else:
            current_group_end = max(current_group_end, end)
        group_ids.append(current_group)
    all_detections["unique_flare_id"] = group_ids

    unique_rows = []
    for unique_id, group in all_detections.groupby("unique_flare_id", sort=True):
        new_start_pos = int(group["new_start_pos"].min())
        new_end_pos = int(group["new_end_pos"].max())
        start_pos = int(group["start_pos"].min()) if "start_pos" in group else new_start_pos
        end_pos = int(group["end_pos"].max()) if "end_pos" in group else new_end_pos
        peak_idx_label = ts_df["final_residual"].iloc[new_start_pos:new_end_pos + 1].idxmax()
        peak_pos = int(ts_df.index.get_indexer([peak_idx_label])[0])
        source_passes = sorted(group["source_pass"].dropna().unique().tolist())
        first_peaks = group.loc[group["source_pass"].eq("first_pass"), "peak_final_residual"]
        second_peaks = group.loc[group["source_pass"].eq("second_pass"), "peak_final_residual"]

        unique_rows.append({
            "unique_flare_id": int(unique_id),
            "new_start_pos": new_start_pos,
            "new_end_pos": new_end_pos,
            "new_start_time": ts_df["time"].iloc[new_start_pos],
            "new_end_time": ts_df["time"].iloc[new_end_pos],
            "start_pos": start_pos,
            "end_pos": end_pos,
            "start_time": ts_df["time"].iloc[start_pos],
            "end_time": ts_df["time"].iloc[end_pos],
            "peak_pos": peak_pos,
            "peak_idx": ts_df.index[peak_pos],
            "t_peak": ts_df["time"].iloc[peak_pos],
            "peak_final_residual": ts_df["final_residual"].iloc[peak_pos],
            "duration": ts_df["time"].iloc[new_end_pos] - ts_df["time"].iloc[new_start_pos],
            "n_points_above_threshold": int(group.get("n_points_above_threshold", pd.Series(dtype=float)).sum()),
            "found_in_first_pass": "first_pass" in source_passes,
            "found_in_second_pass": "second_pass" in source_passes,
            "source_passes": "+".join(source_passes),
            "n_pass_detections_merged": int(len(group)),
            "first_pass_detection_count": int(group["source_pass"].eq("first_pass").sum()),
            "second_pass_detection_count": int(group["source_pass"].eq("second_pass").sum()),
            "first_pass_peak_final_residual": first_peaks.max() if len(first_peaks) else np.nan,
            "second_pass_peak_final_residual": second_peaks.max() if len(second_peaks) else np.nan,
        })

    unique_flares_df = pd.DataFrame(unique_rows).sort_values("new_start_pos").reset_index(drop=True)
    unique_flares_df["unique_flare_id"] = np.arange(len(unique_flares_df), dtype=int)
    all_detections = all_detections.merge(
        unique_flares_df[["unique_flare_id", "source_passes"]],
        on="unique_flare_id",
        how="left",
        suffixes=("", "_unique"),
    )
    return unique_flares_df, all_detections


def summarize_flares(name: str, flares_df: pd.DataFrame) -> pd.Series:
    """Summarize a flare catalog in one compact row.

    Parameters
    ----------
    name : str
        Name written to the ``catalog`` column.
    flares_df : pandas.DataFrame
        Flare catalog to summarize.

    Returns
    -------
    pandas.Series
        Summary row with counts, median duration, maximum peak, and total time.
    """

    if flares_df is None or flares_df.empty:
        return pd.Series({
            "catalog": name,
            "n_flares": 0,
            "median_duration_hr": np.nan,
            "max_peak_residual": np.nan,
            "total_window_time_days": 0.0,
        })
    return pd.Series({
        "catalog": name,
        "n_flares": len(flares_df),
        "median_duration_hr": np.nanmedian(flares_df["duration"] * 24),
        "max_peak_residual": np.nanmax(flares_df["peak_final_residual"]),
        "total_window_time_days": np.nansum(flares_df["duration"]),
    })


def mark_flare_mask(ts_df: pd.DataFrame, flares_df: pd.DataFrame, mask_col: str) -> pd.DataFrame:
    """Return a copy of a time-series table with a flare-window mask.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Input time-series table.
    flares_df : pandas.DataFrame
        Flare catalog containing expanded window positions.
    mask_col : str
        Name of the boolean mask column to add.

    Returns
    -------
    pandas.DataFrame
        Copy of ``ts_df`` with ``mask_col`` added.
    """

    out = ts_df.copy()
    out[mask_col] = False
    if flares_df is None or flares_df.empty:
        return out
    for _, flare in flares_df.iterrows():
        out.loc[int(flare["new_start_pos"]): int(flare["new_end_pos"]), mask_col] = True
    return out


def apply_ma10_residual_filter(
    ts_df: pd.DataFrame,
    flares_df: pd.DataFrame,
    sigma_col: str = "local_sigma_clean",
    ma10_filter_sigma: float = 2.0,
    residual_filter_sigma: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter weak candidates after building the unique catalog.

    A flare is kept if either ``ma_10`` reaches ``ma10_filter_sigma`` times the
    chosen sigma column, or ``final_residual`` reaches ``residual_filter_sigma``
    times that sigma column inside the flare window.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    flares_df : pandas.DataFrame
        Unique flare catalog.
    sigma_col : str, optional
        Sigma column used for the post-filter.
    ma10_filter_sigma : float, optional
        Sigma multiplier for the moving-average condition.
    residual_filter_sigma : float, optional
        Sigma multiplier for the residual condition.

    Returns
    -------
    tuple of pandas.DataFrame
        Kept flares and removed flares.

    Raises
    ------
    KeyError
        If required time-series columns are missing.
    """

    if flares_df is None or flares_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    _require_columns(ts_df, ["time", "ma_10", "final_residual", sigma_col], "ma_10/residual filtering")

    kept_rows = []
    removed_rows = []
    for _, flare in flares_df.iterrows():
        in_flare = (ts_df["time"] >= flare["new_start_time"]) & (ts_df["time"] <= flare["new_end_time"])
        points = ts_df.loc[in_flare]
        if points.empty:
            removed_rows.append(flare)
            continue
        ma10 = points["ma_10"].to_numpy(dtype=float)
        residual = points["final_residual"].to_numpy(dtype=float)
        sigma = points[sigma_col].to_numpy(dtype=float)
        valid_ma10 = np.isfinite(ma10) & np.isfinite(sigma)
        valid_residual = np.isfinite(residual) & np.isfinite(sigma)
        keep = np.any(valid_ma10 & (ma10 >= ma10_filter_sigma * sigma)) or np.any(
            valid_residual & (residual >= residual_filter_sigma * sigma)
        )
        if keep:
            kept_rows.append(flare)
        else:
            removed_rows.append(flare)

    return pd.DataFrame(kept_rows).reset_index(drop=True), pd.DataFrame(removed_rows).reset_index(drop=True)


def _safe_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Convert a value to ``int`` unless it is missing.

    Parameters
    ----------
    value : object
        Value to convert.
    fallback : int or None, optional
        Value returned when ``value`` is missing.

    Returns
    -------
    int or None
        Converted integer or ``fallback``.
    """

    if pd.isna(value):
        return fallback
    return int(value)


def _map_ma10_peak_to_datapoint(
    ts_df: pd.DataFrame,
    ma10_peak_pos: int,
    config: FlareFinderConfig,
    search_left: Optional[int] = None,
    search_right: Optional[int] = None,
) -> int:
    """Map a moving-average peak to the nearest datapoint peak.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    ma10_peak_pos : int
        Position of the moving-average peak.
    config : FlareFinderConfig
        Pipeline settings.
    search_left : int or None, optional
        Left search boundary. Defaults to the configured correspondence radius.
    search_right : int or None, optional
        Right search boundary. Defaults to the configured correspondence radius.

    Returns
    -------
    int
        Position of the highest datapoint residual inside the search window.
    """

    n = len(ts_df)
    ma10_peak_pos = int(max(0, min(int(ma10_peak_pos), n - 1)))
    if search_left is None:
        search_left = ma10_peak_pos - config.multi_peak_correspondence_radius_points
    if search_right is None:
        search_right = ma10_peak_pos + config.multi_peak_correspondence_radius_points
    search_left = max(0, min(int(search_left), n - 1))
    search_right = max(0, min(int(search_right), n - 1))
    if search_right < search_left:
        search_left, search_right = search_right, search_left
    assert 0 <= search_left <= search_right < n, "Search bounds must be valid row positions."
    values = ts_df[config.multi_peak_datapoint_column].iloc[search_left:search_right + 1].to_numpy(dtype=float)
    if len(values) == 0 or np.all(~np.isfinite(values)):
        return ma10_peak_pos
    return search_left + int(np.nanargmax(values))


def _best_ma10_peak_in_window(ts_df: pd.DataFrame, start_pos: int, end_pos: int, config: FlareFinderConfig) -> int:
    """Return the strongest moving-average peak inside a window.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    start_pos : int
        Inclusive left window boundary.
    end_pos : int
        Inclusive right window boundary.
    config : FlareFinderConfig
        Pipeline settings.

    Returns
    -------
    int
        Position of the strongest finite moving-average value, or ``start_pos``.
    """

    values = ts_df[config.multi_peak_detection_column].iloc[start_pos:end_pos + 1].to_numpy(dtype=float)
    if len(values) == 0 or np.all(~np.isfinite(values)):
        return int(start_pos)
    return int(start_pos + np.nanargmax(values))


def _preserve_original_catalog_peak(row: dict[str, Any], parent_row: pd.Series) -> dict[str, Any]:
    """Store the parent catalog peak before recomputing a split row.

    Parameters
    ----------
    row : dict
        Row dictionary to update.
    parent_row : pandas.Series
        Original parent flare row.

    Returns
    -------
    dict
        Updated row dictionary.
    """

    for col in ["peak_pos", "peak_idx", "t_peak", "peak_final_residual"]:
        row[f"original_catalog_{col}"] = parent_row.get(col, np.nan)
    return row


def _recompute_flare_row_from_window(
    ts_df: pd.DataFrame,
    parent_row: pd.Series,
    start_pos: int,
    end_pos: int,
    config: FlareFinderConfig,
    ma10_peak_pos: Optional[int] = None,
    subflare_id: int = 0,
    n_subflares: int = 1,
) -> Optional[dict[str, Any]]:
    """Recompute flare-row values for a proposed sub-window.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    parent_row : pandas.Series
        Original flare row used as the template.
    start_pos : int
        Inclusive left boundary of the new window.
    end_pos : int
        Inclusive right boundary of the new window.
    config : FlareFinderConfig
        Pipeline settings.
    ma10_peak_pos : int or None, optional
        Moving-average peak assigned to this sub-window.
    subflare_id : int, optional
        Index of this sub-flare within the parent window.
    n_subflares : int, optional
        Total number of sub-flares proposed for the parent.

    Returns
    -------
    dict or None
        Recomputed flare row, or None when the proposed window is empty.
    """

    n = len(ts_df)
    start_pos = max(0, min(int(start_pos), n - 1))
    end_pos = max(0, min(int(end_pos), n - 1))
    if end_pos < start_pos:
        start_pos, end_pos = end_pos, start_pos
    assert 0 <= start_pos <= end_pos < n, "Flare window bounds must be valid row positions."
    if ts_df.iloc[start_pos:end_pos + 1].empty:
        return None
    if ma10_peak_pos is None or not (start_pos <= int(ma10_peak_pos) <= end_pos):
        ma10_peak_pos = _best_ma10_peak_in_window(ts_df, start_pos, end_pos, config)
    ma10_peak_pos = int(ma10_peak_pos)
    datapoint_peak_pos = _map_ma10_peak_to_datapoint(
        ts_df,
        ma10_peak_pos,
        config,
        search_left=max(start_pos, ma10_peak_pos - config.multi_peak_correspondence_radius_points),
        search_right=min(end_pos, ma10_peak_pos + config.multi_peak_correspondence_radius_points),
    )

    row = _preserve_original_catalog_peak(parent_row.to_dict(), parent_row)
    row["new_start_pos"] = int(start_pos)
    row["new_end_pos"] = int(end_pos)
    row["new_start_time"] = ts_df["time"].iloc[start_pos]
    row["new_end_time"] = ts_df["time"].iloc[end_pos]
    row["start_pos"] = int(start_pos)
    row["end_pos"] = int(end_pos)
    row["start_time"] = ts_df["time"].iloc[start_pos]
    row["end_time"] = ts_df["time"].iloc[end_pos]

    row["ma10_peak_pos"] = int(ma10_peak_pos)
    row["ma10_peak_idx"] = ts_df.index[ma10_peak_pos]
    row["ma10_t_peak"] = ts_df["time"].iloc[ma10_peak_pos]
    row["ma10_peak_value"] = ts_df[config.multi_peak_detection_column].iloc[ma10_peak_pos]

    row["corresponding_datapoint_peak_pos"] = int(datapoint_peak_pos)
    row["corresponding_datapoint_peak_idx"] = ts_df.index[datapoint_peak_pos]
    row["corresponding_datapoint_t_peak"] = ts_df["time"].iloc[datapoint_peak_pos]
    row["corresponding_datapoint_peak_final_residual"] = ts_df[config.multi_peak_datapoint_column].iloc[datapoint_peak_pos]

    row["peak_pos"] = int(datapoint_peak_pos)
    row["peak_idx"] = ts_df.index[datapoint_peak_pos]
    row["t_peak"] = ts_df["time"].iloc[datapoint_peak_pos]
    row["peak_final_residual"] = ts_df[config.multi_peak_datapoint_column].iloc[datapoint_peak_pos]
    row["duration"] = ts_df["time"].iloc[end_pos] - ts_df["time"].iloc[start_pos]

    sigma_col = "local_sigma_clean" if "local_sigma_clean" in ts_df.columns else "local_sigma"
    residual = ts_df[config.multi_peak_datapoint_column].iloc[start_pos:end_pos + 1].to_numpy(dtype=float)
    sigma = ts_df[sigma_col].iloc[start_pos:end_pos + 1].to_numpy(dtype=float)
    valid = np.isfinite(residual) & np.isfinite(sigma)
    row["n_points_above_threshold"] = int(np.sum(valid & (residual >= config.multi_peak_min_peak_sigma * sigma)))

    row["was_split_from_multi_peak"] = bool(n_subflares > 1)
    row["parent_unique_flare_id"] = parent_row.get("unique_flare_id", np.nan)
    row["parent_new_start_pos"] = parent_row.get("new_start_pos", np.nan)
    row["parent_new_end_pos"] = parent_row.get("new_end_pos", np.nan)
    row["parent_new_start_time"] = parent_row.get("new_start_time", np.nan)
    row["parent_new_end_time"] = parent_row.get("new_end_time", np.nan)
    row["parent_peak_pos"] = parent_row.get("peak_pos", np.nan)
    row["parent_t_peak"] = parent_row.get("t_peak", np.nan)
    row["parent_peak_final_residual"] = parent_row.get("peak_final_residual", np.nan)
    row["subflare_id"] = int(subflare_id)
    row["n_subflares_from_parent"] = int(n_subflares)
    return row


def _find_multi_peak_split_points(
    ts_df: pd.DataFrame,
    flare: pd.Series,
    config: FlareFinderConfig,
    sigma_col: str = "local_sigma_clean",
) -> Tuple[np.ndarray, np.ndarray, list[int]]:
    """Find valid split points inside a complex flare window.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    flare : pandas.Series
        Parent flare row.
    config : FlareFinderConfig
        Pipeline settings.
    sigma_col : str, optional
        Sigma column used for multi-peak thresholds.

    Returns
    -------
    tuple
        Moving-average peak positions, corresponding datapoint peak positions, and
        valid valley positions used as split boundaries.
    """

    n = len(ts_df)
    start = max(0, _safe_int(flare["new_start_pos"], 0))
    end = min(n - 1, _safe_int(flare["new_end_pos"], n - 1))
    if end <= start:
        return np.array([], dtype=int), np.array([], dtype=int), []

    ma_values = ts_df[config.multi_peak_detection_column].iloc[start:end + 1].to_numpy(dtype=float)
    sigma = ts_df[sigma_col].iloc[start:end + 1].to_numpy(dtype=float)
    finite = np.isfinite(ma_values) & np.isfinite(sigma)
    if finite.sum() < max(3, config.multi_peak_min_segment_points):
        return np.array([], dtype=int), np.array([], dtype=int), []

    ma_for_peaks = ma_values.copy()
    ma_for_peaks[~np.isfinite(ma_for_peaks)] = -np.inf
    height_threshold = config.multi_peak_min_peak_sigma * sigma
    height_threshold[~np.isfinite(height_threshold)] = np.inf
    peaks_local, _ = find_peaks(ma_for_peaks, height=height_threshold, distance=config.multi_peak_min_separation_points)

    if len(peaks_local) < 2:
        ma10_peaks_global = start + peaks_local
        datapoint_peaks_global = np.array([
            _map_ma10_peak_to_datapoint(
                ts_df,
                int(p),
                config,
                max(start, int(p) - config.multi_peak_correspondence_radius_points),
                min(end, int(p) + config.multi_peak_correspondence_radius_points),
            )
            for p in ma10_peaks_global
        ], dtype=int)
        return ma10_peaks_global, datapoint_peaks_global, []

    valid_split_valleys: list[int] = []
    kept_peak_indices = [0]
    for i in range(len(peaks_local) - 1):
        left_peak = int(peaks_local[i])
        right_peak = int(peaks_local[i + 1])
        if right_peak <= left_peak:
            continue
        between = ma_values[left_peak:right_peak + 1]
        if np.all(~np.isfinite(between)):
            continue
        valley_local = left_peak + int(np.nanargmin(between))
        valley_value = ma_values[valley_local]
        left_value = ma_values[left_peak]
        right_value = ma_values[right_peak]
        valley_sigma = sigma[valley_local]
        valley_deep_by_peak_ratio = valley_value <= config.multi_peak_valley_fraction * min(left_value, right_value)
        valley_deep_by_sigma = np.isfinite(valley_sigma) and valley_value <= config.multi_peak_valley_sigma * valley_sigma
        previous_boundary_local = valid_split_valleys[-1] - start + 1 if valid_split_valleys else 0
        left_segment_len = valley_local - previous_boundary_local + 1
        right_segment_len = right_peak - valley_local
        segment_lengths_ok = left_segment_len >= config.multi_peak_min_segment_points and right_segment_len >= config.multi_peak_min_segment_points
        if (valley_deep_by_peak_ratio or valley_deep_by_sigma) and segment_lengths_ok:
            valid_split_valleys.append(start + valley_local)
            kept_peak_indices.append(i + 1)

    if not valid_split_valleys:
        ma10_peaks_global = start + peaks_local
    else:
        ma10_peaks_global = start + peaks_local[kept_peak_indices]

    datapoint_peaks_global = np.array([
        _map_ma10_peak_to_datapoint(
            ts_df,
            int(p),
            config,
            max(start, int(p) - config.multi_peak_correspondence_radius_points),
            min(end, int(p) + config.multi_peak_correspondence_radius_points),
        )
        for p in ma10_peaks_global
    ], dtype=int)
    return ma10_peaks_global, datapoint_peaks_global, valid_split_valleys


def plot_multi_peak_candidate(
    ts_df: pd.DataFrame,
    flare: pd.Series,
    ma10_peaks_global: np.ndarray,
    datapoint_peaks_global: np.ndarray,
    split_valleys: list[int],
    parent_idx: int,
    config: Optional[FlareFinderConfig] = None,
    results_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot one multi-peak split candidate in thesis style.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    flare : pandas.Series
        Parent flare row.
    ma10_peaks_global : numpy.ndarray
        Moving-average peak positions in global row coordinates.
    datapoint_peaks_global : numpy.ndarray
        Datapoint peak positions in global row coordinates.
    split_valleys : list of int
        Valley positions used as proposed split points.
    parent_idx : int
        Row number of the parent flare in the input catalog.
    config : FlareFinderConfig or None, optional
        Pipeline settings.
    results_dir : str, pathlib.Path, or None, optional
        Directory where the individual PDF is saved. If None, the figure is shown
        only when ``show`` is True.
    prefix : str or None, optional
        Filename prefix for the saved PDF.
    show : bool, optional
        Show the figure interactively instead of silently closing when no
        ``results_dir`` is supplied.

    Returns
    -------
    pathlib.Path or None
        Saved PDF path, or None when the figure was not saved.
    """

    config = config or FlareFinderConfig()
    start = int(flare["new_start_pos"])
    end = int(flare["new_end_pos"])
    t0 = flare["new_start_time"]
    t1 = flare["new_end_time"]
    plot_mask = (ts_df["time"] >= t0 - config.multi_peak_plot_padding_days) & (ts_df["time"] <= t1 + config.multi_peak_plot_padding_days)
    local_df = ts_df.loc[plot_mask]
    if local_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        local_df["time"],
        local_df[config.multi_peak_detection_column],
        color=THESIS_MAIN_COLOR,
        lw=LINEWIDTH,
        zorder=4,
    )
    ax.scatter(
        local_df["time"],
        local_df[config.multi_peak_datapoint_column],
        s=22 * SCALE,
        alpha=THESIS_RAW_ALPHA,
        color=THESIS_RAW_COLOR,
        edgecolors="none",
        label="Final Residual",
        zorder=2,
    )
    if "local_sigma_clean" in local_df.columns:
        ax.plot(
            local_df["time"],
            3 * local_df["local_sigma_clean"],
            color=THESIS_HIGHLIGHT_COLOR,
            lw=LINEWIDTH,
            zorder=3,
        )
        ax.plot(
            local_df["time"],
            2 * local_df["local_sigma_clean"],
            color=THESIS_HIGHLIGHT_COLOR,
            lw=LINEWIDTH * 0.8,
            alpha=0.65,
            linestyle="--",
            zorder=3,
        )
    ax.axvspan(
        t0,
        t1,
        alpha=THESIS_SPAN_ALPHA,
        color=THESIS_FLARE_WINDOW_COLOR,
        linewidth=0,
        zorder=0,
    )

    boundaries = [start] + [int(v) + 1 for v in split_valleys] + [end + 1]
    for sub_id, (left, right_exclusive) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        right = right_exclusive - 1
        ax.axvspan(
            ts_df["time"].iloc[left],
            ts_df["time"].iloc[right],
            alpha=0.11,
            color=THESIS_FLARE_WINDOW_COLOR,
            linewidth=0,
            zorder=1,
        )

    for i, p in enumerate(ma10_peaks_global):
        pos = int(p)
        ax.axvline(ts_df["time"].iloc[pos], color=THESIS_HIGHLIGHT_COLOR, linestyle="--", lw=LINEWIDTH * 0.8, alpha=0.9)
        ax.scatter(
            ts_df["time"].iloc[pos],
            ts_df[config.multi_peak_detection_column].iloc[pos],
            s=95 * SCALE,
            color=THESIS_HIGHLIGHT_COLOR,
            marker="o",
            zorder=5,
            label="10p Moving Average Peak" if i == 0 else None,
        )
    for i, p in enumerate(datapoint_peaks_global):
        pos = int(p)
        ax.scatter(
            ts_df["time"].iloc[pos],
            ts_df[config.multi_peak_datapoint_column].iloc[pos],
            s=105 * SCALE,
            color=THESIS_HIGHLIGHT_COLOR,
            marker="x",
            linewidths=LINEWIDTH,
            zorder=6
        )

    original_peak = flare.get("peak_pos", np.nan)
    if pd.notna(original_peak):
        original_peak = int(original_peak)
        if 0 <= original_peak < len(ts_df):
            ax.axvline(ts_df["time"].iloc[original_peak], color=THESIS_RAW_COLOR, linestyle="-.", lw=LINEWIDTH * 0.8, alpha=0.8)
            ax.scatter(
                ts_df["time"].iloc[original_peak],
                ts_df[config.multi_peak_datapoint_column].iloc[original_peak],
                s=120 * SCALE,
                color=THESIS_RAW_COLOR,
                marker="*",
                zorder=7,
            )
    for i, v in enumerate(split_valleys):
        ax.axvline(
            ts_df["time"].iloc[int(v)],
            color=THESIS_HIGHLIGHT_COLOR,
            linestyle=":",
            lw=LINEWIDTH * 0.8,
            alpha=0.95
        )

    ymax = np.nanmax(local_df[[config.multi_peak_detection_column, config.multi_peak_datapoint_column]].to_numpy(dtype=float))
    if not np.isfinite(ymax):
        ymax = 200
    ax.set_ylim(-50, max(200, 1.15 * ymax))
    ax.set_xlim(t0 - config.multi_peak_plot_padding_days, t1 + config.multi_peak_plot_padding_days)
    ax.axhline(0, color=THESIS_RAW_COLOR, lw=LINEWIDTH * 0.8, alpha=0.55)
    ax.set_xlabel("Time BJTD [days]")
    ax.set_ylabel("Flux Residual [e⁻ s⁻¹]")
    _style_axis(ax, panel_label="A")

    path = None
    if results_dir is not None:
        safe_prefix = _safe_filename_token(prefix or "flare_finder")
        path = Path(results_dir) / f"{safe_prefix}_multi_peak_parent_{parent_idx:04d}.pdf"
    return _finish_thesis_figure(fig, ax, path=path, show=show, legend_loc="upper right")


def split_multi_peak_flares(
    ts_df: pd.DataFrame,
    flares_df: pd.DataFrame,
    config: Optional[FlareFinderConfig] = None,
    make_plots: bool = False,
    plot_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split complex flare windows into multiple sub-flares.

    Sub-peaks are detected on ``config.multi_peak_detection_column`` and then mapped
    back to the nearest high point in ``config.multi_peak_datapoint_column``.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    flares_df : pandas.DataFrame
        Flare catalog to split.
    config : FlareFinderConfig or None, optional
        Pipeline settings.
    make_plots : bool, optional
        Save thesis-style PDF plots for split candidates when True.
    plot_dir : str, pathlib.Path, or None, optional
        Directory where multi-peak candidate PDFs are saved.
    prefix : str or None, optional
        Filename prefix for saved PDFs.

    Returns
    -------
    tuple of pandas.DataFrame
        Final split catalog and split-summary table.

    Raises
    ------
    KeyError
        If required time-series columns are missing.
    """

    config = config or FlareFinderConfig()
    if flares_df is None or flares_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    _require_columns(ts_df, ["time", config.multi_peak_detection_column, config.multi_peak_datapoint_column], "multi-peak splitting")
    sigma_col = "local_sigma_clean" if "local_sigma_clean" in ts_df.columns else "local_sigma"
    _require_columns(ts_df, [sigma_col], "multi-peak splitting")

    split_rows = []
    records = []
    n_plotted = 0
    before = flares_df.copy()

    for parent_row_idx in range(len(before)):
        flare = before.iloc[parent_row_idx]
        start = int(flare["new_start_pos"])
        end = int(flare["new_end_pos"])
        ma10_peaks, datapoint_peaks, split_valleys = _find_multi_peak_split_points(ts_df, flare, config, sigma_col=sigma_col)
        n_subflares = len(split_valleys) + 1

        if split_valleys:
            record = {
                "parent_row_idx": int(parent_row_idx),
                "parent_unique_flare_id": flare.get("unique_flare_id", np.nan),
                "parent_new_start_pos": start,
                "parent_new_end_pos": end,
                "parent_new_start_time": flare["new_start_time"],
                "parent_new_end_time": flare["new_end_time"],
                "original_catalog_peak_pos": flare.get("peak_pos", np.nan),
                "original_catalog_t_peak": flare.get("t_peak", np.nan),
                "original_catalog_peak_final_residual": flare.get("peak_final_residual", np.nan),
                "n_candidate_ma10_peaks": int(len(ma10_peaks)),
                "n_subflares": int(n_subflares),
                "ma10_peak_positions": list(map(int, ma10_peaks)),
                "corresponding_datapoint_peak_positions": list(map(int, datapoint_peaks)),
                "split_valley_positions": list(map(int, split_valleys)),
                "ma10_peak_times": [ts_df["time"].iloc[int(p)] for p in ma10_peaks],
                "corresponding_datapoint_peak_times": [ts_df["time"].iloc[int(p)] for p in datapoint_peaks],
                "split_valley_times": [ts_df["time"].iloc[int(v)] for v in split_valleys],
                "plot_path": None,
            }
            if make_plots and plot_dir is not None and (config.multi_peak_max_plots is None or n_plotted < config.multi_peak_max_plots):
                path = plot_multi_peak_candidate(
                    ts_df,
                    flare,
                    ma10_peaks,
                    datapoint_peaks,
                    split_valleys,
                    parent_row_idx,
                    config,
                    results_dir=plot_dir,
                    prefix=prefix,
                    show=False,
                )
                record["plot_path"] = str(path) if path is not None else None
                n_plotted += 1
            records.append(record)

        boundaries = [start] + [int(v) + 1 for v in split_valleys] + [end + 1]
        for sub_id, (left, right_exclusive) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            right = right_exclusive - 1
            if right - left + 1 < config.multi_peak_min_segment_points:
                continue
            candidate_peaks = [int(p) for p in ma10_peaks if left <= int(p) <= right]
            segment_peak = candidate_peaks[0] if candidate_peaks else _best_ma10_peak_in_window(ts_df, left, right, config)
            split_row = _recompute_flare_row_from_window(
                ts_df,
                flare,
                left,
                right,
                config,
                ma10_peak_pos=segment_peak,
                subflare_id=sub_id,
                n_subflares=n_subflares,
            )
            if split_row is not None:
                split_rows.append(split_row)

    final_df = pd.DataFrame(split_rows).sort_values("new_start_pos").reset_index(drop=True) if split_rows else pd.DataFrame()
    if not final_df.empty:
        final_df["unique_flare_id"] = np.arange(len(final_df), dtype=int)
    return final_df, pd.DataFrame(records)


# HEREE!!!!!!!
def plot_sigma_comparison(
    ts_df: pd.DataFrame,
    strong_flares_df: pd.DataFrame,
    max_plots: Optional[int] = None,
    results_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    show: bool = False,
) -> list[Path]:
    """Plot old and cleaned local sigma near strong flares in thesis style.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table containing old and cleaned sigma columns.
    strong_flares_df : pandas.DataFrame
        Strong first-pass flares that were masked for the second-pass noise
        estimate.
    max_plots : int or None, optional
        Maximum number of plots to produce.
    results_dir : str, pathlib.Path, or None, optional
        Directory where individual PDFs are saved.
    prefix : str or None, optional
        Filename prefix for saved PDFs.
    show : bool, optional
        Show figures interactively when no ``results_dir`` is supplied.

    Returns
    -------
    list of pathlib.Path
        Saved PDF paths.
    """

    if strong_flares_df is None or strong_flares_df.empty:
        return []
    _require_columns(ts_df, ["time", "ma_10", "final_residual", "local_sigma", "local_sigma_clean"], "sigma-comparison plotting")

    saved_paths: list[Path] = []
    safe_prefix = _safe_filename_token(prefix or "flare_finder")
    for flare_idx, (_, flare) in enumerate(strong_flares_df.iterrows()):
        if max_plots is not None and flare_idx >= max_plots:
            break

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(ts_df["time"], ts_df["ma_10"], color=THESIS_MAIN_COLOR, lw=LINEWIDTH, label="ma_10", zorder=4)
        ax.scatter(
            ts_df["time"],
            ts_df["final_residual"],
            s=18 * SCALE,
            alpha=THESIS_RAW_ALPHA,
            color=THESIS_RAW_COLOR,
            edgecolors="none",
            label="final_residual",
            zorder=2,
        )
        ax.plot(ts_df["time"], 2 * ts_df["local_sigma"], color=THESIS_RAW_COLOR, lw=LINEWIDTH * 0.8, alpha=0.65, linestyle="--", label="old 2σ", zorder=3)
        ax.plot(ts_df["time"], 2 * ts_df["local_sigma_clean"], color=THESIS_HIGHLIGHT_COLOR, lw=LINEWIDTH, label="cleaned 2σ", zorder=3)
        ax.axvspan(flare["new_start_time"], flare["new_end_time"], alpha=THESIS_SPAN_ALPHA, color=THESIS_FLARE_WINDOW_COLOR, linewidth=0, zorder=0)
        ax.set_xlim(flare["new_start_time"] - 0.5, flare["new_end_time"] + 0.5)
        ax.axhline(0, color=THESIS_RAW_COLOR, lw=LINEWIDTH * 0.8, alpha=0.55)
        ax.set_ylim(-50, max(200, 1.1 * flare["peak_final_residual"]))
        ax.set_xlabel("Time BJTD [days]")
        ax.set_ylabel("Flux Residual [e⁻ s⁻¹]")
        _style_axis(ax, panel_label="A")

        path = Path(results_dir) / f"{safe_prefix}_sigma_comparison_{flare_idx:04d}.pdf" if results_dir is not None else None
        saved = _finish_thesis_figure(fig, ax, path=path, show=show, legend_loc="upper right")
        if saved is not None:
            saved_paths.append(saved)
    return saved_paths


def plot_final_flares(
    ts_df: pd.DataFrame,
    flares_df: pd.DataFrame,
    max_plots: Optional[int] = None,
    results_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    show: bool = False,
) -> list[Path]:
    """Plot final flare windows one by one in thesis style.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Time-series table.
    flares_df : pandas.DataFrame
        Final flare catalog.
    max_plots : int or None, optional
        Maximum number of plots to save or show.
    results_dir : str, pathlib.Path, or None, optional
        Directory where individual PDFs are saved.
    prefix : str or None, optional
        Filename prefix for saved PDFs.
    show : bool, optional
        Show figures interactively when no ``results_dir`` is supplied.

    Returns
    -------
    list of pathlib.Path
        Saved PDF paths.
    """

    if flares_df is None or flares_df.empty:
        return []
    _require_columns(ts_df, ["time", "ma_10", "final_residual"], "final-flare plotting")
    sigma_col = "local_sigma_clean" if "local_sigma_clean" in ts_df.columns else "local_sigma"
    _require_columns(ts_df, [sigma_col], "final-flare plotting")

    saved_paths: list[Path] = []
    safe_prefix = _safe_filename_token(prefix or "flare_finder")
    for flare_idx, (_, flare) in enumerate(flares_df.iterrows()):
        if max_plots is not None and flare_idx >= max_plots:
            break

        fig, ax = plt.subplots(figsize=(12, 7))
        t_min = flare["new_start_time"] - 0.5
        t_max = flare["new_end_time"] + 0.5
        plot_mask = (ts_df["time"] >= t_min) & (ts_df["time"] <= t_max)
        local_df = ts_df.loc[plot_mask]
        if local_df.empty:
            plt.close(fig)
            continue

        ax.plot(local_df["time"], local_df["ma_10"], color=THESIS_MAIN_COLOR, lw=LINEWIDTH, zorder=4)
        ax.plot(local_df["time"], 3 * local_df[sigma_col], color=THESIS_HIGHLIGHT_COLOR, lw=LINEWIDTH, label="3σ", zorder=3)
        ax.plot(local_df["time"], 2 * local_df[sigma_col], color=THESIS_HIGHLIGHT_COLOR, lw=LINEWIDTH * 0.8, alpha=0.7, linestyle="--", zorder=3)
        ax.plot(local_df["time"], local_df[sigma_col], color=THESIS_RAW_COLOR, lw=LINEWIDTH * 0.65, alpha=0.7, linestyle=":", zorder=3)
        ax.scatter(
            local_df["time"],
            local_df["final_residual"],
            s=20 * SCALE,
            alpha=THESIS_RAW_ALPHA,
            color=THESIS_RAW_COLOR,
            edgecolors="none",
            label="final_residual",
            zorder=2,
        )
        ax.axvspan(
            flare["new_start_time"],
            flare["new_end_time"],
            alpha=THESIS_SPAN_ALPHA,
            color=THESIS_FLARE_WINDOW_COLOR,
            linewidth=0,
            label="flare window",
            zorder=0,
        )
        if pd.notna(flare.get("t_peak", np.nan)):
            ax.axvline(flare["t_peak"], color=THESIS_HIGHLIGHT_COLOR, lw=LINEWIDTH * 0.8, linestyle="--", label="peak", zorder=5)
        ax.set_xlim(t_min, t_max)
        ax.axhline(0, color=THESIS_RAW_COLOR, lw=LINEWIDTH * 0.8, alpha=0.55)
        ax.set_ylim(-50, max(200, 1.1 * flare["peak_final_residual"]))
        ax.set_xlabel("Time BJTD [days]")
        ax.set_ylabel("Flux Residual [e⁻ s⁻¹]")
        _style_axis(ax, panel_label="A")

        source = _safe_filename_token(flare.get("source_passes", "final"))
        path = Path(results_dir) / f"{safe_prefix}_final_flare_{flare_idx:04d}_{source}.pdf" if results_dir is not None else None
        saved = _finish_thesis_figure(fig, ax, path=path, show=show, legend_loc="upper right")
        if saved is not None:
            saved_paths.append(saved)
    return saved_paths


def plot_duration_distribution(
    flares_df: pd.DataFrame,
    bins: int = 50,
    results_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the flare-duration distribution in thesis style.

    Parameters
    ----------
    flares_df : pandas.DataFrame
        Flare catalog containing a ``duration`` column in days.
    bins : int, optional
        Number of histogram bins.
    results_dir : str, pathlib.Path, or None, optional
        Directory where the PDF is saved.
    prefix : str or None, optional
        Filename prefix for the saved PDF.
    show : bool, optional
        Show the figure interactively when no ``results_dir`` is supplied.

    Returns
    -------
    pathlib.Path or None
        Saved PDF path, or None when no figure was saved.
    """

    if flares_df is None or flares_df.empty:
        return None
    _require_columns(flares_df, ["duration"], "duration-distribution plotting")

    duration_hours = flares_df["duration"].to_numpy(dtype=float) * 24
    duration_hours = duration_hours[np.isfinite(duration_hours) & (duration_hours >= 0)]
    if duration_hours.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(
        duration_hours,
        bins=bins,
        density=True,
        color=THESIS_RAW_COLOR,
        alpha=THESIS_RAW_ALPHA,
        edgecolor=THESIS_RAW_COLOR,
        linewidth=LINEWIDTH * 0.6,
    )
    ax.set_xlabel("Duration [hours]")
    ax.set_ylabel("Density")
    _style_axis(ax, panel_label="A")

    safe_prefix = _safe_filename_token(prefix or "flare_finder")
    path = Path(results_dir) / f"{safe_prefix}_duration_distribution.pdf" if results_dir is not None else None
    return _finish_thesis_figure(fig, ax, path=path, show=show, legend_loc=None)


def plot_peak_height_distribution(
    flares_df: pd.DataFrame,
    bins: int = 100,
    results_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the flare peak-height distribution in thesis style.

    Parameters
    ----------
    flares_df : pandas.DataFrame
        Flare catalog containing ``peak_final_residual``.
    bins : int, optional
        Number of histogram bins.
    results_dir : str, pathlib.Path, or None, optional
        Directory where the PDF is saved.
    prefix : str or None, optional
        Filename prefix for the saved PDF.
    show : bool, optional
        Show the figure interactively when no ``results_dir`` is supplied.

    Returns
    -------
    pathlib.Path or None
        Saved PDF path, or None when no figure was saved.
    """

    if flares_df is None or flares_df.empty:
        return None
    _require_columns(flares_df, ["peak_final_residual"], "peak-height-distribution plotting")

    peaks = flares_df["peak_final_residual"].to_numpy(dtype=float)
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
    if peaks.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(
        peaks,
        bins=bins,
        density=True,
        color=THESIS_RAW_COLOR,
        alpha=THESIS_RAW_ALPHA,
        edgecolor=THESIS_RAW_COLOR,
        linewidth=LINEWIDTH * 0.6,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peak Height [e⁻ s⁻¹]")
    ax.set_ylabel("Density")
    _style_axis(ax, panel_label="A", format_x=False, format_y=False)

    safe_prefix = _safe_filename_token(prefix or "flare_finder")
    path = Path(results_dir) / f"{safe_prefix}_peak_height_distribution.pdf" if results_dir is not None else None
    return _finish_thesis_figure(fig, ax, path=path, show=show, legend_loc=None)


def save_flare_outputs(
    results: Dict[str, Any],
    out_dir: str | Path,
    tic_id: Optional[int | str] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Path]:
    """Save the main output tables to CSV files.

    Parameters
    ----------
    results : dict
        Dictionary returned by :func:`run_two_pass_flare_finder`.
    out_dir : str or pathlib.Path
        Folder where CSV files are written.
    tic_id : int, str, or None, optional
        Optional TIC identifier used in default filenames.
    prefix : str or None, optional
        Optional filename prefix. If omitted, ``TIC_<tic_id>`` is used when a TIC ID
        is supplied, otherwise ``flare_finder`` is used.

    Returns
    -------
    dict
        Mapping from result name to the path that was saved.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"TIC_{tic_id}" if tic_id is not None else "flare_finder"

    paths = {
        "ts_df_second_pass": out_path / f"{prefix}_detrended_second_pass_clean_sigma.csv",
        "final_flares_df": out_path / f"{prefix}_unique_union_flares.csv",
        "all_pass_detections_df": out_path / f"{prefix}_all_pass_flare_detections_before_dedup.csv",
        "comparison_summary": out_path / f"{prefix}_comparison_summary.csv",
        "multi_peak_split_summary_df": out_path / f"{prefix}_multi_peak_split_summary.csv",
    }
    for key, path in paths.items():
        value = results.get(key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(path, index=False)
    return paths


# HEREE!!!!!!!
def run_two_pass_flare_finder(
    ts_df: pd.DataFrame,
    config: Optional[FlareFinderConfig] = None,
    make_plots: bool = False,
    save_outputs: bool = False,
    out_dir: Optional[str | Path] = None,
    tic_id: Optional[int | str] = None,
    verbose: bool = True,
    plot_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the full two-pass flare-finding pipeline.

    This is the main notebook entry point. It runs pass 1 with ``local_sigma``,
    recomputes a cleaned local-noise estimate after masking strong pass-1 flares,
    runs pass 2 with ``local_sigma_clean``, merges both passes into a unique union
    catalog, applies optional post-filters and optional multi-peak splitting, marks
    useful masks on the output time series, optionally saves thesis-style PDF
    plots for the final flares and multi-peak split candidates, and optionally
    saves CSV files.

    Parameters
    ----------
    ts_df : pandas.DataFrame
        Input time-series table. It is copied before new columns are added.
    config : FlareFinderConfig or None, optional
        Pipeline settings.
    make_plots : bool, optional
        When True, save thesis-style PDF plots for the final accepted flares and
        the multi-peak split candidates.
    save_outputs : bool, optional
        Save the main output DataFrames to CSV files when True.
    out_dir : str, pathlib.Path, or None, optional
        Output folder used when ``save_outputs`` is True. When ``make_plots`` is
        True and ``plot_dir`` is not supplied, figures are saved in
        ``out_dir / "Figures"``.
    tic_id : int, str, or None, optional
        Optional TIC identifier used in output and plot filenames.
    verbose : bool, optional
        Print progress summaries.
    plot_dir : str, pathlib.Path, or None, optional
        Explicit output folder for individual PDF figures. If omitted while
        ``make_plots`` is True and ``out_dir`` is also omitted, figures are saved
        to ``flare_finder_figures`` in the current working directory.

    Returns
    -------
    dict
        Main output DataFrames and metadata from the two-pass pipeline.

    Raises
    ------
    KeyError
        If required input columns are missing.
    ValueError
        If ``save_outputs`` is True and ``out_dir`` is not provided.
    """

    config = config or FlareFinderConfig()
    _require_columns(ts_df, ["time", "final_residual", "ma_10", "local_sigma"], "two-pass flare finder")
    ts_df_input = ts_df.copy()

    plot_output_dir = _resolve_plot_dir(plot_dir=plot_dir, out_dir=out_dir) if make_plots else None
    plot_prefix = _default_plot_prefix(tic_id=tic_id)
    plot_paths: list[str] = []

    first_pass = run_flare_finder(ts_df_input, config=config, sigma_col="local_sigma", label="first pass", verbose=verbose)
    first_pass_final = first_pass["final_flares_df"]

    ts_df_second_pass, strong_first_pass_flares_df, clean_noise_window_points = add_clean_local_sigma(
        ts_df_input,
        first_pass_final,
        config=config,
        verbose=verbose,
    )

    second_pass = run_flare_finder(ts_df_second_pass, config=config, sigma_col="local_sigma_clean", label="second pass", verbose=verbose)
    second_pass_final = second_pass["final_flares_df"]

    unique_flares_df, all_pass_detections_df = make_unique_flare_catalog(
        ts_df_second_pass,
        first_pass_final,
        second_pass_final,
        merge_gap_points=config.unique_flare_merge_gap_points,
    )
    final_flares_df_before_ma10_residual_filter = unique_flares_df.copy()
    removed_by_ma10_residual_filter_df = pd.DataFrame()

    if config.apply_ma10_residual_filter:
        final_flares_df, removed_by_ma10_residual_filter_df = apply_ma10_residual_filter(
            ts_df_second_pass,
            unique_flares_df,
            sigma_col="local_sigma_clean",
            ma10_filter_sigma=config.ma10_filter_sigma,
            residual_filter_sigma=config.residual_filter_sigma,
        )
        if verbose:
            n_before = len(final_flares_df_before_ma10_residual_filter)
            n_after = len(final_flares_df)
            print(f"Flares before ma_10 / residual filter: {n_before:,}")
            print(f"Flares after ma_10 / residual filter:  {n_after:,}")
            print(f"Removed:                               {n_before - n_after:,}")
    else:
        final_flares_df = unique_flares_df.copy()

    final_flares_df_before_multi_peak_split = final_flares_df.copy()
    multi_peak_split_summary_df = pd.DataFrame()
    if config.split_multi_peak_flares:
        final_flares_df, multi_peak_split_summary_df = split_multi_peak_flares(
            ts_df_second_pass,
            final_flares_df,
            config=config,
            make_plots=make_plots,
            plot_dir=plot_output_dir,
            prefix=plot_prefix,
        )
        if "plot_path" in multi_peak_split_summary_df.columns:
            plot_paths.extend([
                str(path)
                for path in multi_peak_split_summary_df["plot_path"].dropna().tolist()
                if str(path)
            ])
        if verbose:
            n_before = len(final_flares_df_before_multi_peak_split)
            n_after = len(final_flares_df)
            print(f"Flares before multi-peak split: {n_before:,}")
            print(f"Flares after multi-peak split:  {n_after:,}")
            print(f"Multi-peak parent flares split: {len(multi_peak_split_summary_df):,}")

    ts_df_second_pass = mark_flare_mask(ts_df_second_pass, first_pass_final, "final_flare_mask_first_pass_recomputed")
    ts_df_second_pass = mark_flare_mask(ts_df_second_pass, second_pass_final, "final_flare_mask_second_pass")
    ts_df_second_pass = mark_flare_mask(ts_df_second_pass, unique_flares_df, "final_flare_mask_unique_union")
    ts_df_second_pass = mark_flare_mask(ts_df_second_pass, final_flares_df, "final_flare_mask_unique_union_split")

    comparison_summary = pd.DataFrame([
        summarize_flares("first_pass_original_sigma", first_pass_final),
        summarize_flares("second_pass_clean_sigma", second_pass_final),
        summarize_flares("final_unique_union", unique_flares_df),
        summarize_flares("final_after_filters_and_splitting", final_flares_df),
    ])

    if make_plots:
        final_flare_plot_paths = plot_final_flares(
            ts_df_second_pass,
            final_flares_df,
            max_plots=config.multi_peak_max_plots,
            results_dir=plot_output_dir,
            prefix=plot_prefix,
            show=False,
        )
        plot_paths.extend(str(path) for path in final_flare_plot_paths)

    results: Dict[str, Any] = {
        "config": asdict(config),
        "ts_df_second_pass": ts_df_second_pass,
        "first_pass": first_pass,
        "second_pass": second_pass,
        "first_pass_flares_df": first_pass_final,
        "second_pass_flares_df": second_pass_final,
        "unique_flares_df_before_post_filters": unique_flares_df,
        "final_flares_df_before_ma10_residual_filter": final_flares_df_before_ma10_residual_filter,
        "removed_by_ma10_residual_filter_df": removed_by_ma10_residual_filter_df,
        "final_flares_df_before_multi_peak_split": final_flares_df_before_multi_peak_split,
        "final_flares_df": final_flares_df,
        "all_pass_detections_df": all_pass_detections_df,
        "comparison_summary": comparison_summary,
        "strong_first_pass_flares_df": strong_first_pass_flares_df,
        "multi_peak_split_summary_df": multi_peak_split_summary_df,
        "clean_noise_window_points": clean_noise_window_points,
        "plot_paths": plot_paths,
    }

    if save_outputs:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save_outputs=True.")
        results["saved_paths"] = save_flare_outputs(results, out_dir=out_dir, tic_id=tic_id)

    return results
