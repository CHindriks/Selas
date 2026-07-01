"""
Reusable SELAS pipeline runner.

This module was refactored from `Full Runner(3).ipynb` so the full
workflow can be imported and run for one TIC ID or a list of TIC IDs.

Typical notebook usage:

    from pathlib import Path
    from selas_runner import *

    result = run_single_tic_pipeline(
        tic_id="290716988",
        data_root=Path("../Data/Selas-TIC-ids"),
        selas_path=Path("."),
        make_plots=False,
        raise_on_error=True,
    )
"""

from __future__ import annotations

import ast
import logging
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterable

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
from matplotlib import font_manager as fm
from matplotlib.ticker import FixedLocator, FuncFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde, linregress


# These are imported lazily after `configure_environment(...)` adds the SELAS
# source folder to sys.path. This makes the module easier to import from a
# notebook, even if the notebook is not inside the SELAS package folder.
DetrendConfig = None
get_timeseries = None
detrend_dataframe = None
save_result = None
print_summary = None
plot_trends = None
plot_residuals = None
plot_selected_windows = None
plot_segment_quality = None

FlareFinderConfig = None
load_time_series = None
run_two_pass_flare_finder = None
save_flare_outputs = None
plot_final_flares = None
plot_duration_distribution = None
plot_peak_height_distribution = None

AnalysisConfig = None
run_waiting_time_statistics = None

PeriodicityConfig = None
run_periodicity_workflow = None
run_jackknife_if_rayleigh_exceedance = None


class Tee:
    """Write stdout/stderr to multiple file-like objects at the same time."""

    def __init__(self, *files: Any):
        self.files = files

    def write(self, data: str) -> None:
        for file_obj in self.files:
            file_obj.write(data)
            file_obj.flush()

    def flush(self) -> None:
        for file_obj in self.files:
            file_obj.flush()


def configure_environment(
    selas_path: str | Path = "../Selas",
    font_path: str | Path = "/usr/share/fonts/dejavu-serif-fonts/DejaVuSerifCondensed.ttf",
    suppress_warnings: bool = True,
) -> Path:
    """
    Configure warnings, plotting defaults, pandas display, and SELAS imports.

    Parameters
    ----------
    selas_path:
        Folder containing the SELAS source files, such as
        `lightcurve_detrender.py`, `flare_finder.py`,
        `waiting_time_statistics.py`, and `periodicity_statistics.py`.
    font_path:
        Optional font path used for Matplotlib.
    suppress_warnings:
        Suppress common RuntimeWarning/FutureWarning messages.

    Returns
    -------
    Path
        The resolved SELAS path.
    """
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    logging.getLogger("lightkurve").setLevel(logging.ERROR)

    resolved_selas_path = Path(selas_path).expanduser().resolve()
    if str(resolved_selas_path) not in sys.path:
        sys.path.insert(0, str(resolved_selas_path))

    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", None)

    resolved_font_path = Path(font_path).expanduser()
    if resolved_font_path.exists():
        custom_font = fm.FontProperties(fname=str(resolved_font_path))
        plt.rcParams["font.family"] = custom_font.get_name()

    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.size": 12,
            "figure.dpi": 150,
        }
    )

    return resolved_selas_path


def import_selas_components(selas_path: str | Path | None = None) -> None:
    """
    Import SELAS workflow components into this module's global namespace.

    Call this explicitly only if you need the individual imported objects.
    `run_single_tic_pipeline(...)` calls it automatically.
    """
    global DetrendConfig, get_timeseries, detrend_dataframe, save_result
    global print_summary, plot_trends, plot_residuals, plot_selected_windows
    global plot_segment_quality
    global FlareFinderConfig, load_time_series, run_two_pass_flare_finder
    global save_flare_outputs, plot_final_flares, plot_duration_distribution
    global plot_peak_height_distribution
    global AnalysisConfig, run_waiting_time_statistics
    global PeriodicityConfig, run_periodicity_workflow
    global run_jackknife_if_rayleigh_exceedance

    if selas_path is not None:
        configure_environment(selas_path=selas_path)

    from lightcurve_detrender import (  # pylint: disable=import-error,import-outside-toplevel
        DetrendConfig as _DetrendConfig,
        detrend_dataframe as _detrend_dataframe,
        get_timeseries as _get_timeseries,
        plot_residuals as _plot_residuals,
        plot_segment_quality as _plot_segment_quality,
        plot_selected_windows as _plot_selected_windows,
        plot_trends as _plot_trends,
        print_summary as _print_summary,
        save_result as _save_result,
    )
    from flare_finder import (  # pylint: disable=import-error,import-outside-toplevel
        FlareFinderConfig as _FlareFinderConfig,
        load_time_series as _load_time_series,
        plot_duration_distribution as _plot_duration_distribution,
        plot_final_flares as _plot_final_flares,
        plot_peak_height_distribution as _plot_peak_height_distribution,
        run_two_pass_flare_finder as _run_two_pass_flare_finder,
        save_flare_outputs as _save_flare_outputs,
    )
    from periodicity_statistics import (  # pylint: disable=import-error,import-outside-toplevel
        PeriodicityConfig as _PeriodicityConfig,
        run_periodicity_workflow as _run_periodicity_workflow,
    )
    from waiting_time_statistics import (  # pylint: disable=import-error,import-outside-toplevel
        AnalysisConfig as _AnalysisConfig,
        run_waiting_time_statistics as _run_waiting_time_statistics,
    )
    from jackknife_test import (  # pylint: disable=import-error,import-outside-toplevel
        run_jackknife_if_rayleigh_exceedance as _run_jackknife_if_rayleigh_exceedance,
    )

    DetrendConfig = _DetrendConfig
    get_timeseries = _get_timeseries
    detrend_dataframe = _detrend_dataframe
    save_result = _save_result
    print_summary = _print_summary
    plot_trends = _plot_trends
    plot_residuals = _plot_residuals
    plot_selected_windows = _plot_selected_windows
    plot_segment_quality = _plot_segment_quality

    FlareFinderConfig = _FlareFinderConfig
    load_time_series = _load_time_series
    run_two_pass_flare_finder = _run_two_pass_flare_finder
    save_flare_outputs = _save_flare_outputs
    plot_final_flares = _plot_final_flares
    plot_duration_distribution = _plot_duration_distribution
    plot_peak_height_distribution = _plot_peak_height_distribution

    AnalysisConfig = _AnalysisConfig
    run_waiting_time_statistics = _run_waiting_time_statistics

    PeriodicityConfig = _PeriodicityConfig
    run_periodicity_workflow = _run_periodicity_workflow
    run_jackknife_if_rayleigh_exceedance = _run_jackknife_if_rayleigh_exceedance


def _display_dataframe(df: pd.DataFrame, enabled: bool = True) -> None:
    """Display a dataframe in notebooks and fall back to print elsewhere."""
    if not enabled:
        return

    try:
        from IPython.display import display  # pylint: disable=import-outside-toplevel

        display(df)
    except Exception:
        print(df)


def _load_full_pipeline_plot_data(
    tic_id: str,
    base_path: Path,
    waiting_time_summary: dict[str, Any],
    rayleigh_result: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = base_path / "Data"
    flare_path = data_dir / f"{tic_id}_flares.csv"
    if not flare_path.exists():
        flare_path = data_dir / f"{tic_id}_combined_final_flares.csv"

    if not flare_path.exists():
        raise FileNotFoundError(
            f"Cannot find flare file for TIC {tic_id}: {flare_path}"
        )

    flare_df = pd.read_csv(flare_path)

    pdf_path = waiting_time_summary.get("pdf_data_path")
    cdf_path = waiting_time_summary.get("cdf_data_path")
    if pdf_path is None or cdf_path is None:
        raise ValueError(
            "Waiting-time summary must provide pdf_data_path and cdf_data_path."
        )

    pdf_df = pd.read_csv(pdf_path)
    cdf_df = pd.read_csv(cdf_path)

    summary_df = rayleigh_result.get("summary")
    if not isinstance(summary_df, pd.DataFrame):
        raise ValueError("Rayleigh result is missing a valid summary dataframe.")

    rayleigh = rayleigh_result.get("rayleigh")
    if rayleigh is None or not hasattr(rayleigh, "periods") or not hasattr(rayleigh, "p_values"):
        raise ValueError("Rayleigh result is missing period search output.")

    rt_df = pd.DataFrame(
        {
            "T": np.asarray(rayleigh.periods, dtype=float),
            "p_values": np.asarray(rayleigh.p_values, dtype=float),
        }
    )

    return flare_df, summary_df, pdf_df, cdf_df, rt_df


def _align_best_flare_phases(flare_df: pd.DataFrame, best_period: float) -> pd.DataFrame:
    folded = (flare_df["new_start_time"].to_numpy(dtype=float) / best_period) % 1
    offset_grid = np.linspace(0, 1, 10000, endpoint=False)
    counts = []

    for test_offset in offset_grid:
        shifted_phase = ((folded - test_offset + 0.5) % 1) - 0.5
        n_inside = np.sum((shifted_phase >= -0.25) & (shifted_phase <= 0.25))
        counts.append(n_inside)

    counts = np.array(counts, dtype=int)
    best_offset = float(offset_grid[np.argmax(counts)])
    folded_shifted = ((folded - best_offset + 0.5) % 1) - 0.5

    flare_stacked = flare_df.copy()
    flare_stacked["phase"] = folded_shifted
    return flare_stacked


def _make_full_pipeline_plot(
    tic_id: str,
    base_path: Path,
    flare_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rt_df: pd.DataFrame,
    pdf_df: pd.DataFrame,
    cdf_df: pd.DataFrame,
    jackknife_df: pd.DataFrame,
    output_path: Path,
    show: bool = False,
) -> Path:
    best_period = float(summary_df["RT_best_period"].iloc[0])
    flare_stacked = _align_best_flare_phases(flare_df, best_period)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    BASE_FONTSIZE = 20
    BASE_LINEWIDTH = 2.2
    SCALE = 1.8

    FONTSIZE = BASE_FONTSIZE * SCALE
    LINEWIDTH = BASE_LINEWIDTH * SCALE

    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "axes.linewidth": LINEWIDTH,
        }
    )

    def plain_scalar_formatter():
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        return formatter

    def integer_log_tick_formatter(x, pos):
        if np.isclose(x, np.arange(1, 13)).any():
            return f"{int(round(x))}"
        return ""

    fig, axes = plt.subplots(3, 2, figsize=(24, 21))
    ax_pdf, ax_cdf = axes[0]
    ax_rt, ax_phase = axes[1]
    ax_jk, ax_sigma = axes[2]

    pdf_mask = pdf_df["waiting_time_days"] >= 0.1
    ax_pdf.plot(
        pdf_df.loc[pdf_mask, "waiting_time_days"],
        pdf_df.loc[pdf_mask, "pdf_empirical_density"],
        color="black",
        alpha=0.95,
        lw=LINEWIDTH,
        zorder=5,
    )
    ax_pdf.plot(
        pdf_df.loc[pdf_mask, "waiting_time_days"],
        pdf_df.loc[pdf_mask, "pdf_theory_density"],
        color="crimson",
        alpha=0.85,
        lw=LINEWIDTH,
    )
    ax_pdf.set_xlim(0, 24)
    ax_pdf.set_xticks([0, 5, 10, 15, 20])
    ax_pdf.set_xlabel("Waiting time [days]")
    ax_pdf.set_ylabel("Probability density")
    ax_pdf.xaxis.set_major_formatter(plain_scalar_formatter())
    pdf_ymax = np.nanmax(
        [
            pdf_df.loc[pdf_mask, "pdf_empirical_density"].max() if pdf_mask.any() else 0,
            pdf_df.loc[pdf_mask, "pdf_theory_density"].max() if pdf_mask.any() else 0,
        ]
    )
    ax_pdf.set_ylim(0, 1.12 * max(pdf_ymax, 1.0))

    ax_cdf.plot(
        cdf_df["waiting_time_days"],
        cdf_df["cdf_empirical_mean"],
        color="black",
        alpha=0.95,
        lw=LINEWIDTH,
        zorder=5,
    )
    ax_cdf.plot(
        cdf_df["waiting_time_days"],
        cdf_df["cdf_theory"],
        color="crimson",
        alpha=0.85,
        lw=LINEWIDTH,
    )
    ax_cdf.set_xlim(0, 24)
    ax_cdf.set_xlabel("Waiting time [days]")
    ax_cdf.set_ylabel("Cumulative density")
    ax_cdf.xaxis.set_major_formatter(plain_scalar_formatter())
    ax_cdf.set_ylim(0, 1.05)

    ax_rt.plot(
        rt_df["T"],
        rt_df["p_values"],
        color="black",
        alpha=0.95,
        lw=LINEWIDTH,
    )
    ax_rt.scatter(
        summary_df["RT_best_period"],
        summary_df["RT_best_p_value"],
        color="crimson",
        alpha=0.85,
        s=80 * SCALE**2,
        zorder=6,
    )
    ax_rt.axhline(
        summary_df["one_detection_three_sigma"].iloc[0],
        color="crimson",
        alpha=0.85,
        lw=LINEWIDTH,
        ls="--",
    )
    ax_rt.set_xlim(1, 12)
    ax_rt.set_xlabel("Trial period [days]")
    ax_rt.set_ylabel("p-value")
    ax_rt.set_xscale("log")
    ax_rt.set_yscale("log")
    rt_xticks = np.arange(1, 13, 2)
    ax_rt.set_xticks(rt_xticks)
    ax_rt.xaxis.set_major_locator(FixedLocator(rt_xticks))
    ax_rt.xaxis.set_major_formatter(FuncFormatter(integer_log_tick_formatter))
    positive_p_values = rt_df["p_values"][rt_df["p_values"] > 0]
    if positive_p_values.size > 0:
        ax_rt.set_ylim(1e-7, 1.0)
    else:
        ax_rt.set_ylim(1e-7, 1.0)

    counts_phase, *_ = ax_phase.hist(
        flare_stacked["phase"],
        color="grey",
        alpha=0.65,
        linewidth=0,
        zorder=5,
    )
    ax_phase.axvspan(-0.25, 0.25, color="lightblue", alpha=0.35, zorder=0)
    ax_phase.axvline(-0.25, color="grey", lw=LINEWIDTH, ls=":")
    ax_phase.axvline(0.25, color="grey", lw=LINEWIDTH, ls=":")
    ax_phase.set_xlim(-0.49, 0.5)
    ax_phase.set_xlabel("Phase [1/days]")
    ax_phase.set_ylabel("Number of flares")
    ax_phase.xaxis.set_major_formatter(plain_scalar_formatter())
    if len(counts_phase) > 0:
        ax_phase.set_ylim(0, 1.15 * counts_phase.max())

    jk_period = jackknife_df["best_period"].to_numpy(dtype=float)
    jk_pvalue = jackknife_df["min_p_value"].to_numpy(dtype=float)
    jk_mask = np.isfinite(jk_period) & np.isfinite(jk_pvalue) & (jk_pvalue > 0)
    jk_period_valid = jk_period[jk_mask]
    jk_pvalue_valid = jk_pvalue[jk_mask]
    ax_jk.scatter(
        jk_period_valid,
        jk_pvalue_valid,
        color="black",
        alpha=0.9,
        s=10 * SCALE**2,
        linewidths=0,
        zorder=5,
    )
    ax_jk.set_xlim(1, 12.5)
    ax_jk.set_ylim(1e-7, 1)
    ax_jk.set_yscale("log")
    ax_jk.set_xscale("log")
    ax_jk.set_xlabel("Best period [days]")
    ax_jk.set_ylabel("Minimum p-value")
    jk_xticks = np.arange(1, 13, 2)
    ax_jk.set_xticks(jk_xticks)
    ax_jk.xaxis.set_major_locator(FixedLocator(jk_xticks))
    ax_jk.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(round(x))}" if np.isclose(x, jk_xticks).any() else "")
    )

    sigma_vals = jackknife_df["RT_exceedance_sigma"].to_numpy(dtype=float)
    sigma_vals = sigma_vals[np.isfinite(sigma_vals)]
    # Replace negative exceedances with zero for plotting and stats.
    sigma_vals = np.clip(sigma_vals, 0.0, None)
    counts_sigma, _, _ = ax_sigma.hist(
        sigma_vals,
        color="grey",
        alpha=0.65,
        linewidth=0,
    )
    if sigma_vals.size > 0 and np.max(sigma_vals) > 0:
        ax_sigma.set_xlim(0, 1.1 * np.max(sigma_vals))
    else:
        ax_sigma.set_xlim(0, 1.0)
    sigma_xticks = np.arange(0, 7, 1)
    ax_sigma.set_xticks(sigma_xticks)
    ax_sigma.set_xlabel(r"RT exceedance [$\sigma$]")
    ax_sigma.set_ylabel("Count")
    ax_sigma.xaxis.set_major_formatter(plain_scalar_formatter())
    if len(counts_sigma) > 0:
        ax_sigma.set_ylim(0, 1.15 * counts_sigma.max())
    # Annotate percentage of trials >3 sigma.
    if sigma_vals.size > 0:
        pct_gt_3 = 100.0 * np.sum(sigma_vals > 3.0) / sigma_vals.size
    else:
        pct_gt_3 = 0.0
    ax_sigma.text(
        0.98,
        0.98,
        f">{pct_gt_3:.1f}% >3σ",
        transform=ax_sigma.transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE * 0.65,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    )

    panel_labels = ["A", "B", "C", "D", "E", "F"]
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            0.03,
            0.93,
            label,
            transform=ax.transAxes,
            fontsize=FONTSIZE * 1.05,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
            zorder=50,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=4),
        )

    for ax in axes.flat:
        ax.tick_params(axis="both", which="major", width=LINEWIDTH, length=7 * SCALE)
        ax.tick_params(axis="both", which="minor", width=LINEWIDTH * 0.8, length=4 * SCALE)
        for spine in ax.spines.values():
            spine.set_linewidth(LINEWIDTH)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.98, wspace=0.24, hspace=0.32)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def ensure_tic_dirs(
    tic_id: str | int,
    data_root: str | Path = "../Data/Selas-TIC-ids",
) -> tuple[Path, Path]:
    """Create and return `(base_path, data_path)` for a TIC run."""
    tic_id = str(tic_id)
    base_path = Path(data_root).expanduser() / tic_id
    data_path = base_path / "Data"
    data_path.mkdir(parents=True, exist_ok=True)
    return base_path, data_path


def one_detection_k_sigma(k: float, N: int) -> float:
    """Compute the one-detection p-value threshold for a k-sigma event."""
    s = (-k + np.sqrt(k**2 + 4)) / 2
    y = s**2
    return y / N


def p_value_to_exceedance_sigma(p: float, N: int) -> float:
    """
    Inverse of one_detection_k_sigma(k, N).

    Converts a p-value into the corresponding k-sigma exceedance level.
    """
    pN = p * N
    pN = np.clip(pN, 1e-300, None)
    return (1 - pN) / np.sqrt(pN)


def add_rayleigh_exceedance_sigma(rayleigh_result: dict[str, Any]) -> dict[str, Any]:
    """Add an RT_exceedance_sigma column to a Rayleigh summary DataFrame."""
    if not isinstance(rayleigh_result, dict):
        raise ValueError("rayleigh_result must be a dict")

    summary = rayleigh_result.get("summary")
    if not isinstance(summary, pd.DataFrame):
        return rayleigh_result

    summary = summary.copy()
    summary["RT_exceedance_sigma"] = np.nan

    mask = (
        (summary["rayleigh_performed"] == True)
        & summary["RT_best_p_value"].notna()
        & summary["gridsize"].notna()
        & (summary["gridsize"] > 0)
    )
    if mask.any():
        summary.loc[mask, "RT_exceedance_sigma"] = p_value_to_exceedance_sigma(
            summary.loc[mask, "RT_best_p_value"],
            summary.loc[mask, "gridsize"],
        )

    updated_result = dict(rayleigh_result)
    updated_result["summary"] = summary
    return updated_result


def find_catalog_file(base_path: str | Path, filename: str) -> Path | None:
    """
    Search `base_path`, its parents, and the current working directory for a file.
    """
    base_path = Path(base_path).expanduser().resolve()

    search_roots: list[Path] = []
    for root in [base_path, *base_path.parents, Path.cwd(), *Path.cwd().parents]:
        if root.exists() and root not in search_roots:
            search_roots.append(root)

    for root in search_roots:
        direct = root / filename
        if direct.exists():
            return direct

        matches = list(root.rglob(filename))
        if matches:
            return matches[0]

    return None


def star_properties(base_path: str | Path, tic_id: str | int) -> pd.DataFrame:
    """Query TIC star properties, save them as CSV, and return the dataframe."""
    tic_id = str(tic_id)
    outdir = Path(base_path).expanduser()
    data_dir = outdir / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tic_data = Catalogs.query_object(f"TIC {tic_id}", radius=0.0001, catalog="TIC")

    if tic_data is None or len(tic_data) == 0:
        raise LookupError(f"No TIC entry found for TIC {tic_id}")

    keep_cols = [
        "ID",
        "ra",
        "dec",
        "GAIA",
        "Teff",
        "e_Teff",
        "logg",
        "e_logg",
        "rad",
        "e_rad",
        "mass",
        "e_mass",
        "lum",
        "e_lum",
        "MH",
        "e_MH",
        "rho",
        "e_rho",
        "lumclass",
        "Tmag",
        "e_Tmag",
        "GAIAmag",
        "gaiabp",
        "gaiarp",
        "Bmag",
        "Vmag",
        "Jmag",
        "Hmag",
        "Kmag",
        "w1mag",
        "w2mag",
        "plx",
        "e_plx",
        "d",
        "e_d",
        "ebv",
        "e_ebv",
        "pmRA",
        "pmDEC",
        "TeffFlag",
        "PARflag",
        "SPFlag",
        "GAIAqflag",
        "objType",
        "wdflag",
    ]

    filtered = tic_data[[c for c in keep_cols if c in tic_data.colnames]]
    df_star = filtered.to_pandas()

    if df_star.empty:
        raise ValueError(f"TIC {tic_id} returned an empty star_properties dataframe")

    outfile = data_dir / f"{tic_id}_star_properties.csv"
    df_star.to_csv(outfile, index=False)

    print(f"Saved {outfile}")
    return df_star


def find_stellar_rotation_periods(all_lightcurves: Iterable[Any]) -> np.ndarray:
    """Calculate a Lomb-Scargle rotation period for each available light curve."""
    min_period = 0.1
    max_period = 15

    periods = []

    for lc in all_lightcurves:
        try:
            pg = lc.to_periodogram(
                method="lombscargle",
                minimum_period=min_period,
                maximum_period=max_period,
            )
            periods.append(pg.period_at_max_power)
        except Exception:
            periods.append(np.nan * u.day)

    periods_clean = u.Quantity(periods).to_value(u.day)
    periods_clean = np.asarray(periods_clean, dtype=float)
    periods_clean = periods_clean[np.isfinite(periods_clean)]

    if len(periods_clean) == 0:
        raise ValueError("Could not calculate any valid stellar rotation periods")

    return periods_clean


def find_best_stellar_rotation_period(stellar_rotation_periods: Iterable[float]) -> float:
    """Estimate the representative rotation period using a KDE peak."""
    stellar_rotation_periods = np.asarray(stellar_rotation_periods, dtype=float)

    if len(stellar_rotation_periods) == 1:
        return float(stellar_rotation_periods[0])

    if np.nanstd(stellar_rotation_periods) == 0:
        return float(stellar_rotation_periods[0])

    kde = gaussian_kde(stellar_rotation_periods)

    x = np.linspace(
        max(0, np.min(stellar_rotation_periods) - 3 * np.std(stellar_rotation_periods)),
        np.max(stellar_rotation_periods) + 3 * np.std(stellar_rotation_periods),
        500,
    )

    y = kde(x)
    return float(x[np.argmax(y)])


def stellar_rotation_period_evolution(
    stellar_rotation_periods: Iterable[float],
    sectors: Iterable[float],
) -> tuple[float, float, float, float]:
    """Fit a linear trend to rotation periods over TESS sectors."""
    stellar_rotation_periods = np.asarray(stellar_rotation_periods, dtype=float)
    sectors = np.asarray(sectors, dtype=float)

    n = min(len(stellar_rotation_periods), len(sectors))
    stellar_rotation_periods = stellar_rotation_periods[:n]
    sectors = sectors[:n]

    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    sectors_days = sectors * 27.4
    result = linregress(sectors_days, stellar_rotation_periods)

    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue**2
    p_value = result.pvalue

    return slope, intercept, r_squared, p_value


def stellar_rotation_calculator(tic_id: str | int, base_path: str | Path) -> pd.DataFrame:
    """Download TESS light curves, estimate rotation periods, save CSV, return dataframe."""
    tic_id = str(tic_id)
    outdir = Path(base_path).expanduser()
    data_dir = outdir / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    download_dir = data_dir / "lightkurve_cache"
    download_dir.mkdir(parents=True, exist_ok=True)

    lc_table = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        author="SPOC",
        exptime=120,
    )

    if len(lc_table) == 0:
        raise LookupError(f"No TESS 120s SPOC light curves found for TIC {tic_id}")

    sectors = np.array([int(str(m).split("Sector")[-1]) for m in lc_table.table["mission"]])

    lcc = lc_table.download_all(download_dir=str(download_dir))

    if lcc is None or len(lcc) == 0:
        raise LookupError(
            f"Lightkurve found entries but could not download light curves for TIC {tic_id}"
        )

    stellar_rotation_periods = find_stellar_rotation_periods(lcc)
    stellar_rotation_period = find_best_stellar_rotation_period(stellar_rotation_periods)
    slope, intercept, r_squared, p_value = stellar_rotation_period_evolution(
        stellar_rotation_periods,
        sectors,
    )

    df_stellar_rotation = pd.DataFrame(
        {
            "TIC_id": [tic_id],
            "n_sectors": [len(sectors)],
            "stellar_rotation_period": [stellar_rotation_period],
            "slope_evolution": [slope],
            "slope_start": [intercept],
            "r_squared": [r_squared],
            "p_value": [p_value],
            "all_stellar_rotation_periods": [list(stellar_rotation_periods)],
        }
    )

    outfile = data_dir / f"{tic_id}_df_stellar_rotation.csv"
    df_stellar_rotation.to_csv(outfile, index=False)

    print(f"Saved {outfile}")
    return df_stellar_rotation


def get_star_type_and_variability(
    tic_id: str | int,
    base_path: str | Path,
    skip_if_catalogs_missing: bool = True,
) -> pd.DataFrame:
    """
    Classify basic star type and known variability using TIC and local catalogs.

    This step depends on local catalog CSV files for the variability checks. By
    default, if any required local catalog is missing, the function prints a
    clear recommendation to download the missing catalog files and returns a
    small "skipped" dataframe instead of failing or silently producing an
    incomplete classification.

    Set `skip_if_catalogs_missing=False` to run with whichever catalog files are
    available.
    """
    tic_id_int = int(tic_id)
    tic_id = str(tic_id)
    outdir = Path(base_path).expanduser()
    data_dir = outdir / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    catalog_filenames = {
        "eclipsing_binary_catalog": "hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv",
        "stellar_variability_catalog": "hlsp_tess-svc_tess_lcf_all-s0001-s0026_tess_v1.0_cat.csv",
        "toi_catalog": "hlsp_tess-data-alerts_tess_phot_alert-summary-s01+s02+s03+s04_tess_v9_spoc.csv",
    }

    eb_catalog_path = find_catalog_file(outdir, catalog_filenames["eclipsing_binary_catalog"])
    svc_catalog_path = find_catalog_file(outdir, catalog_filenames["stellar_variability_catalog"])
    toi_catalog_path = find_catalog_file(outdir, catalog_filenames["toi_catalog"])

    catalog_paths = {
        "eclipsing_binary_catalog": eb_catalog_path,
        "stellar_variability_catalog": svc_catalog_path,
        "toi_catalog": toi_catalog_path,
    }

    missing_catalog_files = [
        filename
        for key, filename in catalog_filenames.items()
        if catalog_paths[key] is None
    ]

    if skip_if_catalogs_missing and missing_catalog_files:
        note = (
            "Skipping get_star_type_and_variability because one or more required "
            "local catalog files are missing. It would be wise to download the "
            "missing catalog files so this TIC can be checked properly."
        )
        print(note)
        print("Missing catalog files:")
        for filename in missing_catalog_files:
            print(f"  - {filename}")

        star_type_df = pd.DataFrame(
            {
                "tic_id": [tic_id_int],
                "star_type": ["skipped_missing_catalog_files"],
                "variability_type": ["skipped_missing_catalog_files"],
                "in_eclipsing_binary": [pd.NA],
                "catalog_check_skipped": [True],
                "missing_catalog_files": ["; ".join(missing_catalog_files)],
                "note": [note],
            }
        )

        outfile = data_dir / f"{tic_id}_star_type_df.csv"
        star_type_df.to_csv(outfile, index=False)
        print(f"Saved skipped star-type output to {outfile}")
        return star_type_df

    # Eclipsing binary catalog check.
    in_eclipsing_binary = False

    if eb_catalog_path is not None:
        try:
            eb = pd.read_csv(eb_catalog_path)
            eb.columns = [c.strip() for c in eb.columns]

            eb_id_col = None
            for col in ["tess_id", "tic_id", "ID", "id"]:
                if col in eb.columns:
                    eb_id_col = col
                    break

            if eb_id_col is not None:
                eb[eb_id_col] = pd.to_numeric(eb[eb_id_col], errors="coerce")
                in_eclipsing_binary = bool((eb[eb_id_col] == tic_id_int).any())
        except Exception:
            in_eclipsing_binary = False

    # Physical star type from TIC.
    star_type = "unknown"

    try:
        tbl = Catalogs.query_criteria(catalog="Tic", ID=tic_id_int)

        if len(tbl) > 0:
            row = tbl[0]

            def getv(name: str, default: Any = None) -> Any:
                try:
                    value = row[name]
                    if pd.isna(value):
                        return default
                    return value
                except Exception:
                    return default

            obj_type = getv("objType")
            teff = getv("Teff")
            logg = getv("logg")
            radius = getv("rad")

            if obj_type != "STAR":
                star_type = str(obj_type).lower() if obj_type is not None else "unknown"
            else:
                star_type = "star"

                if teff is not None and radius is not None:
                    try:
                        teff = float(teff)
                        radius = float(radius)

                        if teff > 10000 and radius < 0.1:
                            star_type = "compact star"
                    except Exception:
                        pass

                if logg is not None:
                    try:
                        logg = float(logg)

                        if logg >= 4.1:
                            star_type = "main-sequence star"
                        elif 3.2 <= logg < 4.1:
                            star_type = "subgiant"
                        elif logg < 3.2:
                            star_type = "giant"
                    except Exception:
                        pass

    except Exception:
        star_type = "unknown"

    # Variability type priority: EB > TOI > SVC > none.
    variability_type = "no known variability class"

    if in_eclipsing_binary:
        variability_type = "eclipsing binary"

    if variability_type == "no known variability class" and toi_catalog_path is not None:
        try:
            toi = pd.read_csv(toi_catalog_path)
            toi.columns = [c.strip() for c in toi.columns]

            toi_id_col = None
            for col in ["TIC ID", "tic_id", "ticid", "TIC"]:
                if col in toi.columns:
                    toi_id_col = col
                    break

            if toi_id_col is not None:
                toi[toi_id_col] = pd.to_numeric(toi[toi_id_col], errors="coerce")

                if bool((toi[toi_id_col] == tic_id_int).any()):
                    variability_type = "planet/transit-like"
        except Exception:
            pass

    if variability_type == "no known variability class" and svc_catalog_path is not None:
        try:
            svc = pd.read_csv(svc_catalog_path)
            svc.columns = [c.strip() for c in svc.columns]

            svc_id_col = None
            for col in ["tess_id", "tic_id", "ID", "id"]:
                if col in svc.columns:
                    svc_id_col = col
                    break

            if svc_id_col is not None:
                svc[svc_id_col] = pd.to_numeric(svc[svc_id_col], errors="coerce")
                match = svc.loc[svc[svc_id_col] == tic_id_int]

                if not match.empty:
                    row = match.iloc[0]

                    solution = None
                    for col in ["solution", "Solution"]:
                        if col in match.columns:
                            solution = str(row[col]).strip().lower()
                            break

                    if solution == "acf":
                        variability_type = "rotational variable"
                    elif solution == "2sin":
                        variability_type = "reflection binary candidate"
                    elif solution == "1sin":
                        variability_type = "pulsating variable"

        except Exception:
            pass

    star_type_df = pd.DataFrame(
        {
            "tic_id": [tic_id_int],
            "star_type": [star_type],
            "variability_type": [variability_type],
            "in_eclipsing_binary": [in_eclipsing_binary],
            "catalog_check_skipped": [False],
            "missing_catalog_files": ["; ".join(missing_catalog_files)],
            "note": [""],
        }
    )

    outfile = data_dir / f"{tic_id}_star_type_df.csv"
    star_type_df.to_csv(outfile, index=False)

    print(f"Saved {outfile}")
    return star_type_df

def detrend_and_flares(
    base_path: str | Path,
    tic_id: str | int,
    make_plots: bool = False,
    display_dataframes: bool = True,
) -> dict[str, Any]:
    """Run the light-curve detrending and two-pass flare finder for one TIC."""
    import_selas_components()

    if get_timeseries is None or detrend_dataframe is None or run_two_pass_flare_finder is None:
        raise RuntimeError("SELAS components were not imported correctly.")

    tic_id = str(tic_id)
    base_path = Path(base_path).expanduser()
    data_dir = base_path / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_csv = data_dir / f"{tic_id}_detrended.csv"
    flares_csv = data_dir / f"{tic_id}_flares.csv"

    print("\n")
    print(f"Started Lightcurve Detrending Algorithm for TIC {tic_id}.")
    print("\n")

    ts_df = get_timeseries(base_path, tic_id)
    _display_dataframe(ts_df, display_dataframes)

    result, gap_info = detrend_dataframe(ts_df, clean_gaps=True)

    print()
    print_summary(result, tic_id=tic_id)

    result.final_df.to_csv(output_csv, index=False)
    detrended_df = result.final_df
    _display_dataframe(detrended_df.head(), display_dataframes)
    print(f"Saved detrended output to: {output_csv}")

    flare_results = run_two_pass_flare_finder(
        detrended_df,
        make_plots=make_plots,
        save_outputs=False,
        out_dir=data_dir,
        tic_id=tic_id,
        verbose=True,
    )

    print("\n")
    print(f"Finished Flare Finding Algorithm for TIC {tic_id}.")
    print("\n")

    final_flares_df = flare_results["final_flares_df"]

    if len(final_flares_df) > 0:
        final_flares_df = final_flares_df.copy()
        final_flares_df["tstart"] = final_flares_df["new_start_time"]

    final_flares_df.to_csv(flares_csv, index=False)
    _display_dataframe(final_flares_df, display_dataframes)

    return {
        "input_timeseries_df": ts_df,
        "detrend_result": result,
        "gap_info": gap_info,
        "detrended_df": detrended_df,
        "flare_results": flare_results,
        "final_flares_df": final_flares_df,
        "detrended_csv": output_csv,
        "flares_csv": flares_csv,
    }


def load_tic_ids_from_file(path: str | Path, start_id: str | int | None = None) -> list[str]:
    """
    Load TIC IDs from an `All_ids.txt` style file.

    The original notebook expected a line like `all_ids=[...]` and parsed the
    right side of the equals sign.
    """
    path = Path(path).expanduser()
    pd_read_csv = pd.read_csv(path, sep="=", header=None)
    all_ids = ast.literal_eval(pd_read_csv.iloc[0, 1].strip())
    all_ids = [str(tic_id) for tic_id in all_ids]

    if start_id is not None:
        start_id = str(start_id)
        if start_id in all_ids:
            start_idx = all_ids.index(start_id)
            all_ids = all_ids[start_idx:]
        else:
            raise ValueError(f"ID {start_id} not found in all_ids")

    return all_ids


def _print_dataframe_section(title: str, df: pd.DataFrame) -> None:
    """Print a section header and dataframe in log-friendly text format."""
    print("-" * 65)
    print(f"\n{title}:")
    print(df.to_string(index=False))
    print("\n")


def _run_single_tic_pipeline_impl(
    tic_id: str | int,
    data_root: str | Path,
    make_plots: bool,
    display_dataframes: bool,
    run_rotation: bool,
    run_star_properties_step: bool,
    run_star_type_step: bool,
    run_detrend_flares_step: bool,
    run_waiting_time_step: bool,
    run_periodicity_step: bool,
    run_jackknife_step: bool,
) -> dict[str, Any]:
    """Internal implementation of the full one-TIC workflow."""
    import_selas_components()

    tic_id = str(tic_id)
    base_path, data_path = ensure_tic_dirs(tic_id, data_root=data_root)

    results: dict[str, Any] = {
        "tic_id": tic_id,
        "base_path": base_path,
        "data_path": data_path,
        "status": "running",
    }

    print("=" * 65)
    print(f"Started full pipeline for TIC {tic_id}")
    print(f"Base path: {base_path}")
    print("=" * 65)

    if run_rotation:
        df_stellar_rotation = stellar_rotation_calculator(tic_id, base_path)
        results["df_stellar_rotation"] = df_stellar_rotation
        _print_dataframe_section("Stellar rotation dataframe", df_stellar_rotation)

    if run_star_properties_step:
        df_star = star_properties(base_path, tic_id)
        results["df_star"] = df_star
        _print_dataframe_section("Star properties dataframe", df_star)

    # if run_star_type_step:
    #     star_type_df = get_star_type_and_variability(tic_id, base_path)
    #     results["star_type_df"] = star_type_df
    #     _print_dataframe_section("Star type dataframe", star_type_df)

    if run_detrend_flares_step:
        results["detrend_and_flares"] = detrend_and_flares(
            base_path,
            tic_id,
            make_plots=make_plots,
            display_dataframes=display_dataframes,
        )

    if run_waiting_time_step:
        if run_waiting_time_statistics is None:
            raise RuntimeError("run_waiting_time_statistics was not imported correctly.")

        print("\n" + "-" * 65 + "\n")
        print(f"Started Waiting Time Distribution Analysis for TIC {tic_id}.")
        waiting_time_summary = run_waiting_time_statistics(
            tic_id,
            base_path,
            make_plots=make_plots,
            save_distribution_data=True,
        )
        results["waiting_time_summary"] = waiting_time_summary
        print("PDF data:", waiting_time_summary.get("pdf_data_path"))
        print("CDF data:", waiting_time_summary.get("cdf_data_path"))

    if run_periodicity_step:
        if run_periodicity_workflow is None:
            raise RuntimeError("run_periodicity_workflow was not imported correctly.")

        print("\n" + "-" * 65 + "\n")
        print("Started Clustering Analysis")
        print("-- May take a minute")
        rayleigh_result = run_periodicity_workflow(tic_id, base_path, make_plots=make_plots)
        rayleigh_result = add_rayleigh_exceedance_sigma(rayleigh_result)
        results["rayleigh_result"] = rayleigh_result

        if isinstance(rayleigh_result, dict) and "summary" in rayleigh_result:
            _display_dataframe(rayleigh_result["summary"], display_dataframes)

        if run_jackknife_step and run_jackknife_if_rayleigh_exceedance is not None:
            summary_df = rayleigh_result.get("summary")
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                try:
                    best_p_value = float(summary_df["RT_best_p_value"].iloc[0])
                    gridsize = int(summary_df["gridsize"].iloc[0])
                    exceedance = float(summary_df["RT_exceedance_sigma"].iloc[0])
                except Exception:
                    best_p_value = np.nan
                    gridsize = 0
                    exceedance = np.nan

                print(f"RT exceedance sigma: {exceedance}")
                results["RT_exceedance_sigma"] = exceedance

                if np.isfinite(best_p_value) and gridsize > 0 and np.isfinite(exceedance) and exceedance > 0:
                    try:
                        ray = rayleigh_result.get("rayleigh")
                        periods = getattr(ray, "periods", None)
                        period_seeds = getattr(ray, "period_seeds", None)
                        phase_bins = None
                        try:
                            phase_bins = int(rayleigh_result.get("summary").get("phase_bins").iloc[0])
                        except Exception:
                            phase_bins = None

                        jackknife_df = run_jackknife_if_rayleigh_exceedance(
                            tic_id,
                            best_p_value,
                            gridsize,
                            base_root=base_path.parent,
                            n_jobs=-1,
                            show_progress=True,
                            periods=periods,
                            period_seeds=period_seeds,
                            phase_bins=phase_bins,
                        )
                        results["jackknife_df"] = jackknife_df
                        results["jackknife_path"] = str(base_path / "Results" / "Period_statistics" / f"{tic_id}_jackknife_df.csv")

                        if not jackknife_df.empty:
                            try:
                                flare_df, summary_df, pdf_df, cdf_df, rt_df = _load_full_pipeline_plot_data(
                                    tic_id, base_path, waiting_time_summary, rayleigh_result
                                )
                                plot_path = _make_full_pipeline_plot(
                                    tic_id=tic_id,
                                    base_path=base_path,
                                    flare_df=flare_df,
                                    summary_df=summary_df,
                                    rt_df=rt_df,
                                    pdf_df=pdf_df,
                                    cdf_df=cdf_df,
                                    jackknife_df=jackknife_df,
                                    output_path=base_path / "Data" / f"{tic_id}_full_pipeline_plot.pdf",
                                    show=False,
                                )
                                results["full_pipeline_plot"] = str(plot_path)
                            except Exception as exc:
                                print(f"Full pipeline plot failed: {exc}")
                                results["full_pipeline_plot_error"] = str(exc)
                    except Exception as exc:
                        print(f"Jackknife test failed: {exc}")
                        results["jackknife_error"] = str(exc)

    print("\n" + "=" * 65)
    print(f"Finished TIC {tic_id} successfully")
    print("=" * 65)

    results["status"] = "success"
    return results


def run_single_tic_pipeline(
    tic_id: str | int,
    data_root: str | Path = "../Data/Selas-TIC-ids",
    selas_path: str | Path = "../Selas",
    make_plots: bool = False,
    capture_log: bool = True,
    display_dataframes: bool = True,
    raise_on_error: bool = True,
    run_rotation: bool = True,
    run_star_properties_step: bool = True,
    run_star_type_step: bool = True,
    run_detrend_flares_step: bool = True,
    run_waiting_time_step: bool = True,
    run_periodicity_step: bool = True,
    run_jackknife_step: bool = True,
) -> dict[str, Any]:
    """
    Run the full SELAS workflow for one TIC ID.

    Parameters can be used to turn expensive workflow stages on or off while
    developing/debugging.
    """
    configure_environment(selas_path=selas_path)

    tic_id = str(tic_id)
    base_path, data_path = ensure_tic_dirs(tic_id, data_root=data_root)
    log_txt = data_path / f"{tic_id}_run_log.txt"

    try:
        if capture_log:
            with open(log_txt, "w", encoding="utf-8") as log_file:
                tee_stdout = Tee(sys.stdout, log_file)
                tee_stderr = Tee(sys.stderr, log_file)

                with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                    results = _run_single_tic_pipeline_impl(
                        tic_id=tic_id,
                        data_root=data_root,
                        make_plots=make_plots,
                        display_dataframes=display_dataframes,
                        run_rotation=run_rotation,
                        run_star_properties_step=run_star_properties_step,
                        run_star_type_step=run_star_type_step,
                        run_detrend_flares_step=run_detrend_flares_step,
                        run_waiting_time_step=run_waiting_time_step,
                        run_periodicity_step=run_periodicity_step,
                        run_jackknife_step=run_jackknife_step,
                    )
        else:
            results = _run_single_tic_pipeline_impl(
                tic_id=tic_id,
                data_root=data_root,
                make_plots=make_plots,
                display_dataframes=display_dataframes,
                run_rotation=run_rotation,
                run_star_properties_step=run_star_properties_step,
                run_star_type_step=run_star_type_step,
                run_detrend_flares_step=run_detrend_flares_step,
                run_waiting_time_step=run_waiting_time_step,
                run_periodicity_step=run_periodicity_step,
                run_jackknife_step=run_jackknife_step,
            )

        results["log_path"] = log_txt if capture_log else None
        if capture_log:
            print(f"Saved log for TIC {tic_id} to: {log_txt}")
        return results

    except Exception as exc:
        error_traceback = traceback.format_exc()
        error_result = {
            "tic_id": tic_id,
            "base_path": base_path,
            "data_path": data_path,
            "log_path": log_txt if capture_log else None,
            "status": "error",
            "error": str(exc),
            "traceback": error_traceback,
        }

        print("\n" + "=" * 65)
        print(f"ERROR while processing TIC {tic_id}")
        print("=" * 65)
        print(error_traceback)

        if raise_on_error:
            raise

        return error_result


def run_many_tic_pipelines(
    tic_ids: Iterable[str | int],
    data_root: str | Path = "../Data/Selas-TIC-ids",
    selas_path: str | Path = "../Selas",
    make_plots: bool = False,
    capture_log: bool = True,
    display_dataframes: bool = False,
    raise_on_error: bool = False,
    **pipeline_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run `run_single_tic_pipeline(...)` for multiple TIC IDs."""
    all_results: dict[str, dict[str, Any]] = {}

    for tic_id in tic_ids:
        tic_key = str(tic_id)
        result = run_single_tic_pipeline(
            tic_key,
            data_root=data_root,
            selas_path=selas_path,
            make_plots=make_plots,
            capture_log=capture_log,
            display_dataframes=display_dataframes,
            raise_on_error=raise_on_error,
            **pipeline_kwargs,
        )
        all_results[tic_key] = result

    return all_results


__all__ = [
    "AnalysisConfig",
    "DetrendConfig",
    "FlareFinderConfig",
    "PeriodicityConfig",
    "Tee",
    "configure_environment",
    "detrend_and_flares",
    "detrend_dataframe",
    "ensure_tic_dirs",
    "find_best_stellar_rotation_period",
    "find_catalog_file",
    "find_stellar_rotation_periods",
    "get_star_type_and_variability",
    "get_timeseries",
    "import_selas_components",
    "load_tic_ids_from_file",
    "load_time_series",
    "plot_duration_distribution",
    "plot_final_flares",
    "plot_peak_height_distribution",
    "plot_residuals",
    "plot_segment_quality",
    "plot_selected_windows",
    "plot_trends",
    "print_summary",
    "run_many_tic_pipelines",
    "run_periodicity_workflow",
    "run_single_tic_pipeline",
    "run_two_pass_flare_finder",
    "run_waiting_time_statistics",
    "one_detection_k_sigma",
    "p_value_to_exceedance_sigma",
    "save_flare_outputs",
    "save_result",
    "star_properties",
    "stellar_rotation_calculator",
    "stellar_rotation_period_evolution",
]
