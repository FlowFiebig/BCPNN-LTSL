import os
import pickle
import warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import Utils
import MultislicePlan

_CMAP_GREEN_RED = LinearSegmentedColormap.from_list("green_red", ["#2ca02c", "#d62728"])


def _normalize_confidence_levels(spec):
    """Normalize ``confidence_levels`` to a sorted-descending list of floats in (0,1).

    Accepts ``None`` (returns ``[]``), a single float, or any iterable of
    floats. Sorting descending ensures the wider bands are drawn first so the
    narrower bands sit on top in the fan chart.
    """
    if spec is None:
        return []
    if isinstance(spec, (int, float)):
        spec = [float(spec)]
    levels = [float(L) for L in spec]
    for L in levels:
        if not (0.0 < L < 1.0):
            raise ValueError(
                f"confidence_levels must lie strictly in (0,1); got {L}"
            )
    levels.sort(reverse=True)
    return levels


def _central_band_bounds(dists, centers, level):
    """Per-row central credible interval ``[Q_{(1-L)/2}, Q_{(1+L)/2}]``.

    Treats each row of ``dists`` as a discrete pmf over ``centers`` and uses
    the row-CDF (``np.cumsum``) with ``np.interp`` to map the requested
    quantiles back to bin-center values. Rows are defensively renormalized in
    case the simulator emits an unnormalized residual.

    Parameters
    ----------
    dists : ndarray, shape (N, n_bins)
    centers : ndarray, shape (n_bins,)
    level : float in (0,1)
        Central credible interval level (e.g. 0.68 -> 16th-84th percentile).

    Returns
    -------
    lo, hi : ndarray, each shape (N,)
    """
    p = np.asarray(dists, dtype=float)
    s = p.sum(axis=1, keepdims=True)
    p = p / np.where(s > 0, s, 1.0)
    cdf = np.cumsum(p, axis=1)
    q_lo = (1.0 - level) / 2.0
    q_hi = 1.0 - q_lo
    lo = np.array([np.interp(q_lo, cdf_row, centers) for cdf_row in cdf])
    hi = np.array([np.interp(q_hi, cdf_row, centers) for cdf_row in cdf])
    return lo, hi


def plot_general_prediction(
    paramfile="Parameters.par",
    plot_range=None,
    figno=1,
    confidence_levels=None,
    *,
    feats=None,
    tetroutx=None,
    vanoutx=None,
    teusoutx=None,
    genoutx=None,
    tetroutdists=None,
    vanoutdists=None,
    teusoutdists=None,
    genoutdists=None,
    centers=None,
    rmses=None,
):
    """
    Plot the BCPNN timeseries prediction figure produced by ``Mains.general``.

    The function supports two calling conventions:

    1. **Standalone re-plot after a completed run.** Pass just ``paramfile``.
       The timeseries pickle is re-read, the rolling features are recomputed,
       the model output bin files written by ``ltslmain``
       (``netw1_outpop_{tetr,van,teus,gen}_dist.bin``) are reloaded and
       decoded, and the RMSEs are recomputed from those arrays. Use this
       mode to iterate on the plotting code without rerunning the simulator::

           import importlib, Display
           importlib.reload(Display)
           Display.plot_general_prediction(
               "Parameters.par",
               plot_range=(params["trnpat"] - 500,
                           params["trnpat"] + params["tenpat"]),
               confidence_levels=(0.5, 0.9))

    2. **Called from Mains.general.** ``Mains.general`` already computes
       ``feats``, the four decoded output arrays, the four raw bin
       distributions, the bin centers and the five RMSEs, so it passes them
       through as keyword arguments to avoid redoing the work.

    Parameters
    ----------
    paramfile : str
        Parameter file for the run being plotted (used to read ``offs``,
        ``trnpat``, ``vanpat``, ``tenpat``, ``ngenstep`` and, when reloading
        from disk, ``timeseries_file``, ``mean_nbin`` and ``deriv1st_nbin``).
    plot_range : tuple of two numbers or None
        Optional ``(left, right)`` passed to ``matplotlib.pyplot.xlim`` after
        all series are drawn (zoom only; data are not cropped before plotting).
        ``None`` leaves limits to autoscale.
    figno : int
        matplotlib figure number; if ``<= 0`` nothing is drawn (RMSEs are
        still computed and returned).
    confidence_levels : None, float, or iterable of floats, default None
        Pointwise central credible interval(s) drawn as shaded bands behind
        each predicted phase line. ``None`` disables the band (current
        behaviour). A single float (e.g. ``0.68``) draws one band. An
        iterable like ``(0.5, 0.9)`` draws a fan chart with one band per
        level (outer bands rendered with lower alpha, inner bands with
        higher alpha). Each level ``L`` is the central CI fraction, so the
        plotted band runs from the ``(1-L)/2`` to the ``(1+L)/2`` quantile
        of the per-step bin posterior. Skipped with a warning if the
        underlying ``..._dist`` arrays or ``centers`` are unavailable.

        Caveat: this is a *pointwise* interval over the model's one-step
        posterior at each timestep -- it is not a joint trajectory band.
        For the autoregressive ``generation`` phase it does not capture the
        compounding error of the rollout, only the network's local
        uncertainty. With ``mean_nbin`` bin centers the quantile resolution
        is also limited to roughly half a bin width, so very tight
        posteriors will still show a band of width
        ``~ (x_max - x_min) / mean_nbin``.
    feats, tetroutx, vanoutx, teusoutx, genoutx : ndarray or None
        Precomputed arrays. If any of ``feats``/``tetroutx``/``teusoutx``/
        ``genoutx`` is None all of them are reloaded from disk together with
        the matching ``..._dist`` arrays. ``vanoutx`` may legitimately be
        ``None`` when ``vanpat <= offs``.
    tetroutdists, vanoutdists, teusoutdists, genoutdists : ndarray or None
        Raw bin distributions (rows of the corresponding
        ``netw1_outpop_*_dist.bin`` files). Only consulted when
        ``confidence_levels`` is set. Reloaded together with the decoded
        outputs in standalone mode.
    centers : ndarray or None
        ``rbfbinner_mean.centers`` for the run being plotted. Required when
        ``confidence_levels`` is set; reconstructed in standalone mode.
    rmses : 5-tuple of floats or None
        ``(rmse_train, rmse_validation, rmse_unseen, rmse_generation,
        rmse_persistence)``. Recomputed from the arrays when ``None``.

    Returns
    -------
    tuple of 5 floats
        ``(rmse_train, rmse_validation, rmse_unseen, rmse_generation,
        rmse_persistence)``.
    """
    import FeatureGeneration

    params = Utils.load_params(paramfile)
    offs = int(params["offs"])
    trnpat, vanpat, tenpat = Utils.phase_totals_from_paramfile(paramfile)
    ngenstep = int(params["ngenstep"])
    cfg = FeatureGeneration.feature_config(paramfile)
    pcol = cfg.primary_col  # decode-target column (ground truth)

    need_reload = any(a is None for a in (feats, tetroutx, teusoutx, genoutx))
    if need_reload:
        primary_nbin = cfg.primary_nbin
        timeseries_file = params["timeseries_file"]
        timeseries_path = FeatureGeneration.dataset_path(timeseries_file)
        with open(timeseries_path, "rb") as fp:
            timeseries_data = pickle.load(fp)
        timeseries_x = FeatureGeneration.timeseries_from_dataset(timeseries_data)
        if timeseries_x is None:
            raise ValueError(
                f"Timeseries '{timeseries_file}' contains no usable "
                f"'x' or 'x_norm' array."
            )
        x = np.asarray(timeseries_x, dtype=float)
        feats = cfg.func(x=x, figno=0)

        # Prefer the centres persisted at encode time so the decode grid is exactly
        # what was encoded (essential for rbf-quantile, where centres depend on the
        # full pooled distribution); fall back to recomputing for legacy runs.
        infilename_md = params.get("infilename_md")
        rbfbinner_primary = (
            (FeatureGeneration.load_primary_encoder(infilename_md) if infilename_md else None)
            or FeatureGeneration.primary_encoder(feats, cfg))
        centers = rbfbinner_primary.centers

        tetroutdists = Utils.loadbin("netw1_outpop_tetr_dist.bin", primary_nbin)
        tetroutx = FeatureGeneration.delog_primary(
            cfg, rbfbinner_primary.decode(tetroutdists))

        if vanpat > offs:
            vanoutdists = Utils.loadbin("netw1_outpop_van_dist.bin", primary_nbin)
            vanoutx = FeatureGeneration.delog_primary(
                cfg, rbfbinner_primary.decode(vanoutdists))
        else:
            vanoutdists = None
            vanoutx = None

        teusoutdists = Utils.loadbin("netw1_outpop_teus_dist.bin", primary_nbin)
        teusoutx = FeatureGeneration.delog_primary(
            cfg, rbfbinner_primary.decode(teusoutdists))

        genoutdists = Utils.loadbin("netw1_outpop_gen_dist.bin", primary_nbin)
        genoutx = FeatureGeneration.delog_primary(
            cfg, rbfbinner_primary.decode(genoutdists))

    truth_primary = FeatureGeneration.delog_primary(cfg, feats[:, pcol])

    if rmses is None:
        train_skip = 100
        if trnpat - offs > train_skip:
            rmse_train = Utils.RMSE(
                truth_primary[train_skip + offs : trnpat],
                tetroutx[train_skip : trnpat - offs],
            )
        else:
            rmse_train = float("nan")
        if vanpat > offs and vanoutx is not None:
            rmse_validation = Utils.RMSE(
                truth_primary[trnpat + offs : trnpat + vanpat],
                vanoutx[0 : vanpat - offs],
            )
        else:
            rmse_validation = float("nan")
        if tenpat > offs:
            rmse_unseen = Utils.RMSE(
                truth_primary[trnpat + vanpat + offs : trnpat + vanpat + tenpat],
                teusoutx[0 : tenpat - offs],
            )
        else:
            rmse_unseen = float("nan")
        if ngenstep > 0:
            rmse_generation = Utils.RMSE(
                truth_primary[trnpat + vanpat + tenpat : trnpat + vanpat + tenpat + ngenstep],
                genoutx[0 : ngenstep],
            )
        else:
            rmse_generation = float("nan")
        rmse_persistence = Utils.persistence_rmse(
            truth_primary,
            offs=offs,
            start=trnpat,
            end=trnpat + vanpat + tenpat + ngenstep,
        )
        rmses = (rmse_train, rmse_validation, rmse_unseen,
                 rmse_generation, rmse_persistence)
    else:
        rmse_train, rmse_validation, rmse_unseen, rmse_generation, rmse_persistence = rmses

    full_stop = int(trnpat + vanpat + tenpat + ngenstep)
    xlim = None
    if plot_range is not None:
        if len(plot_range) != 2:
            raise ValueError(
                "plot_range must be None or a length-2 tuple (left, right)"
            )
        xa, xb = float(plot_range[0]), float(plot_range[1])
        if xa > xb:
            raise ValueError("plot_range must have left <= right")
        xlim = (xa, xb)

    if figno > 0:
        levels = _normalize_confidence_levels(confidence_levels)
        if levels and centers is None:
            warnings.warn(
                "confidence_levels was requested but `centers` is None; "
                "skipping confidence bands. Pass centers=rbfbinner_mean.centers "
                "or call this function without precomputed arrays so it can "
                "reconstruct them from disk.",
                UserWarning,
                stacklevel=2,
            )
            levels = []

        def _add_bands(x_range, phase_dists, sliced_dists_factory, color):
            """Draw the requested fan-chart bands for one phase, in-place."""
            if not levels:
                return
            if phase_dists is None:
                return
            dists_slice = sliced_dists_factory(phase_dists)
            if dists_slice.shape[0] == 0:
                return
            # Outer bands first (already at the head of `levels` since the
            # list is sorted descending); each successive (narrower) band is
            # drawn on top with a higher alpha for a classic fan look.
            n = len(levels)
            for i, L in enumerate(levels):
                alpha = 0.25 + 0.20 * (i / max(n - 1, 1))
                lo, hi = _central_band_bounds(dists_slice, centers, L)
                if cfg is not None and cfg.primary_log_encoded:
                    lo, hi = np.exp(lo), np.exp(hi)
                plt.fill_between(x_range, lo, hi, color=color, alpha=alpha,
                                 linewidth=0, zorder=1.5)

        plt.figure(figno + 1, figsize=(14, 6))
        plt.clf()

        n_feat = feats.shape[0]
        ts_b = min(n_feat, full_stop)
        if ts_b > 0:
            plt.plot(
                np.arange(0, ts_b),
                truth_primary[0:ts_b],
                label="timeseries",
            )

        train_x = np.arange(offs, trnpat)
        if train_x.size > 0:
            (train_line,) = plt.plot(
                train_x, tetroutx[0 : trnpat - offs], label="training")
            _add_bands(
                train_x,
                tetroutdists,
                lambda d: d[0 : trnpat - offs],
                train_line.get_color(),
            )

        if vanpat > offs and vanoutx is not None:
            van_x = np.arange(trnpat + offs, trnpat + vanpat)
            (van_line,) = plt.plot(
                van_x, vanoutx[0:vanpat - offs], label="validation")
            _add_bands(
                van_x,
                vanoutdists,
                lambda d: d[0:vanpat - offs],
                van_line.get_color(),
            )

        test_x = np.arange(trnpat + vanpat + offs, trnpat + vanpat + tenpat)
        if test_x.size > 0:
            (test_line,) = plt.plot(
                test_x, teusoutx[0:tenpat - offs], label="testing")
            _add_bands(
                test_x,
                teusoutdists,
                lambda d: d[0:tenpat - offs],
                test_line.get_color(),
            )

        if ngenstep > 0:
            gen_x = np.arange(
                trnpat + vanpat + tenpat,
                trnpat + vanpat + tenpat + ngenstep,
            )
            (gen_line,) = plt.plot(
                gen_x, genoutx[0:ngenstep], label="generation")
            _add_bands(
                gen_x,
                genoutdists,
                lambda d: d[0:ngenstep],
                gen_line.get_color(),
            )

        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])

        plt.title("BCPNN timeseries prediction (" + str(offs) + " steps ahead)  "
                  + "rmse_train = {:.2e}".format(rmse_train) + "  "
                  + "rmse_validation = {:.2e}".format(rmse_validation) + "  "
                  + "rmse_unseen = {:.2e}".format(rmse_unseen) + "  "
                  + "rmse_generation = {:.2e}".format(rmse_generation) + "  "
                  + "rmse_persistence = {:.2e}".format(rmse_persistence), fontsize=11)
        plt.xlabel("time (steps)")
        plt.legend(fontsize=12)
        plt.tight_layout()

    return rmses


def _plot_prediction_bands(x_range, dists, centers, levels, color, ax=None, cfg=None):
    """Draw fan-chart credible bands (see :func:`plot_general_prediction`)."""
    if not levels or dists is None or len(x_range) == 0:
        return
    dists = np.asarray(dists, dtype=float)
    if dists.shape[0] != len(x_range):
        return
    fill = ax.fill_between if ax is not None else plt.fill_between
    n = len(levels)
    for i, level in enumerate(levels):
        alpha = 0.25 + 0.20 * (i / max(n - 1, 1))
        lo, hi = _central_band_bounds(dists, centers, level)
        if cfg is not None and cfg.primary_log_encoded:
            lo, hi = np.exp(lo), np.exp(hi)
        fill(x_range, lo, hi, color=color, alpha=alpha,
             linewidth=0, zorder=1.5)


def _multislice_block_plan(paramfile, *, rundir=None):
    """
    Rebuild the multislice block schedule and global mean RBF encoder.

    Deprecated alias for :func:`MultislicePlan.build_multislice_plan` kept for
    backward compatibility.
    """
    return MultislicePlan.build_multislice_plan(
        paramfile, encode=False, prefer_saved_encoder=True, rundir=rundir)


def stitch_multislice_ticker(paramfile, ticker, *, plan=None, rundir=None,
                             tetroutx=None, vanoutx=None, teusoutx=None,
                             tetroutdists=None, vanoutdists=None,
                             teusoutdists=None):
    """
    Extract one ticker's stitched prediction stream from a multislice run.

    Walk-forward blocks concatenate tickers phase-major inside each block.
    This function inverts that layout for a single symbol: for every block
    where the ticker is active it collects train / val / test segments in
    calendar order and returns them as contiguous arrays.

    Parameters
    ----------
    paramfile : str
        Multislice ``.par`` file used for the run (defines schedule and bins).
    ticker : str
        Column name / symbol to isolate (e.g. ``"NVDA"``).
    plan : dict or None
        Optional precomputed block plan from :func:`_multislice_block_plan`
        or :func:`MultislicePlan.build_multislice_plan`.
    rundir : str or None
        Directory containing ``netw1_outpop_*_dist.bin`` (``None`` = CWD).
    tetroutx, vanoutx, teusoutx : ndarray or None
        Pre-decoded output streams. Reloaded from ``netw1_outpop_*_dist.bin``
        when ``None``.
    tetroutdists, vanoutdists, teusoutdists : ndarray or None
        Raw bin posteriors matching the decoded streams above.

    Returns
    -------
    dict with keys ``x_feat``, ``truth``, ``pred``, ``dists``, ``phase``
    (0=train, 1=val, 2=test), ``block``, ``feats``, ``centers``, ``offs``,
    ``trnpat``, ``vanpat``, ``tenpat``, ``ticker``, and ``plan``.
    """
    import FeatureGeneration

    if plan is None:
        plan = _multislice_block_plan(paramfile, rundir=rundir)
    elif rundir is None:
        rundir = plan.get("rundir")

    ticker = str(ticker).strip()
    if ticker not in plan["per_ticker"]:
        raise KeyError(
            f"Ticker '{ticker}' not in multislice run "
            f"({len(plan['ticker_labels'])} symbols loaded).")

    offs = plan["offs"]
    trnpat = plan["trnpat"]
    vanpat = plan["vanpat"]
    tenpat = plan["tenpat"]
    warmup_pat = plan["warmup_pat"]
    mean_nbin = plan["mean_nbin"]
    pcol = plan["primary_col"]
    cfg = plan["cfg"]
    rbfbinner_mean = plan["rbfbinner_mean"]
    per_ticker = plan["per_ticker"]
    block_meta = plan["block_meta"]
    feats = per_ticker[ticker]["feats"]

    need_reload = tetroutx is None
    if need_reload:
        tetr_path = MultislicePlan.multislice_artifact_path(
            "tetr_dist", rundir=rundir, plan=plan)
        van_path = MultislicePlan.multislice_artifact_path(
            "van_dist", rundir=rundir, plan=plan)
        teus_path = MultislicePlan.multislice_artifact_path(
            "teus_dist", rundir=rundir, plan=plan)
        tetroutdists = Utils.loadbin(tetr_path, mean_nbin)
        tetroutx = FeatureGeneration.delog_primary(
            cfg, rbfbinner_mean.decode(tetroutdists))
        if vanpat > offs:
            vanoutdists = Utils.loadbin(van_path, mean_nbin)
            vanoutx = FeatureGeneration.delog_primary(
                cfg, rbfbinner_mean.decode(vanoutdists))
        else:
            vanoutdists = None
            vanoutx = None
        teusoutdists = Utils.loadbin(teus_path, mean_nbin)
        teusoutx = FeatureGeneration.delog_primary(
            cfg, rbfbinner_mean.decode(teusoutdists))

    expected = plan["stream_lengths"]
    got = (len(tetroutx), len(vanoutx) if vanoutx is not None else 0,
           len(teusoutx))
    if got != expected:
        raise RuntimeError(
            f"Decoded stream length mismatch (tetr/van/teus): expected "
            f"{expected} but got {got}. ltslmain likely ran with a different "
            f"schedule than '{paramfile}'.")

    x_list, truth_list, pred_list, dist_list = [], [], [], []
    phase_list, block_list = [], []

    phase_specs = [
        (0, trnpat, 0, tetroutx, tetroutdists, "tetr_block_start"),
        (1, vanpat, trnpat, vanoutx, vanoutdists, "van_block_start"),
        (2, tenpat, trnpat + vanpat, teusoutx, teusoutdists, "teus_block_start"),
    ]

    for b_local, meta in enumerate(block_meta):
        active = meta["active"]
        match = [(k, sl) for k, (label, sl) in enumerate(active) if label == ticker]
        if not match:
            continue
        k, start_local = match[0]
        for phase_idx, per_t_len, phase_off, stream, dist_stream, block_start_key \
                in phase_specs:
            if per_t_len <= offs or stream is None or dist_stream is None:
                continue
            block_start = meta[block_start_key]
            w = Utils.multislice_warmup_len(k, warmup_pat)
            scored_len = Utils.multislice_scored_len(
                k, per_t_len, offs, warmup_pat)
            if scored_len <= 0:
                continue
            if warmup_pat > 0:
                sl_stream = (block_start
                             + Utils.multislice_stream_offset(
                                 k, per_t_len, offs, warmup_pat))
            else:
                sl_stream = block_start + k * per_t_len
            pred = stream[sl_stream : sl_stream + scored_len]
            dists = dist_stream[sl_stream : sl_stream + scored_len]
            if warmup_pat > 0:
                x = np.arange(start_local + phase_off + w + offs,
                              start_local + phase_off + per_t_len)
            else:
                x = np.arange(start_local + phase_off + offs,
                              start_local + phase_off + per_t_len)
            truth = FeatureGeneration.delog_primary(cfg, feats[x, pcol])
            if len(pred) != len(truth) or len(pred) == 0:
                continue
            x_list.append(x)
            truth_list.append(truth)
            pred_list.append(pred)
            dist_list.append(dists)
            phase_list.append(np.full(len(pred), phase_idx, dtype=int))
            block_list.append(np.full(len(pred), b_local, dtype=int))

    if not x_list:
        raise ValueError(
            f"Ticker '{ticker}' has no stitched prediction segments "
            f"(not active in any block?).")

    return {
        "x_feat" : np.concatenate(x_list),
        "truth"  : np.concatenate(truth_list),
        "pred"   : np.concatenate(pred_list),
        "dists"  : np.concatenate(dist_list, axis=0),
        "phase"  : np.concatenate(phase_list),
        "block"  : np.concatenate(block_list),
        "feats"  : feats,
        "centers": plan["centers"],
        "offs"   : offs,
        "trnpat" : trnpat,
        "vanpat" : vanpat,
        "tenpat" : tenpat,
        "primary_col" : pcol,
        "cfg"         : cfg,
        "ticker" : ticker,
        "plan"   : plan,
    }


def ticker_feat_dates(date_index, series):
    """
    Calendar dates aligned with ``series['feats']`` from
    :func:`stitch_multislice_ticker`.

    Use the wide CSV ``Date`` column (or any index with one row per calendar
    day) together with the stitched ``series`` dict so plotting dates stay in
    sync with the param file that built the run.
    """
    ticker = series["ticker"]
    d = series["plan"]["per_ticker"][ticker]
    fv = int(d["first_valid_feat_row"])
    n = int(series["feats"].shape[0])
    return date_index.iloc[fv : fv + n]


def resolve_multislice_step(series, step=None, *, feat_row=None, date=None,
                            dates=None):
    """
    Resolve a stitched prediction step to an integer index in ``[0, N)``.

    Exactly one selector must be given. ``step`` supports negative indexing.
    ``date`` requires ``dates`` aligned with ``series['feats']`` (e.g. from
    :func:`ticker_feat_dates`).
    """
    n = int(len(series["pred"]))
    selectors = sum(x is not None for x in (step, feat_row, date))
    if selectors != 1:
        raise ValueError(
            "Pass exactly one of step, feat_row, or date.")
    if step is not None:
        idx = int(step)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"step {step!r} out of range for {n} scored steps.")
        return idx
    if feat_row is not None:
        matches = np.where(series["x_feat"] == int(feat_row))[0]
        if matches.size == 0:
            raise KeyError(f"No scored step at feature row {feat_row}.")
        if matches.size > 1:
            raise ValueError(
                f"Feature row {feat_row} appears in {matches.size} scored "
                f"steps; disambiguate with step=.")
        return int(matches[0])
    if dates is None:
        raise ValueError("date= requires dates aligned with series['feats'].")
    import pandas as pd
    dates_idx = pd.DatetimeIndex(dates)
    target = pd.Timestamp(date)
    if target not in dates_idx:
        raise KeyError(f"Date {target.date()} not in ticker feature calendar.")
    row = int(dates_idx.get_loc(target))
    matches = np.where(series["x_feat"] == row)[0]
    if matches.size == 0:
        raise KeyError(
            f"Date {target.date()} (feature row {row}) has no scored model output.")
    if matches.size > 1:
        raise ValueError(
            f"Date {target.date()} maps to {matches.size} scored steps "
            f"(overlapping train/val/test segments); pass step=.")
    return int(matches[0])


def _resolve_dist_xscale(cfg, xscale):
    """``linear`` = bar widths from centre gaps; ``quantile`` = equal-width bins."""
    if xscale is not None:
        xscale = str(xscale).strip().lower()
        if xscale not in ("linear", "quantile"):
            raise ValueError(
                f"xscale must be 'linear', 'quantile', or None; got {xscale!r}")
        return xscale
    binmode = getattr(cfg, "binmode", "rbf")
    return "quantile" if binmode == "rbf-quantile" else "linear"


def _value_to_bin_index(value, centers):
    """Map a feature value to fractional bin index (for quantile-scale pmf plots)."""
    centers = np.asarray(centers, dtype=float)
    idx = np.arange(len(centers), dtype=float)
    if len(centers) < 2:
        return 0.0
    return float(np.interp(float(value), centers, idx))


def _even_bin_axis_ticks(n_bins, x_centers, *, max_ticks=7):
    """Parent-axis tick positions evenly spaced in bin index, labelled by centre."""
    x_centers = np.asarray(x_centers, dtype=float)
    n_bins = int(n_bins)
    if n_bins <= 0:
        return np.array([]), []
    if n_bins == 1:
        return np.array([0.0]), [f"{x_centers[0]:.4g}"]
    n_ticks = min(int(max_ticks), n_bins)
    tick_bins = np.unique(
        np.round(np.linspace(0, n_bins - 1, n_ticks)).astype(int))
    positions = tick_bins.astype(float)
    labels = [f"{x_centers[i]:.4g}" for i in tick_bins]
    return positions, labels


def _resolve_var_from_result(result, ticker):
    """
    Extract per-stitched-step VaR and var_level from an analyze_multislice result.
    """
    if result is None:
        raise ValueError(
            "result= is required (output of Mains.analyze_multislice with "
            "metrics including 'var' and var_level=VAR_LEVEL).")
    if "var" not in result:
        raise ValueError(
            "result is missing 'var'; rerun analyze with "
            "metrics=(..., 'var') and var_level=VAR_LEVEL.")
    if "var_level" not in result:
        raise ValueError(
            "result is missing 'var_level'; rerun analyze with "
            "metrics=(..., 'var') and var_level=VAR_LEVEL.")
    ticker = str(ticker).strip()
    var_by_ticker = result["var"]
    if ticker not in var_by_ticker:
        raise KeyError(
            f"Ticker '{ticker}' not in result['var'] "
            f"({list(var_by_ticker)}).")
    var_level = float(result["var_level"])
    if not 0.0 < var_level < 1.0:
        raise ValueError(f"var_level must lie strictly in (0, 1); got {var_level}")
    return np.asarray(var_by_ticker[ticker], dtype=float), var_level


def _draw_multislice_step_pmf(ax, pmf, x_centers, *, dist_xscale, x_truth, x_pred,
                              var_level, value_axis_label, title=None,
                              show_legend=True, var_value):
    """Draw pmf bars and VaR / truth / decode-mean markers on *ax*."""
    pmf = np.asarray(pmf, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)
    n_bins = len(pmf)

    if dist_xscale == "quantile":
        x_pos = np.arange(n_bins, dtype=float)
        widths = np.ones(n_bins, dtype=float)
        x_truth_plot = _value_to_bin_index(x_truth, x_centers)
        x_pred_plot = _value_to_bin_index(x_pred, x_centers)
        tick_pos, tick_labels = _even_bin_axis_ticks(n_bins, x_centers)
        ax.set_xlim(-0.5, n_bins - 0.5)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        xlabel = (f"{value_axis_label} "
                  f"(quantile bins centre; even-spaced bins plot)")
    else:
        if n_bins > 1:
            half = 0.5 * np.diff(x_centers)
            edges = np.concatenate([[x_centers[0] - half[0]], x_centers[:-1] + half,
                                    [x_centers[-1] + half[-1]]])
            widths = np.diff(edges)
            x_pos = edges[:-1]
        else:
            x_pos = np.array([x_centers[0] - 0.5])
            widths = np.array([1.0])
        x_truth_plot = x_truth
        x_pred_plot = x_pred
        xlabel = value_axis_label

    ax.bar(x_pos, pmf, width=widths, align="edge",
           color="#4C72B0", alpha=0.75, edgecolor="none", label="model pmf")
    x_var = float(var_value)
    x_var_plot = (_value_to_bin_index(x_var, x_centers)
                  if dist_xscale == "quantile" else x_var)
    pct = float(var_level) * 100.0
    var_label = (f"VaR ({int(round(pct))}%)"
                 if abs(pct - round(pct)) < 1e-9 else f"VaR ({pct:g}%)")
    ax.axvline(x_var_plot, color="#DD8452", lw=2.5, label=var_label)
    ax.axvline(x_truth_plot, color="0.25", lw=2.5, ls="--", label="truth")
    ax.axvline(x_pred_plot, color="#2ca02c", lw=2.5, label="decode mean")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability mass")
    if title is not None:
        ax.set_title(title)
    if show_legend:
        ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    return x_var


def plot_multislice_step_distribution(series, step=None, *, feat_row=None,
                                      date=None, dates=None, ax=None,
                                      ax_linear=None,
                                      figsize=(9, 4), figno=2,
                                      log_scale=None, xscale=None,
                                      dual_xscale=False,
                                      var_value, var_level=0.95,
                                      savepath=None, save_as=None, layout=True):
    """
    Plot the discrete output pmf over return-bin centres for one scored step.

    Parameters
    ----------
    series : dict
        Output of :func:`stitch_multislice_ticker`.
    step, feat_row, date : selectors
        Exactly one must be given; see :func:`resolve_multislice_step`.
    dates : array-like or None
        Calendar dates aligned with ``series['feats']`` when using ``date=``.
    ax : matplotlib.axes.Axes or None
    figsize : tuple
    figno : int
    log_scale : bool or None
        ``None`` (default): plot in the model's encoded primary units — log
        return when ``primary_log_encoded``, else raw feature values.
        ``True``: force log-return / encoded units.
        ``False``: map log-encoded outputs to daily return ratio via ``exp``.
    xscale : str or None
        ``None`` (default): ``'quantile'`` when ``binmode`` is ``rbf-quantile``
        (equal-width bars per bin; tick labels show bin-centre values at evenly
        spaced quantile indices), else ``'linear'`` (bar edges from midpoint gaps
        between centres). Pass ``'quantile'`` or ``'linear'`` to override.
    ax_linear : matplotlib.axes.Axes or None
        Optional second axes for the linear encoded-range pmf. Used with
        ``dual_xscale=True`` when the caller owns a multi-row figure.
    dual_xscale : bool
        When ``True``, also plot the pmf over the linear encoded-value range
        (variable bin widths) on a second panel below the primary panel.
        Ignored when the primary ``xscale`` is already ``'linear'``.
    var_level : float
        VaR level for the legend label (e.g. ``0.95`` → ``VaR (95%)``).
    var_value : float
        Precomputed VaR for this step (from
        :func:`MultisliceAnalyze.analyze_multislice` ``result['var']``).
    savepath, save_as : optional figure save targets (same convention as
        :func:`plot_multislice_prediction`).

    Returns
    -------
    fig, ax : matplotlib figure and axes
    info : dict with step index, phase, block, feat row, truth, pred, pmf.
    """
    import FeatureGeneration

    idx = resolve_multislice_step(
        series, step=step, feat_row=feat_row, date=date, dates=dates)

    cfg = series.get("cfg", series["plan"]["cfg"])
    centers = np.asarray(series["centers"], dtype=float)
    pmf = np.asarray(series["dists"][idx], dtype=float)
    s = pmf.sum()
    if s > 0:
        pmf = pmf / s

    if log_scale is None:
        log_scale = bool(cfg.primary_log_encoded)
    dist_xscale = _resolve_dist_xscale(cfg, xscale)
    var_level = float(var_level)
    if not 0.0 < var_level < 1.0:
        raise ValueError(f"var_level must lie strictly in (0, 1); got {var_level}")

    x_mean_enc = float(centers @ pmf)
    if log_scale or not cfg.primary_log_encoded:
        x_centers = centers
        if cfg.primary_log_encoded:
            x_truth = float(np.log(series["truth"][idx]))
            x_pred = x_mean_enc
        else:
            x_truth = float(series["truth"][idx])
            x_pred = float(series["pred"][idx])
        value_axis_label = ("log return" if cfg.primary_log_encoded
                            else "primary feature")
    else:
        x_centers = FeatureGeneration.delog_primary(cfg, centers)
        x_truth = float(series["truth"][idx])
        x_pred = float(FeatureGeneration.delog_primary(cfg, x_mean_enc))
        value_axis_label = "daily return (ratio)"

    phase_names = ("train", "validation", "test")
    phase = int(series["phase"][idx])
    block = int(series["block"][idx])
    feat_row_i = int(series["x_feat"][idx])
    ticker = series["ticker"]

    show_linear = bool(dual_xscale) and dist_xscale != "linear"

    date_str = ""
    if dates is not None:
        dates_arr = np.asarray(dates)
        if 0 <= feat_row_i < len(dates_arr):
            d = dates_arr[feat_row_i]
            if hasattr(d, "date"):
                date_str = f"  {d.date()}"
            else:
                date_str = f"  {d}"

    title = (f"{ticker} — output distribution  "
             f"block {block}  {phase_names[phase]}{date_str}  "
             f"(feat row {feat_row_i})")

    if ax is None:
        if show_linear and ax_linear is None:
            fig, axes = plt.subplots(
                2, 1, figsize=(figsize[0], figsize[1] * 1.75),
                num=figno, sharey=True)
            ax, ax_linear = axes
        else:
            fig, ax = plt.subplots(figsize=figsize, num=figno)
    else:
        fig = ax.figure

    x_var = _draw_multislice_step_pmf(
        ax, pmf, x_centers, dist_xscale=dist_xscale,
        x_truth=x_truth, x_pred=x_pred, var_level=var_level,
        value_axis_label=value_axis_label, title=title,
        var_value=var_value)

    if show_linear:
        if ax_linear is None:
            raise ValueError(
                "dual_xscale=True with a supplied ax requires ax_linear=.")
        ax_linear.set_visible(True)
        _draw_multislice_step_pmf(
            ax_linear, pmf, x_centers, dist_xscale="linear",
            x_truth=x_truth, x_pred=x_pred, var_level=var_level,
            value_axis_label=(f"{value_axis_label} "
                                f"(linear encoded range; variable bin width)"),
            title=None, var_value=var_value)
    elif ax_linear is not None:
        ax_linear.set_visible(False)

    if layout:
        fig.tight_layout()
    _save_fig(fig, savepath, f"output_dist_{ticker}_step{idx}.png",
              save_as=save_as)

    info = {
        "step": idx,
        "phase": phase,
        "block": block,
        "feat_row": feat_row_i,
        "truth": x_truth,
        "pred": x_pred,
        "pmf": pmf,
        "centers": x_centers,
        "xscale": dist_xscale,
        "var_level": var_level,
        "var": x_var,
    }
    if show_linear:
        info["ax_linear"] = ax_linear
        info["xscale_linear"] = "linear"
    return fig, ax, info


def _normalize_phase_filter(phases):
    if phases is None:
        return {0, 1, 2}
    if isinstance(phases, (str, int)):
        phases = (phases,)
    aliases = {
        "train": 0, "training": 0,
        "val": 1, "validation": 1,
        "test": 2, "testing": 2, "unseen": 2,
    }
    out = set()
    for p in phases:
        if isinstance(p, str):
            key = p.strip().lower()
            if key not in aliases:
                raise ValueError(
                    f"Unknown phase '{p}'; expected train, val, or test.")
            out.add(aliases[key])
        else:
            idx = int(p)
            if idx not in (0, 1, 2):
                raise ValueError(f"Phase index must be 0, 1, or 2; got {idx}")
            out.add(idx)
    return out


def plot_multislice_prediction(paramfile, ticker, *, ax=None, figsize=(14, 6),
                               figno=1, plot_range=None,
                               confidence_levels=(0.5, 0.9), dates=None,
                               series=None, savepath=None, save_as=None,
                               phases=None, rundir=None):
    """
    Plot one ticker's stitched multislice mean prediction with uncertainty bands.

    Based on :func:`plot_general_prediction`: the active feature set's primary
    (decode-target) channel is drawn as the ground truth, and train / val / test
    decoded outputs from the walk-forward schedule are overlaid with optional
    central credible intervals from the ``netw1_outpop_*_dist.bin`` files.

    Parameters
    ----------
    paramfile : str
        Multislice ``.par`` file for the completed run.
    ticker : str
        Symbol to plot (e.g. ``"NVDA"``).
    ax : matplotlib.axes.Axes or None
        Axes to draw on. When ``None``, a new figure is created (unless
        ``figno <= 0``, which skips drawing entirely).
    figsize : tuple
        Figure size when ``ax`` is ``None`` and drawing is enabled.
    figno : int
        Matplotlib figure number for a newly created figure; ``<= 0`` skips
        drawing (RMSE tuple is still computed).
    plot_range : length-2 tuple or None
        Optional ``xlim`` after plotting (feature index or date depending on
        ``dates``).
    confidence_levels : None, float, or iterable of floats
        Same convention as :func:`plot_general_prediction`.
    dates : array-like or None
        Optional calendar dates aligned with the ticker's feature rows (length
        ``feats.shape[0]``). When set, the x-axis shows dates instead of local
        feature indices.
    series : dict or None
        Precomputed output of :func:`stitch_multislice_ticker`.
    savepath : str or None
        If set, saves ``multislice_prediction_<TICKER>.png`` under this path.
    save_as : str or None
        If set, saves the figure to this exact file path (overrides ``savepath``).
    phases : None, str, int, or iterable
        Which walk-forward phases to draw. ``None`` shows train, validation,
        and test. Pass ``"test"`` (or ``2``) to show only the unseen/test
        segments.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure containing the plot, or ``None`` when drawing is skipped.
    ax : matplotlib.axes.Axes or None
        The axes containing the plot, or ``None`` when drawing is skipped.
    tuple of 4 floats
        ``(rmse_train, rmse_validation, rmse_unseen, rmse_persistence)``.
        RMSE entries are ``nan`` for phases not selected; persistence is
        computed over the plotted points only.
    """
    if series is None:
        series = stitch_multislice_ticker(
            paramfile, ticker, rundir=rundir)

    import FeatureGeneration

    offs = series["offs"]
    feats = series["feats"]
    centers = series["centers"]
    cfg = series.get("cfg", series["plan"]["cfg"])
    pcol = series.get("primary_col", 3)
    truth_display = FeatureGeneration.delog_primary(cfg, feats[:, pcol])
    x_pred = series["x_feat"]
    truth_pred = series["truth"]
    pred = series["pred"]
    dists = series["dists"]
    phase_ids = series["phase"]
    block_ids = series["block"]
    ticker = series["ticker"]

    if dates is not None:
        dates = np.asarray(dates)
        if len(dates) != feats.shape[0]:
            raise ValueError(
                f"dates length {len(dates)} != feats rows {feats.shape[0]}. "
                f"Rebuild dates from the stitched series via "
                f"Display.ticker_feat_dates(df['Date'], series) "
                f"(notebook per_ticker may be stale after param changes).")
        x_truth = dates
        x_pred_plot = dates[x_pred]
    else:
        x_truth = np.arange(feats.shape[0])
        x_pred_plot = x_pred

    phase_names = ("train", "validation", "testing")
    phase_colors = ("#4C72B0", "#DD8452", "#55A868")
    phase_filter = _normalize_phase_filter(phases)
    show_mask = np.isin(phase_ids, list(phase_filter))

    rmses = []
    for phase_idx in range(3):
        m = (phase_ids == phase_idx) & show_mask
        if m.any():
            rmses.append(float(Utils.RMSE(truth_pred[m], pred[m])))
        else:
            rmses.append(float("nan"))
    rmse_train, rmse_validation, rmse_unseen = rmses
    if show_mask.any():
        x_show = x_pred[show_mask]
        rmse_persistence = float(np.sqrt(np.mean(
            (truth_pred[show_mask]
             - FeatureGeneration.delog_primary(cfg, feats[x_show - offs, pcol])) ** 2)))
    else:
        rmse_persistence = float("nan")

    fig = None
    if ax is not None:
        fig = ax.figure
    elif figno > 0:
        fig, ax = plt.subplots(figsize=figsize, num=figno)

    if ax is not None:
        levels = _normalize_confidence_levels(confidence_levels)
        if levels and centers is None:
            warnings.warn(
                "confidence_levels requested but centers is None; skipping bands.",
                UserWarning, stacklevel=2)
            levels = []

        ax.plot(x_truth, truth_display, color="0.35", lw=0.9, label="timeseries",
                zorder=1)

        for phase_idx, (name, color) in enumerate(zip(phase_names, phase_colors)):
            if phase_idx not in phase_filter:
                continue
            m = phase_ids == phase_idx
            if not np.any(m):
                continue
            _plot_disjoint_segments(
                x_pred_plot[m], pred[m], dists[m], centers, levels,
                color=color, label=name, segment_ids=block_ids[m], ax=ax,
                cfg=cfg,
            )

        if plot_range is not None:
            if len(plot_range) != 2:
                raise ValueError(
                    "plot_range must be None or a length-2 tuple (left, right)")
            xa, xb = float(plot_range[0]), float(plot_range[1])
            if xa > xb:
                raise ValueError("plot_range must have left <= right")
            ax.set_xlim(xa, xb)

        xlabel = "date" if dates is not None else "feature row index"
        title_parts = [f"{ticker} — multislice target prediction ({offs} steps ahead)"]
        if 0 in phase_filter and np.isfinite(rmse_train):
            title_parts.append(f"rmse_train = {rmse_train:.2e}")
        if 1 in phase_filter and np.isfinite(rmse_validation):
            title_parts.append(f"rmse_validation = {rmse_validation:.2e}")
        if 2 in phase_filter and np.isfinite(rmse_unseen):
            title_parts.append(f"rmse_unseen = {rmse_unseen:.2e}")
        if show_mask.any() and np.isfinite(rmse_persistence):
            title_parts.append(f"rmse_persistence = {rmse_persistence:.2e}")
        ax.set_title("  ".join(title_parts), fontsize=11)
        ax.set_xlabel(xlabel)
        ylabel = ("daily return (ratio)" if cfg.primary_log_encoded
                  else "normalized target (primary) feature")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=11)
        fig.tight_layout()
        _save_fig(fig, savepath, f"multislice_prediction_{ticker}.png",
                  save_as=save_as)

    rmses = (rmse_train, rmse_validation, rmse_unseen, rmse_persistence)
    return fig, ax, rmses


def _nearest_index_1d(x_click, x_values):
    """Index of the closest entry in ``x_values`` to scalar ``x_click``."""
    import matplotlib.dates as mdates
    import pandas as pd

    x_click = float(x_click)
    xv = np.asarray(x_values)
    if xv.size == 0:
        raise ValueError("x_values is empty.")
    sample = xv[0]
    if isinstance(sample, (pd.Timestamp, np.datetime64)):
        xv_num = mdates.date2num(pd.DatetimeIndex(xv))
    elif hasattr(sample, "toordinal"):
        xv_num = mdates.date2num(xv)
    else:
        xv_num = np.asarray(xv, dtype=float)
    return int(np.argmin(np.abs(xv_num - x_click)))


def _as_mpl_date_nums(x_values):
    """Map calendar dates or numeric x to matplotlib date numbers."""
    import matplotlib.dates as mdates
    import pandas as pd

    xv = np.asarray(x_values)
    if xv.size == 0:
        return np.array([], dtype=float)
    sample = xv[0]
    if isinstance(sample, (pd.Timestamp, np.datetime64)) or hasattr(sample, "toordinal"):
        return mdates.date2num(pd.DatetimeIndex(xv))
    return np.asarray(xv, dtype=float)


def _padded_ylim(y_values, *, pad_frac=0.05, min_pad=1e-6):
    """Return ``(lo, hi)`` for finite ``y_values`` with padding."""
    y = np.asarray(y_values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    lo, hi = float(y.min()), float(y.max())
    pad = max((hi - lo) * pad_frac, min_pad)
    return lo - pad, hi + pad


def _plot_disjoint_segments_encoded(x, y, dists, centers, levels, *, color, label,
                                    lw=1.0, zorder=2, segment_ids=None, ax=None):
    """Like :func:`_plot_disjoint_segments` but bands stay in encoded (log) units."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return

    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        groups = [(segment_ids == sid) for sid in np.unique(segment_ids)]
    else:
        groups = []
        start = 0
        for i in range(1, len(x)):
            if x[i] != x[i - 1] + 1:
                groups.append(slice(start, i))
                start = i
        groups.append(slice(start, len(x)))

    levels = _normalize_confidence_levels(levels)
    for i, seg in enumerate(groups):
        if isinstance(seg, slice):
            m = np.zeros(len(x), dtype=bool)
            m[seg] = True
        else:
            m = seg
        if not np.any(m):
            continue
        lbl = label if i == 0 else "_nolegend_"
        ax.plot(x[m], y[m], color=color, lw=lw, label=lbl, zorder=zorder)
        if dists is not None and levels:
            dists_seg = np.asarray(dists[m], dtype=float)
            n = len(levels)
            for j, level in enumerate(levels):
                alpha = 0.25 + 0.20 * (j / max(n - 1, 1))
                lo, hi = _central_band_bounds(dists_seg, centers, level)
                ax.fill_between(x[m], lo, hi, color=color, alpha=alpha,
                                linewidth=0, zorder=1.5)


def plot_multislice_explorer(paramfile, ticker, *, result, series=None,
                             dates=None, close_prices=None, step=None,
                             phases=None, confidence_levels=(0.9, 0.99),
                             plot_range=None, figsize=(14, 10), figno=3,
                             interactive=True, savepath=None, save_as=None,
                             rundir=None):
    """
    Three- or four-panel explorer: close price, encoded return with uncertainty
    bands, and the output pmf at a selected scored step (quantile bins plus,
    when ``binmode`` is ``rbf-quantile``, a second pmf over the linear encoded
    range with variable bin widths).

    Click a point on the price or return panel to update the distribution when
    an interactive matplotlib backend is active (``%matplotlib widget`` with
    *ipympl*, or ``%matplotlib notebook``). The selected step is always marked
    with vertical guides on the upper panels.

    Parameters
    ----------
    paramfile : str
        Multislice ``.par`` file for the completed run.
    ticker : str
        Symbol to plot.
    series : dict or None
        Precomputed output of :func:`stitch_multislice_ticker`.
    dates : array-like
        Calendar dates aligned with ``series['feats']`` (e.g. from
        :func:`ticker_feat_dates`).
    close_prices : array-like or None
        Adjusted closes aligned with ``dates``. When ``None``, reloads the
        wide CSV named in the parameter file.
    step : int or None
        Initial stitched step index. Defaults to the midpoint of the test phase.
    phases : phase filter
        Same convention as :func:`plot_multislice_prediction`.
    confidence_levels : iterable of floats
        Central credible levels for the return panel (encoded units).
    result : dict
        Output of :func:`Mains.analyze_multislice` with ``metrics`` including
        ``'var'``. Supplies ``result['var'][ticker]`` and ``result['var_level']``
        for the pmf VaR marker.
    plot_range : length-2 tuple or None
        Optional ``(left, right)`` x-limits for the upper panels (dates or
        feature indices). ``None`` zooms to the selected phases' prediction
        span with a small margin.
    figsize : tuple
    figno : int
    interactive : bool
        Attach click handlers when the active backend supports them.
    savepath, save_as : optional figure save targets.

    Returns
    -------
    state : dict with ``fig``, ``axes``, ``step``, ``info``, ``select_step``,
    and ``cid`` (click connection id, or ``None``).
    """
    import FeatureGeneration
    import pandas as pd

    if series is None:
        series = stitch_multislice_ticker(
            paramfile, ticker, rundir=rundir)
    if dates is None:
        raise ValueError("dates= is required (use ticker_feat_dates).")

    dates = np.asarray(dates)
    feats = series["feats"]
    if len(dates) != feats.shape[0]:
        raise ValueError(
            f"dates length {len(dates)} != feats rows {feats.shape[0]}.")

    cfg = series.get("cfg", series["plan"]["cfg"])
    pcol = series.get("primary_col", series["plan"]["primary_col"])
    centers = np.asarray(series["centers"], dtype=float)
    ticker = series["ticker"]
    phase_filter = _normalize_phase_filter(phases)

    var_series, var_level = _resolve_var_from_result(result, ticker)
    if len(var_series) != len(series["pred"]):
        raise ValueError(
            f"result['var']['{ticker}'] length {len(var_series)} != stitched "
            f"steps {len(series['pred'])}.")

    if close_prices is None:
        ts_file = series["plan"].get("timeseries_file")
        if not ts_file:
            params = Utils.load_params(paramfile)
            ts_file = params["timeseries_file"]
        df = pd.read_csv(ts_file, parse_dates=["Date"]).set_index("Date")
        close_prices = df[ticker].reindex(pd.DatetimeIndex(dates)).to_numpy()
    else:
        close_prices = np.asarray(close_prices, dtype=float)
        if len(close_prices) != len(dates):
            raise ValueError(
                f"close_prices length {len(close_prices)} != dates {len(dates)}.")

    if step is None:
        test_steps = np.where(series["phase"] == 2)[0]
        if test_steps.size:
            step = int(test_steps[len(test_steps) // 2])
        else:
            step = len(series["pred"]) // 2
    step = resolve_multislice_step(series, step=step)

    x_pred = series["x_feat"]
    x_pred_plot = dates[x_pred]
    truth_enc = np.asarray(feats[:, pcol], dtype=float)
    pred_enc = (np.log(np.asarray(series["pred"], dtype=float))
                if cfg.primary_log_encoded
                else np.asarray(series["pred"], dtype=float))
    phase_ids = series["phase"]
    block_ids = series["block"]
    dists = series["dists"]

    phase_names = ("train", "validation", "testing")
    phase_colors = ("#4C72B0", "#DD8452", "#55A868")
    levels = _normalize_confidence_levels(confidence_levels)

    if figno is not None and plt.fignum_exists(figno):
        plt.close(figno)
    use_dual_dist = _resolve_dist_xscale(cfg, None) == "quantile"
    fig = plt.figure(figsize=figsize, num=figno, layout="constrained")
    if use_dual_dist:
        gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 2.2, 1.4, 1.4], hspace=0.08)
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.2, 1.6], hspace=0.08)
    ax_price = fig.add_subplot(gs[0])
    ax_return = fig.add_subplot(gs[1], sharex=ax_price)
    ax_dist = fig.add_subplot(gs[2])  # quantile / default pmf x-axis
    ax_dist_linear = fig.add_subplot(gs[3], sharey=ax_dist) if use_dual_dist else None

    ax_price.plot(dates, close_prices, color="0.2", lw=0.9, label="close")
    ax_price.set_ylabel("price")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(alpha=0.25)
    ax_price.set_title(
        f"{ticker} — price, log return + uncertainty, output pmf  "
        f"(click upper panels to pick a step)")

    ax_return.plot(dates, truth_enc, color="0.35", lw=0.9,
                   label="truth (log return)" if cfg.primary_log_encoded
                   else "truth", zorder=1)
    for phase_idx, (name, color) in enumerate(zip(phase_names, phase_colors)):
        if phase_idx not in phase_filter:
            continue
        m = phase_ids == phase_idx
        if not np.any(m):
            continue
        _plot_disjoint_segments_encoded(
            x_pred_plot[m], pred_enc[m], dists[m], centers, levels,
            color=color, label=name, segment_ids=block_ids[m], ax=ax_return)

    ylabel = ("log return" if cfg.primary_log_encoded
              else "normalized target (primary) feature")
    ax_return.set_ylabel(ylabel)
    ax_return.legend(loc="upper left", fontsize=9, ncol=3)
    ax_return.grid(alpha=0.25)

    show_mask = np.isin(phase_ids, list(phase_filter))
    if plot_range is not None:
        if len(plot_range) != 2:
            raise ValueError("plot_range must be None or a length-2 tuple")
        xa, xb = float(plot_range[0]), float(plot_range[1])
        if xa > xb:
            raise ValueError("plot_range must have left <= right")
        xlim = (xa, xb)
    else:
        x_vis = np.asarray(x_pred_plot[show_mask] if show_mask.any()
                           else x_pred_plot)
        x_num = _as_mpl_date_nums(x_vis)
        if x_num.size == 0:
            xlim = None
        else:
            lo, hi = float(x_num.min()), float(x_num.max())
            pad = max((hi - lo) * 0.02, 5.0)
            xlim = lo - pad, hi + pad

    if xlim is not None:
        ax_price.set_xlim(xlim)
        date_nums = _as_mpl_date_nums(dates)
        in_range = (date_nums >= xlim[0]) & (date_nums <= xlim[1])
        price_ylim = _padded_ylim(close_prices[in_range])
        if price_ylim is not None:
            ax_price.set_ylim(price_ylim)

        ret_y = list(truth_enc[in_range])
        if show_mask.any():
            ret_y.extend(pred_enc[show_mask])
            if levels:
                for level in levels:
                    lo, hi = _central_band_bounds(dists[show_mask], centers, level)
                    ret_y.extend(lo)
                    ret_y.extend(hi)
        ret_ylim = _padded_ylim(ret_y)
        if ret_ylim is not None:
            ax_return.set_ylim(ret_ylim)

    marker_artists = {
        "v_price": ax_price.axvline(0, color="#C44E52", lw=1.2, ls="--", zorder=5),
        "v_return": ax_return.axvline(0, color="#C44E52", lw=1.2, ls="--", zorder=5),
        "pt_return": ax_return.plot([], [], "o", color="#C44E52", ms=7, zorder=6)[0],
        "pt_price": ax_price.plot([], [], "o", color="#C44E52", ms=7, zorder=6)[0],
    }

    def _draw_distribution(idx):
        ax_dist.cla()
        if ax_dist_linear is not None:
            ax_dist_linear.cla()
        _, _, info = plot_multislice_step_distribution(
            series, step=idx, dates=dates, ax=ax_dist, ax_linear=ax_dist_linear,
            log_scale=True, dual_xscale=use_dual_dist, var_level=var_level,
            var_value=float(var_series[idx]), layout=False)
        ax_dist.set_title(ax_dist.get_title(), fontsize=10)
        dist_axes = (ax_dist,) if ax_dist_linear is None else (ax_dist, ax_dist_linear)
        for dist_ax in dist_axes:
            dist_ax.relim()
            dist_ax.autoscale_view(scalex=False, scaley=True)
        return info

    def select_step(idx, *, draw=True):
        idx = int(resolve_multislice_step(series, step=idx))
        x_sel = x_pred_plot[idx]
        y_ret = float(pred_enc[idx])
        price_row = int(series["x_feat"][idx])
        y_price = float(close_prices[price_row]) if np.isfinite(
            close_prices[price_row]) else np.nan

        marker_artists["v_price"].set_xdata([x_sel, x_sel])
        marker_artists["v_return"].set_xdata([x_sel, x_sel])
        marker_artists["pt_return"].set_data([x_sel], [y_ret])
        marker_artists["pt_price"].set_data([x_sel], [y_price] if np.isfinite(y_price) else [])

        info = _draw_distribution(idx) if draw else None
        if draw:
            fig.canvas.draw_idle()
        return idx, info

    step, dist_info = select_step(step)

    cid = None
    if interactive:
        backend = plt.get_backend().lower()
        non_interactive = {"agg", "cairo", "pdf", "ps", "svg", "template"}

        def _on_click(event):
            if event.inaxes not in (ax_price, ax_return):
                return
            if event.xdata is None:
                return
            idx = _nearest_index_1d(event.xdata, x_pred_plot)
            select_step(idx)

        if backend not in non_interactive and "inline" not in backend:
            cid = fig.canvas.mpl_connect("button_press_event", _on_click)

    ax_return.set_xlabel("date")
    plt.setp(ax_price.get_xticklabels(), visible=False)
    for ax in (ax_return,):
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")
    _save_fig(fig, savepath, f"multislice_explorer_{ticker}.png", save_as=save_as)

    return {
        "fig": fig,
        "axes": ((ax_price, ax_return, ax_dist, ax_dist_linear)
                 if ax_dist_linear is not None
                 else (ax_price, ax_return, ax_dist)),
        "series": series,
        "step": step,
        "info": dist_info,
        "select_step": select_step,
        "cid": cid,
    }


def display_multislice_explorer(state):
    """
    Show the explorer figure exactly once.

    With an interactive backend (``%matplotlib widget`` + *ipympl*), the
    figure is left registered with pyplot so Jupyter renders the live canvas
    once at cell end — do not call ``display(fig)`` again or it duplicates.

    For inline backends, attaches an :mod:`ipywidgets` step slider (or a
    single static ``display``) after unregistering the figure from pyplot so
    IPython does not also auto-render a snapshot.
    """
    from IPython.display import display

    fig = state["fig"]
    if state.get("cid") is not None:
        return

    import matplotlib._pylab_helpers as gcf
    gcf.Gcf.figs.pop(fig.number, None)

    try:
        import ipywidgets as widgets
    except ImportError:
        display(fig)
        print(
            "For click or slider interactivity install one of:\n"
            "  %pip install ipympl      # then re-run with %matplotlib widget\n"
            "  %pip install ipywidgets  # step slider fallback")
        return

    n_steps = len(state["series"]["pred"])
    out = widgets.Output()

    def _redraw():
        with out:
            out.clear_output(wait=True)
            display(fig)

    slider = widgets.IntSlider(
        value=int(state["step"]),
        min=0,
        max=max(0, n_steps - 1),
        description="step",
        continuous_update=False,
    )

    def _on_slider(change):
        state["select_step"](change["new"])
        _redraw()

    slider.observe(_on_slider, names="value")
    with out:
        display(fig)
    display(widgets.VBox([slider, out]))


def _plot_disjoint_segments(x, y, dists, centers, levels, *, color, label, lw=1.0,
                            zorder=2, segment_ids=None, ax=None, cfg=None):
    """
    Plot ``y`` vs ``x`` without connecting across segment boundaries.

    When ``segment_ids`` is given (e.g. block index per point), one line /
    band group is drawn per unique id. Otherwise splits wherever ``x`` is not
    consecutive (integer feature indices or evenly spaced dates).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return

    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        groups = [(segment_ids == sid) for sid in np.unique(segment_ids)]
    else:
        groups = []
        start = 0
        for i in range(1, len(x)):
            if x[i] != x[i - 1] + 1:
                groups.append(slice(start, i))
                start = i
        groups.append(slice(start, len(x)))

    for i, seg in enumerate(groups):
        if isinstance(seg, slice):
            m = np.zeros(len(x), dtype=bool)
            m[seg] = True
        else:
            m = seg
        if not np.any(m):
            continue
        lbl = label if i == 0 else "_nolegend_"
        plot_fn = ax.plot if ax is not None else plt.plot
        plot_fn(x[m], y[m], color=color, lw=lw, label=lbl, zorder=zorder)
        if dists is not None:
            _plot_prediction_bands(x[m], dists[m], centers, levels, color, ax=ax,
                                   cfg=cfg)


def _fixed_params_str(sweep_spec):
    """Build a summary string of parameters that were not varied."""
    fixed = {k: v[0] for k, v in sweep_spec.items() if len(v) == 1}
    if not fixed:
        return ""
    return "Fixed: " + ", ".join(f"{k}={v}" for k, v in fixed.items())


def _save_fig(fig, savepath=None, filename=None, save_as=None):
    """Save figure to ``save_as`` or ``savepath/filename``."""
    if save_as is not None:
        full = save_as
    elif savepath is not None and filename is not None:
        os.makedirs(savepath, exist_ok=True)
        full = os.path.join(savepath, filename)
    else:
        return
    parent = os.path.dirname(os.path.abspath(full))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(full, dpi=150, bbox_inches="tight")
    print(f"  Saved: {full}")


def plot_sweep_heatmaps(results, sweep_spec, metric_label="RMSE (unseen)", cmap=None,
                        savepath=None, save_as=None):
    """
    Plot 2D heatmaps of sweep results.

    Parameters
    ----------
    results : dict mapping (param_name, ...) tuples of param values -> float
        Keys are tuples of values in the order of sweep_spec keys.
    sweep_spec : dict mapping param_name -> list of values (only the varied ones,
        i.e. those with len > 1).
    metric_label : str
    cmap : str
    savepath : str or None
        If set, figures are saved to this directory.
    save_as : str or None
        If set, saves the figure to this exact file path (overrides ``savepath``).
    """
    if cmap is None:
        cmap = _CMAP_GREEN_RED

    varied = {k: v for k, v in sweep_spec.items() if len(v) > 1}
    n_varied = len(varied)
    if n_varied == 0:
        print("Nothing to plot — no parameter was varied.")
        return
    if n_varied > 3:
        print("Plotting only supported for up to 3 varied parameters.")
        return

    # Sort parameters by resolution (number of values), descending
    sorted_params = sorted(varied.items(), key=lambda kv: len(kv[1]), reverse=True)
    param_names = [kv[0] for kv in sorted_params]
    param_vals = [kv[1] for kv in sorted_params]

    # All sweep parameter names in original order (for indexing into result keys)
    all_names = list(sweep_spec.keys())

    if n_varied == 1:
        _plot_1d(results, all_names, param_names, param_vals, metric_label,
                 savepath, sweep_spec, save_as=save_as)
    elif n_varied == 2:
        _plot_2d(results, all_names, param_names, param_vals, metric_label, cmap,
                 savepath, sweep_spec, save_as=save_as)
    else:
        _plot_2d_with_slices(results, all_names, param_names, param_vals,
                             metric_label, cmap, savepath, sweep_spec,
                             save_as=save_as)

def _plot_1d(results, all_names, param_names, param_vals, metric_label, savepath,
             sweep_spec, save_as=None):
    """Line plot for a single swept parameter."""
    pname = param_names[0]
    vals = param_vals[0]
    pidx = all_names.index(pname)

    y = []
    for v in vals:
        for key, err in results.items():
            if key[pidx] == v:
                y.append(err)
                break

    fig, ax = plt.subplots(figsize=(max(6, len(vals) * 0.8), 4.5))
    ax.plot(vals, y, "o-")
    ax.set_xlabel(pname)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs {pname}")
    fixed = _fixed_params_str(sweep_spec)
    if fixed:
        fig.suptitle(fixed, fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, savepath, f"sweep_{pname}.png", save_as=save_as)

def _plot_2d(results, all_names, param_names, param_vals, metric_label, cmap,
             savepath, sweep_spec, save_as=None):
    """Single heatmap for two swept parameters."""
    xname, yname = param_names[0], param_names[1]
    xvals, yvals = param_vals[0], param_vals[1]
    xidx, yidx = all_names.index(xname), all_names.index(yname)

    grid = np.full((len(yvals), len(xvals)), np.nan)
    for key, err in results.items():
        xi = xvals.index(key[xidx])
        yi = yvals.index(key[yidx])
        grid[yi, xi] = err

    cell_w, cell_h = 0.9, 0.7
    fig_w = max(6, len(xvals) * cell_w + 2)
    fig_h = max(4, len(yvals) * cell_h + 2.5)
    fig = plt.figure(figsize=(fig_w, fig_h))

    fixed = _fixed_params_str(sweep_spec)
    if fixed:
        fig.suptitle(fixed, fontsize=9, y=0.98)

    ax = fig.add_axes([0.12, 0.12, 0.65, 0.72])
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([str(v) for v in xvals], rotation=45, ha="right")
    ax.set_yticks(range(len(yvals)))
    ax.set_yticklabels([str(v) for v in yvals])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(metric_label)
    cbar_ax = fig.add_axes([0.82, 0.12, 0.03, 0.72])
    fig.colorbar(im, cax=cbar_ax, label=metric_label)
    _save_fig(fig, savepath, f"sweep_{xname}_vs_{yname}.png", save_as=save_as)


def _plot_2d_with_slices(results, all_names, param_names, param_vals, metric_label,
                         cmap, savepath, sweep_spec, save_as=None):
    """
    One subplot per value of the least-resolved (3rd) parameter.
    Each subplot is a heatmap over the two most-resolved parameters.
    """
    xname, yname, zname = param_names[0], param_names[1], param_names[2]
    xvals, yvals, zvals = param_vals[0], param_vals[1], param_vals[2]
    xidx = all_names.index(xname)
    yidx = all_names.index(yname)
    zidx = all_names.index(zname)

    n_slices = len(zvals)
    cell_w, cell_h = 0.9, 0.7
    sub_w = max(5, len(xvals) * cell_w + 1.5)
    sub_h = max(3.5, len(yvals) * cell_h + 1.5)

    # Reserve right margin for the colorbar
    cbar_width_in = 1.2
    total_w = sub_w * n_slices + cbar_width_in
    fig = plt.figure(figsize=(total_w, sub_h + 0.5))

    fixed = _fixed_params_str(sweep_spec)
    if fixed:
        fig.suptitle(fixed, fontsize=9, y=0.99)

    # Fraction of figure occupied by subplots vs colorbar
    plot_right = 1.0 - cbar_width_in / total_w
    gs = fig.add_gridspec(1, n_slices, left=0.08, right=plot_right, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_slices)]

    vmin = min(results.values())
    vmax = max(results.values())

    for si, zv in enumerate(zvals):
        grid = np.full((len(yvals), len(xvals)), np.nan)
        for key, err in results.items():
            if key[zidx] != zv:
                continue
            xi = xvals.index(key[xidx])
            yi = yvals.index(key[yidx])
            grid[yi, xi] = err

        ax = axes[si]
        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(xvals)))
        ax.set_xticklabels([str(v) for v in xvals], rotation=45, ha="right")
        ax.set_yticks(range(len(yvals)))
        ax.set_yticklabels([str(v) for v in yvals])
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"{zname} = {zv}")

    cbar_ax = fig.add_axes([plot_right + 0.02, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=metric_label)
    _save_fig(fig, savepath, f"sweep_{xname}_vs_{yname}_by_{zname}.png",
              save_as=save_as)


def plot_multislice_rmse(result, *, figno=1, savepath=None, vmin=None, vmax=None,
                         cmap=None, ticker_label_fontsize=8):
    """
    Visualize the [n_tickers, 3, n_blocks] RMSE cube returned by
    :func:`Mains.multislice`.

    Two stacked rows of subplots, one column per phase (train, val, test):

    * Row 1 - heatmaps of RMSE per (ticker, block). Cells where the ticker
      was not active in that block are masked and rendered as grey.
    * Row 2 - one line per ticker (RMSE vs block index), plus a dashed black
      persistence baseline (mean across active tickers per block) and a
      thicker solid line showing the per-block mean RMSE across active
      tickers (i.e. the "model" baseline).

    The colour scale is shared across the three phase heatmaps so visual
    comparisons across phases are meaningful. Pass ``vmin``/``vmax`` to
    override (e.g. clip outliers).

    Parameters
    ----------
    result : dict
        The dict returned by ``Mains.multislice()``; expects keys ``rmse``,
        ``valid``, ``persistence``, ``tickers``, ``block_active``.
    figno : int
        Matplotlib figure number; <= 0 disables drawing (the function still
        returns the assembled figure object if drawn).
    savepath : str or None
        If set, the figure is also saved into this directory as
        ``multislice_rmse.png``.
    vmin, vmax : float or None
        Colourmap limits shared across the three heatmaps.
    cmap : str or matplotlib colormap or None
        Defaults to ``_CMAP_GREEN_RED`` (green for low RMSE -> red for high).
    ticker_label_fontsize : int
        Font size of the ticker labels on the y-axis of the heatmaps.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if figno <= 0:
        return None

    rmse = np.asarray(result["rmse"], dtype=float)
    valid = np.asarray(result["valid"], dtype=bool)
    persistence = np.asarray(result["persistence"], dtype=float)
    tickers = list(result["tickers"])
    n_tickers, n_phases, n_blocks = rmse.shape
    assert n_phases == 3, "rmse must be [T, 3, B]"

    if cmap is None:
        cmap = _CMAP_GREEN_RED

    masked = np.ma.masked_where(~valid, rmse)
    if vmin is None or vmax is None:
        finite = rmse[np.isfinite(rmse)]
        if finite.size == 0:
            vmin_eff, vmax_eff = 0.0, 1.0
        else:
            vmin_eff = float(np.nanmin(finite)) if vmin is None else float(vmin)
            vmax_eff = float(np.nanmax(finite)) if vmax is None else float(vmax)
            if vmax_eff <= vmin_eff:
                vmax_eff = vmin_eff + 1e-12
    else:
        vmin_eff, vmax_eff = float(vmin), float(vmax)

    phase_names = ("train", "val", "test")
    block_x = np.arange(n_blocks)

    # One column per phase + a thin colorbar column. We size each phase column
    # roughly 0.10 inch per block, clamped so the figure never gets absurdly
    # wide for long schedules; the user can override via plt.figure(figno,
    # figsize=...) before calling if they want exact control.
    per_phase_w = float(np.clip(0.10 * n_blocks, 3.5, 7.5))
    fig_w = per_phase_w * n_phases + 1.2
    fig_h = float(np.clip(0.18 * n_tickers + 4.0, 5.0, 11.0))
    fig = plt.figure(figno, figsize=(fig_w, fig_h), constrained_layout=True)
    fig.clf()
    gs = fig.add_gridspec(2, n_phases + 1,
                          width_ratios=[1.0] * n_phases + [0.04],
                          height_ratios=[1.0, 0.9])
    heat_axes = [fig.add_subplot(gs[0, j]) for j in range(n_phases)]
    line_axes = [fig.add_subplot(gs[1, j]) for j in range(n_phases)]
    cbar_ax = fig.add_subplot(gs[0, n_phases])

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    try:
        cmap_obj.set_bad(color="#bbbbbb")
    except AttributeError:
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_bad(color="#bbbbbb")

    im = None
    for p, ax in enumerate(heat_axes):
        im = ax.imshow(masked[:, p, :], aspect="auto", origin="upper",
                       cmap=cmap_obj, vmin=vmin_eff, vmax=vmax_eff,
                       interpolation="nearest")
        ax.set_title(f"{phase_names[p]} RMSE")
        ax.set_xlabel("block index")
        ax.set_yticks(np.arange(n_tickers))
        ax.set_yticklabels(tickers, fontsize=ticker_label_fontsize)
        if p == 0:
            ax.set_ylabel("ticker")
        if n_blocks <= 30:
            ax.set_xticks(block_x)
    if im is not None:
        fig.colorbar(im, cax=cbar_ax, label="RMSE")
    else:
        cbar_ax.set_visible(False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_rmse_per_block = np.nanmean(np.where(valid, rmse, np.nan), axis=0)
        mean_pers_per_block = np.nanmean(
            np.where(valid, persistence, np.nan), axis=0)

    for p, ax in enumerate(line_axes):
        for t in range(n_tickers):
            y = np.where(valid[t, p, :], rmse[t, p, :], np.nan)
            if np.isfinite(y).any():
                ax.plot(block_x, y, lw=0.8, alpha=0.6, label=tickers[t])
        ax.plot(block_x, mean_rmse_per_block[p, :], color="black", lw=2.0,
                label="mean RMSE (active tickers)")
        ax.plot(block_x, mean_pers_per_block[p, :], color="black", lw=1.2,
                ls="--", label="persistence")
        ax.set_title(f"{phase_names[p]} RMSE per block")
        ax.set_xlabel("block index")
        if p == 0:
            ax.set_ylabel("RMSE")
        ax.grid(True, ls=":", alpha=0.5)
        if p == n_phases - 1 and n_tickers <= 12:
            ax.legend(fontsize=7, loc="best")
        elif p == n_phases - 1:
            ax.legend(["mean RMSE (active tickers)", "persistence"],
                      fontsize=7, loc="best")

    block_step = result.get("block_step")
    trnpat = result.get("trnpat")
    vanpat = result.get("vanpat")
    tenpat = result.get("tenpat")
    if block_step is not None:
        fig.suptitle(
            f"Mains.multislice() RMSE  "
            f"(n_tickers={n_tickers}, n_blocks={n_blocks}, "
            f"trnpat={trnpat}, vanpat={vanpat}, tenpat={tenpat}, "
            f"block_step={block_step})",
            fontsize=11)

    _save_fig(fig, savepath, "multislice_rmse.png")
    return fig


def plot_ltsl_delay_ladder(
    paramfile=None,
    *,
    K=None,
    K_mode="logarithmic",
    taumin=None,
    taumax=None,
    figsize=(14, 3.5),
    verbose=False,
):
    """
    LTSL delay ladder visualization (matches TDPop / ltslmain).

    Pass either ``paramfile`` (path to a .par with K, taumin, taumax, and
    optionally K_mode) **or** the explicit parameters ``K``, ``K_mode``,
    ``taumin``, ``taumax``.

    Parameters
    ----------
    paramfile : str or None
        Path to parameter file. If set, ladder values are read from this file
        and ``K``, ``K_mode``, ``taumin``, ``taumax`` must not be passed.
    K, taumin, taumax : int/float or None
        Required when ``paramfile`` is None.
    K_mode : str
        ``\"logarithmic\"`` or ``\"linear\"`` (only used with explicit K, …).
    figsize : tuple
        Figure size passed to ``plt.subplots``.
    verbose : bool
        If True, print K_mode, c, and endpoint taus like the notebook cell.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if paramfile is not None:
        if any(v is not None for v in (K, taumin, taumax)):
            raise ValueError(
                "Pass either paramfile, or (K, taumin, taumax), not both."
            )
        tau, c = Utils.ltsl_tau_ladder(paramfile)
        K_mode_used = str(Utils.load_params(paramfile).get("K_mode", "logarithmic"))
    else:
        if K is None or taumin is None or taumax is None:
            raise ValueError(
                "Without paramfile, K, taumin, and taumax are required."
            )
        tau, c = Utils._ltsl_tau_ladder_values(K, taumin, taumax, K_mode)
        K_mode_used = str(K_mode)

    K = len(tau)
    k = np.arange(K)
    t_lo = float(tau[0])
    t_hi = float(tau[-1])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"LTSL delay ladder  (K_mode = {K_mode_used})", fontsize=13)

    # taumin / taumax: distinct from ladder line (green / orange)
    _c_min, _c_max = "#1a9850", "#e6550d"

    axes[0].semilogy(k, tau, "o-", label=r"$\tau_k$")
    axes[0].axhline(t_lo, color=_c_min, ls="--", lw=1.25, label=f"taumin = {t_lo:g}")
    axes[0].axhline(t_hi, color=_c_max, ls="--", lw=1.25, label=f"taumax = {t_hi:g}")
    axes[0].set_xlabel("ladder index k")
    axes[0].set_ylabel("τ_k")
    axes[0].set_title("τ_k (semilog)")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(k, tau, "o-", label=r"$\tau_k$")
    axes[1].axhline(t_lo, color=_c_min, ls="--", lw=1.25, label=f"taumin = {t_lo:g}")
    axes[1].axhline(t_hi, color=_c_max, ls="--", lw=1.25, label=f"taumax = {t_hi:g}")
    axes[1].set_xlabel("ladder index k")
    axes[1].set_ylabel("τ_k")
    axes[1].set_title("τ_k (linear)")
    axes[1].legend(loc="best", fontsize=8)

    ratios = tau[1:] / tau[:-1]
    axes[2].plot(k[:-1], ratios, "o-", label=r"τ_{k+1} / τ_k")
    if c is not None:
        axes[2].axhline(c, color="C1", ls="--", label=f"c = {c:.6g}")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel(r"τ_{k+1} / τ_k")
    axes[2].legend()
    axes[2].set_title("consecutive ratio")
    fig.tight_layout()

    if verbose:
        print(
            f"K_mode={K_mode_used}, c={c!r}, τ_0={tau[0]}, τ_{K - 1}={tau[-1]}"
        )
    return fig
