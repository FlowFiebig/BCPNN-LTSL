"""
Post-hoc multislice metric computation from simulator output artifacts.
"""
import warnings

import numpy as np

import Utils
import FeatureGeneration
import DeepVaRLosses
from MultislicePlan import multislice_artifact_path

DEEPVAR_LOSS_NAMES = frozenset(DeepVaRLosses.DEEPVAR_LOSS_NAMES)
STITCHED_METRIC_NAMES = frozenset({"hit", "var"}) | DEEPVAR_LOSS_NAMES


def load_decoded_streams(plan, *, rundir=None, load_dists=True) :
    """
    Load and decode tetr / van / teus output streams from ``.bin`` files.

    Returns
    -------
    dict with keys ``tetroutx``, ``vanoutx``, ``teusoutx``, optional
    ``tetroutdists`` / ``vanoutdists`` / ``teusoutdists``, ``expected``,
    ``got``.
    """
    cfg = plan["cfg"]
    mean_nbin = plan["mean_nbin"]
    vanpat = plan["vanpat"]
    offs = plan["offs"]
    rbfbinner = plan["rbfbinner_primary"]

    tetr_path = multislice_artifact_path("tetr_dist", rundir=rundir, plan=plan)
    van_path = multislice_artifact_path("van_dist", rundir=rundir, plan=plan)
    teus_path = multislice_artifact_path("teus_dist", rundir=rundir, plan=plan)

    tetroutdists = Utils.loadbin(tetr_path, mean_nbin)
    tetroutx = FeatureGeneration.delog_primary(
        cfg, rbfbinner.decode(tetroutdists))
    vanoutdists = Utils.loadbin(van_path, mean_nbin)
    vanoutx = FeatureGeneration.delog_primary(
        cfg, rbfbinner.decode(vanoutdists))
    teusoutdists = Utils.loadbin(teus_path, mean_nbin)
    teusoutx = FeatureGeneration.delog_primary(
        cfg, rbfbinner.decode(teusoutdists))

    expected = plan["stream_lengths"]
    got = (len(tetroutx), len(vanoutx), len(teusoutx))
    if got != expected :
        raise RuntimeError(
            f"Decoded stream length mismatch (tetr/van/teus): expected "
            f"{expected} but got {got}. ltslmain likely ran with a different "
            f"schedule than '{plan.get('paramfile', '?')}'.")

    out = {
        "tetroutx" : tetroutx,
        "vanoutx"  : vanoutx if vanpat > offs else None,
        "teusoutx" : teusoutx,
        "expected" : expected,
        "got"      : got,
    }
    if load_dists :
        out["tetroutdists"] = tetroutdists
        out["vanoutdists"] = vanoutdists if vanpat > offs else None
        out["teusoutdists"] = teusoutdists
    return out


def _rmse_cube(plan, streams) :
    """Per-(ticker, phase, block) RMSE and validity mask."""
    cfg = plan["cfg"]
    pcol = plan["primary_col"]
    offs = plan["offs"]
    trnpat = plan["trnpat"]
    vanpat = plan["vanpat"]
    tenpat = plan["tenpat"]
    warmup_pat = plan["warmup_pat"]
    per_ticker = plan["per_ticker"]
    ticker_idx = plan["ticker_idx"]
    block_meta = plan["block_meta"]
    n_tickers = plan["n_tickers"]
    n_blocks = plan["n_blocks"]

    tetroutx = streams["tetroutx"]
    vanoutx = streams["vanoutx"]
    teusoutx = streams["teusoutx"]

    rmse = np.full((n_tickers, 3, n_blocks), np.nan, dtype=float)
    valid = np.zeros((n_tickers, 3, n_blocks), dtype=bool)

    phase_specs = [
        (0, trnpat, 0,               tetroutx, "tetr_block_start"),
        (1, vanpat, trnpat,          vanoutx,  "van_block_start"),
        (2, tenpat, trnpat + vanpat, teusoutx, "teus_block_start"),
    ]

    for b_local, meta in enumerate(block_meta) :
        active = meta["active"]
        for phase_idx, per_t_len, phase_off, stream, block_start_key in phase_specs :
            if stream is None :
                continue
            block_start = meta[block_start_key]
            for k, (label, start_local) in enumerate(active) :
                t_idx = ticker_idx[label]
                w = Utils.multislice_warmup_len(k, warmup_pat)
                scored_len = Utils.multislice_scored_len(
                    k, per_t_len, offs, warmup_pat)
                if scored_len <= 0 :
                    continue
                if warmup_pat > 0 :
                    sl_stream = (block_start
                                 + Utils.multislice_stream_offset(
                                     k, per_t_len, offs, warmup_pat))
                else :
                    sl_stream = block_start + k * per_t_len
                pred = stream[sl_stream : sl_stream + scored_len]
                truth = FeatureGeneration.delog_primary(
                    cfg,
                    per_ticker[label]["feats"][
                        start_local + phase_off + w + offs :
                        start_local + phase_off + per_t_len, pcol
                    ],
                )
                if len(pred) != len(truth) or len(pred) == 0 :
                    continue
                rmse[t_idx, phase_idx, b_local] = float(Utils.RMSE(truth, pred))
                valid[t_idx, phase_idx, b_local] = True

    return {"rmse" : rmse, "valid" : valid}


def _persistence_cube(plan, streams) :
    """Per-(ticker, phase, block) persistence baseline RMSE."""
    cfg = plan["cfg"]
    pcol = plan["primary_col"]
    offs = plan["offs"]
    trnpat = plan["trnpat"]
    vanpat = plan["vanpat"]
    tenpat = plan["tenpat"]
    warmup_pat = plan["warmup_pat"]
    per_ticker = plan["per_ticker"]
    ticker_idx = plan["ticker_idx"]
    block_meta = plan["block_meta"]
    n_tickers = plan["n_tickers"]
    n_blocks = plan["n_blocks"]
    valid = streams.get("valid")

    persistence = np.full((n_tickers, 3, n_blocks), np.nan, dtype=float)

    phase_lens = [
        (0, trnpat, 0),
        (1, vanpat, trnpat),
        (2, tenpat, trnpat + vanpat),
    ]

    for b_local, meta in enumerate(block_meta) :
        active = meta["active"]
        for phase_idx, per_t_len, phase_off in phase_lens :
            for k, (label, start_local) in enumerate(active) :
                t_idx = ticker_idx[label]
                w = Utils.multislice_warmup_len(k, warmup_pat)
                scored_len = Utils.multislice_scored_len(
                    k, per_t_len, offs, warmup_pat)
                if scored_len <= 0 :
                    continue
                if valid is not None and not valid[t_idx, phase_idx, b_local] :
                    continue
                persistence[t_idx, phase_idx, b_local] = Utils.persistence_rmse(
                    FeatureGeneration.delog_primary(
                        cfg, per_ticker[label]["feats"][:, pcol]),
                    offs=offs,
                    start=start_local + phase_off + w,
                    end=start_local + phase_off + per_t_len,
                )

    return {"persistence" : persistence}


def _compute_ticker_hit(series, close_prices) :
    """
    Directional hit indicators on stitched test steps.

    Matches the notebook convention: compare signs of consecutive log returns
    from close prices (truth) and from consecutive decoded predictions.
    """
    test_steps = np.where(series["phase"] == 2)[0]
    if test_steps.size == 0 :
        return np.array([], dtype=np.int8), float("nan")

    test_feat_ixs = series["x_feat"][test_steps]
    pred = np.asarray(series["pred"], dtype=float)
    close_prices = np.asarray(close_prices, dtype=float)

    hit = np.zeros(len(test_steps), dtype=np.int8)
    for i, (ix, tstep) in enumerate(zip(test_feat_ixs, test_steps)) :
        prev_ix = int(ix) - 1
        true_return = np.nan
        if ix > 0 and prev_ix >= 0 :
            c_now, c_prev = close_prices[ix], close_prices[prev_ix]
            if np.isfinite(c_now) and np.isfinite(c_prev) and c_prev > 0 and c_now > 0 :
                true_return = np.log(c_now) - np.log(c_prev)
        predicted_return = np.nan
        if tstep > 0 :
            pred_prev, pred_now = pred[tstep - 1], pred[tstep]
            if (np.isfinite(pred_prev) and np.isfinite(pred_now)
                    and pred_prev > 0 and pred_now > 0) :
                predicted_return = np.log(pred_now) - np.log(pred_prev)
        if np.isfinite(true_return) and np.isfinite(predicted_return) :
            hit[i] = int(np.sign(predicted_return) == np.sign(true_return))
    hit_ratio = float(hit.mean()) if hit.size else float("nan")
    return hit, hit_ratio


def _stitched_step_metrics(plan, streams, *, var_level=0.95, rundir=None,
                           stitched_metrics=None) :
    """Per-ticker stitched hit / var / DeepVaR loss metrics."""
    import Display

    paramfile = plan.get("paramfile")
    if not paramfile :
        raise ValueError("plan must include 'paramfile' for stitched metrics.")

    var_level = float(var_level)
    if not 0.0 < var_level < 1.0 :
        raise ValueError(f"var_level must lie strictly in (0, 1); got {var_level}")

    if stitched_metrics is None :
        stitched_metrics = {"hit", "var"}
    else :
        stitched_metrics = set(stitched_metrics)

    need_hit = "hit" in stitched_metrics
    need_var = bool(stitched_metrics & {"var", *DEEPVAR_LOSS_NAMES})
    need_losses = bool(stitched_metrics & DEEPVAR_LOSS_NAMES)

    n_tickers = plan["n_tickers"]
    ticker_labels = plan["ticker_labels"]
    pcol = plan["primary_col"]
    hit_by_ticker = {}
    var_by_ticker = {}
    hit_ratio = np.full(n_tickers, np.nan, dtype=float)
    quadratic_loss = np.full(n_tickers, np.nan, dtype=float)
    smooth_loss = np.full(n_tickers, np.nan, dtype=float)
    tick_loss = np.full(n_tickers, np.nan, dtype=float)
    firm_loss = np.full(n_tickers, np.nan, dtype=float)
    n_violations = np.full(n_tickers, np.nan, dtype=float)
    violation_rate = np.full(n_tickers, np.nan, dtype=float)

    stream_kwargs = {
        "tetroutx"     : streams["tetroutx"],
        "vanoutx"      : streams["vanoutx"],
        "teusoutx"     : streams["teusoutx"],
        "tetroutdists" : streams.get("tetroutdists"),
        "vanoutdists"  : streams.get("vanoutdists"),
        "teusoutdists" : streams.get("teusoutdists"),
    }

    for ti, ticker in enumerate(ticker_labels) :
        try :
            series = Display.stitch_multislice_ticker(
                paramfile, ticker, plan=plan, rundir=rundir, **stream_kwargs)
        except ValueError :
            continue

        if need_var or need_losses :
            centers = np.asarray(series["centers"], dtype=float)
            var_by_ticker[ticker] = np.array([
                Utils.discrete_pmf_quantile(row, centers, 1.0 - var_level)
                for row in series["dists"]
            ], dtype=float)

        if need_hit :
            close_prices = np.asarray(series["feats"][:, 0], dtype=float)
            hit, hr = _compute_ticker_hit(series, close_prices)
            hit_by_ticker[ticker] = hit
            hit_ratio[ti] = hr

        if need_losses :
            test_steps = np.where(series["phase"] == 2)[0]
            if test_steps.size == 0 :
                continue
            pnl = np.asarray(
                series["feats"][series["x_feat"][test_steps], pcol], dtype=float)
            var_t = var_by_ticker[ticker][test_steps]
            try :
                losses = DeepVaRLosses.aggregate_deepvar_losses(
                    pnl, var_t, var_level)
            except ValueError :
                continue
            quadratic_loss[ti] = losses["quadratic_loss"]
            smooth_loss[ti] = losses["smooth_loss"]
            tick_loss[ti] = losses["tick_loss"]
            firm_loss[ti] = losses["firm_loss"]
            n_violations[ti] = losses["n_violations"]
            violation_rate[ti] = losses["violation_rate"]

    out = {"var_level" : var_level}
    if need_hit :
        out["hit"] = hit_by_ticker
        out["hit_ratio"] = hit_ratio
    if need_var :
        out["var"] = var_by_ticker
    if need_losses :
        out["quadratic_loss"] = quadratic_loss
        out["smooth_loss"] = smooth_loss
        out["tick_loss"] = tick_loss
        out["firm_loss"] = firm_loss
        out["n_violations"] = n_violations
        out["violation_rate"] = violation_rate
    return out


_METRICS = {
    "rmse"         : _rmse_cube,
    "persistence"  : _persistence_cube,
}


def compute_metric_cubes(plan, streams, metrics=("rmse", "persistence"),
                         *, var_level=0.95, rundir=None) :
    """
    Compute requested metric cubes from a plan and decoded output streams.

    ``persistence`` is computed after ``rmse`` when both are requested so the
    validity mask from RMSE can gate persistence entries.
    """
    out = {}
    metric_list = list(metrics)
    if "persistence" in metric_list and "rmse" in metric_list :
        metric_list = [m for m in metric_list if m != "persistence"]
        metric_list.append("persistence")

    requested_stitched = STITCHED_METRIC_NAMES.intersection(metric_list)
    if requested_stitched :
        if streams.get("tetroutdists") is None :
            raise ValueError(
                "Stitched metrics require output distributions; reload streams "
                "with load_dists=True.")
        compute_stitched = set(requested_stitched)
        if compute_stitched & DEEPVAR_LOSS_NAMES :
            compute_stitched.add("var")
        stitched = _stitched_step_metrics(
            plan, streams, var_level=var_level, rundir=rundir,
            stitched_metrics=compute_stitched)
        want_hit = "hit" in requested_stitched
        want_var = "var" in requested_stitched or bool(
            requested_stitched & DEEPVAR_LOSS_NAMES)
        want_losses = bool(requested_stitched & DEEPVAR_LOSS_NAMES)
        if want_hit :
            out["hit"] = stitched["hit"]
            out["hit_ratio"] = stitched["hit_ratio"]
        if want_var :
            out["var"] = stitched["var"]
            out["var_level"] = stitched["var_level"]
        if want_losses :
            for key in ("quadratic_loss", "smooth_loss", "tick_loss", "firm_loss",
                        "n_violations", "violation_rate") :
                out[key] = stitched[key]
        metric_list = [m for m in metric_list if m not in requested_stitched]

    for name in metric_list :
        if name not in _METRICS :
            raise KeyError(
                f"Unknown metric '{name}'; registered: {tuple(_METRICS)} "
                f"plus stitched metrics {tuple(STITCHED_METRIC_NAMES)}")
        fn = _METRICS[name]
        kwargs = {"streams" : streams}
        if name == "persistence" and "valid" in out :
            kwargs["streams"] = {**streams, "valid" : out["valid"]}
        chunk = fn(plan, **kwargs)
        out.update(chunk)
    return out


def _print_metric_summary(plan, result, verbose=True) :
    if not verbose :
        return
    rmse = result.get("rmse")
    valid = result.get("valid")
    persistence = result.get("persistence")
    if rmse is None :
        return
    n_tickers = plan["n_tickers"]
    n_blocks = plan["n_blocks"]
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for phase_idx, name in enumerate(("train", "val  ", "test ")) :
            mean_rmse = float(np.nanmean(rmse[:, phase_idx, :]))
            mean_pers = float("nan")
            if persistence is not None :
                mean_pers = float(np.nanmean(persistence[:, phase_idx, :]))
            n_valid = int(valid[:, phase_idx, :].sum()) if valid is not None else 0
            print(f"  {name}: mean RMSE = {mean_rmse:.5g}  "
                  f"persistence = {mean_pers:.5g}  "
                  f"({n_valid}/{n_tickers * n_blocks} valid entries)")


def analyze_multislice(paramfile, *, plan=None, rundir=None,
                       metrics=("rmse", "persistence"), var_level=0.95,
                       verbose=True, save_metrics=None) :
    """
    Compute multislice outcome metrics from on-disk simulator outputs.

    Parameters
    ----------
    paramfile : str
        Multislice ``.par`` file (defines schedule and feature layout).
    plan : dict or None
        Precomputed plan from :func:`MultislicePlan.build_multislice_plan`
        or :func:`Mains.run_multislice`.
    rundir : str or None
        Directory containing ``netw1_outpop_*_dist.bin`` (``None`` = CWD).
    metrics : sequence of str
        Registered metric names (default: ``rmse``, ``persistence``). Also
        supports stitched per-ticker metrics ``hit``, ``var``, and DeepVaR
        losses ``quadratic_loss``, ``smooth_loss``, ``tick_loss``, ``firm_loss``.
    var_level : float
        Confidence level for ``var`` (left-tail VaR at quantile ``1 - var_level``
        on the log-return pmf). Ignored unless ``var`` is requested.
    verbose : bool
        Print per-phase mean RMSE summary.
    save_metrics : str or None
        If set, save computed arrays to this ``.npz`` path.

    Returns
    -------
    dict
        Same keys as legacy :func:`Mains.multislice` plus ``paramfile``,
        ``rundir``, ``metrics_computed``.
    """
    from MultislicePlan import build_multislice_plan

    if plan is None :
        plan = build_multislice_plan(
            paramfile, encode=False, prefer_saved_encoder=True,
            rundir=rundir)

    streams = load_decoded_streams(plan, rundir=rundir, load_dists=True)
    cubes = compute_metric_cubes(
        plan, streams, metrics=metrics, var_level=var_level, rundir=rundir)

    block_active_local = plan["block_active_local"]
    result = {
        "tickers"      : list(plan["ticker_labels"]),
        "block_active" : [[lbl for lbl, _ in active]
                          for active in block_active_local],
        "trnpat"       : plan["trnpat"],
        "vanpat"       : plan["vanpat"],
        "tenpat"       : plan["tenpat"],
        "warmup_pat"   : plan["warmup_pat"],
        "block_step"   : plan["block_step"],
        "n_blocks"     : plan["n_blocks"],
        "paramfile"    : paramfile,
        "rundir"      : rundir,
        "metrics_computed" : tuple(metrics),
    }
    result.update(cubes)

    if verbose and "rmse" in cubes :
        print(f"Mains.analyze_multislice: paramfile={paramfile}"
              + (f"  rundir={rundir}" if rundir else ""))
        _print_metric_summary(plan, result, verbose=True)

    if save_metrics :
        save_multislice_metrics(result, save_metrics)

    return result


def save_multislice_metrics(result, path) :
    """Persist metric arrays from :func:`analyze_multislice` to ``.npz``."""
    arrays = {}
    for key in ("rmse", "valid", "persistence", "hit_ratio",
                "quadratic_loss", "smooth_loss", "tick_loss", "firm_loss",
                "n_violations", "violation_rate") :
        if key in result :
            arrays[key] = result[key]
    for key in ("hit", "var") :
        if key in result :
            arrays[key] = np.array(list(result[key].items()), dtype=object)
    if "var_level" in result :
        arrays["_meta_var_level"] = result["var_level"]
    meta = {k : result[k] for k in (
        "tickers", "trnpat", "vanpat", "tenpat", "warmup_pat",
        "block_step", "n_blocks", "paramfile", "rundir",
        "metrics_computed")}
    arrays["_meta_tickers"] = np.array(meta["tickers"], dtype=object)
    for k, v in meta.items() :
        if k != "tickers" :
            arrays[f"_meta_{k}"] = v
    np.savez(path, **arrays)


def load_multislice_metrics(path) :
    """Reload a metrics bundle written by :func:`save_multislice_metrics`."""
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in ("rmse", "valid", "persistence", "hit_ratio",
                "quadratic_loss", "smooth_loss", "tick_loss", "firm_loss",
                "n_violations", "violation_rate") :
        if key in data :
            result[key] = data[key]
    for key in ("hit", "var") :
        if key in data :
            result[key] = dict(data[key])
    if "_meta_var_level" in data :
        result["var_level"] = float(data["_meta_var_level"])
    if "_meta_tickers" in data :
        result["tickers"] = list(data["_meta_tickers"])
    for k in data.files :
        if k.startswith("_meta_") and k != "_meta_tickers" :
            result[k[len("_meta_"):]] = data[k].item() if data[k].ndim == 0 else data[k]
    return result
