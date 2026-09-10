import numpy as np
import pandas as pd
import os
import subprocess
import sys
import warnings
import pickle

import Utils
import FeatureGeneration
import Decode
import RBFBinner
import Display
import MultislicePlan
import MultisliceAnalyze

from Utils import RMSE, persistence_rmse  # noqa: F401 (re-exported for back-compat)

#so this is the main function I will need to generalize out to any numpy timeseries from any dataset
def general(paramfile = "Parameters.par", exefile = "./ltslmain",
                     plot_range = None, figno = 0, verbose = True,
                     confidence_levels = None) : # plot_range only affects Display.plot_general_prediction
    #print("paramfile =", paramfile)
    cfg = FeatureGeneration.feature_config(paramfile)
    # primary_nbin is the decode-target dimension (the C++ outpop marginal size,
    # i.e. mean_nbin/return_nbin); it is what loadbin / decode below operate on.
    primary_nbin = cfg.primary_nbin
    binmode = Utils.findparamval(paramfile, "binmode")
    offs = int(Utils.findparamval(paramfile, "offs"))
    trnpat = int(Utils.findparamval(paramfile, "trnpat"))
    vanpat = int(Utils.findparamval(paramfile, "vanpat"))
    tenpat = int(Utils.findparamval(paramfile, "tenpat"))
    ngenstep = int(Utils.findparamval(paramfile, "ngenstep"))
    timeseries_file = Utils.findparamval(paramfile, "timeseries_file")
    timeseries_path = FeatureGeneration.dataset_path(timeseries_file)
    infilename_md = Utils.findparamval(paramfile, "infilename_md")

    # The Mackey-Glass generator already strips its own transient internally
    # (see ``transient_cutoff = 500`` in ``FeatureGeneration.mackey_glass``);
    # the sum-of-sinusoids generator and on-disk OHLCV/CSV series have no
    # transient phase, so no extra feature-side cutoff is applied here.
    W = FeatureGeneration.feature_align(cfg)
    n_needed = trnpat + vanpat + tenpat + ngenstep
    min_raw_length = int(n_needed) + int(W) - 1

    pathkey = str(timeseries_file).lower()
    stem = os.path.splitext(os.path.basename(str(timeseries_file)))[0]
    csv_sibling = FeatureGeneration.dataset_path(stem + ".csv")

    need_generate = True
    cache_len = 0
    probe_ok = False
    if os.path.exists(timeseries_path):
        try:
            timeseries_data = pickle.load(open(timeseries_path, "rb"))
            timeseries_x_probe = FeatureGeneration.timeseries_from_dataset(timeseries_data)
            probe_ok = isinstance(timeseries_x_probe, np.ndarray)
            cache_len = len(timeseries_x_probe) if probe_ok else 0
            if probe_ok and cache_len >= min_raw_length:
                need_generate = False
                if verbose:
                    print(f"Using data from {timeseries_path} (len={cache_len})")
        except Exception:
            probe_ok = False
            cache_len = 0

    short_cache = os.path.exists(timeseries_path) and probe_ok and cache_len < min_raw_length

    if need_generate:
        if "mg" in pathkey:
            gen_len = min_raw_length
            if short_cache:
                warnings.warn(
                    f"Timeseries cache has {cache_len} samples but needs at least {min_raw_length} "
                    f"(n_pat={n_needed}, rolling_window={W}); "
                    f"regenerating Mackey-Glass to {gen_len} samples.",
                    UserWarning,
                    stacklevel=2,
                )
            if verbose:
                print(f"Generating Mackey-Glass data ({gen_len} samples) -> {timeseries_path}")
            FeatureGeneration.mackey_glass(tmax=int(gen_len), savefile=timeseries_path)
        elif "sos" in pathkey:
            gen_len = min_raw_length
            if short_cache:
                warnings.warn(
                    f"Timeseries cache has {cache_len} samples but needs at least {gen_len} "
                    f"(n_pat={n_needed}, rolling_window={W}); "
                    f"regenerating sum-of-sinusoids.",
                    UserWarning,
                    stacklevel=2,
                )
            if verbose:
                print(f"Generating sum-of-sinusoids data ({gen_len} samples) -> {timeseries_path}")
            FeatureGeneration.sum_of_sinusoids(
                omegas=[0.01, 0.1, 1.0], tmax=int(gen_len), savefile=timeseries_path)
        elif os.path.exists(csv_sibling):
            if os.path.abspath(csv_sibling) == os.path.abspath(timeseries_path):
                raise ValueError(
                    f"Refusing to overwrite '{timeseries_path}': Mains.general() "
                    f"expects timeseries_file to be a single-series pickle, but it "
                    f"resolved to the wide CSV itself, so regenerating would clobber "
                    f"the source data with a generated pickle. Use Mains.multislice() "
                    f"for wide CSVs (e.g. sp100_IPO_sorted.csv), or point "
                    f"timeseries_file at a single-series pickle / a CSV with a "
                    f"distinct stem.")
            if short_cache:
                warnings.warn(
                    f"Timeseries cache has {cache_len} samples but needs at least {min_raw_length} "
                    f"(n_pat={n_needed}, rolling_window={W}); "
                    f"rebuilding from {csv_sibling}.",
                    UserWarning,
                    stacklevel=2,
                )
            if verbose:
                print(f"Generating OHLCV close-series from {csv_sibling} -> {timeseries_path}")
            FeatureGeneration.ohlcv_close(csv_sibling, savefile=timeseries_path)
        elif os.path.exists(timeseries_path):
            raise ValueError(
                f"Timeseries '{timeseries_file}' has {cache_len} samples but needs at least "
                f"{min_raw_length} (trnpat+vanpat+tenpat+ngenstep={n_needed}, "
                f"rolling_window={W}). "
                f"Provide a longer dataset or reduce pattern counts."
            )
        else:
            raise RuntimeError("Timeseries data missing. No generator specified")

    timeseries_data = pickle.load(open(timeseries_path, "rb"))
    timeseries_x = FeatureGeneration.timeseries_from_dataset(timeseries_data)
    if timeseries_x is None:
        raise ValueError(f"Timeseries '{timeseries_file}' contains no usable 'x' or 'x_norm' array.")
    x = np.asarray(timeseries_x, dtype=float)
    feats = cfg.func(x = x, figno = 0)
    #print("feats.shape =", feats.shape)

    n_available = feats.shape[0]
    if n_available < n_needed :
        raise ValueError(
            f"Insufficient training data: trnpat + vanpat + tenpat + ngenstep = "
            f"{trnpat} + {vanpat} + {tenpat} + {ngenstep} = {n_needed} patterns requested, "
            f"but only {n_available} available from '{timeseries_file}' "
            f"(len(x)={len(timeseries_x)}, rolling_window={W}; "
            f"max patterns = len(x) - {W} + 1). "
            f"Reduce trnpat/vanpat/tenpat/ngenstep or provide a longer timeseries."
        )

    # Outer product of the primary (decode-target) and secondary channels after
    # RBF binning. ``rbfbinner_primary`` is reused below to decode the network's
    # predicted distribution back into primary-channel values.
    rbfbinner_primary, patterns_md = FeatureGeneration.build_patterns(feats, cfg)

    Utils.savebin(patterns_md, infilename_md) #this is the actual input to the network at runtime, the primary AND secondary channel combined 
    FeatureGeneration.save_primary_centers(infilename_md, rbfbinner_primary) #sidecar: exact decode-target centres for out-of-process plotting

    schedule_file = Utils.findparamval(paramfile, "schedule_file")
    if schedule_file :
        Utils.write_schedule(schedule_file, [(trnpat, vanpat, tenpat)])

    if exefile is not None :
        kwargs = {} if verbose else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        subprocess.run([exefile, paramfile], check=True, **kwargs)

    #output file name decoding table
    """
    act denotes activations
    dist denotes distributions (marginalized over the 1st derivative, so most likely what I wanna plot out)
    sup denotes unit suppors (not all that interesting right now)
    tr denotes training phase data
    tetr denotes testing of training samples
    teus denotes testing of unseen samples
    gen denotes genation phase data
    """

    troutdists = Utils.loadbin("netw1_outpop_tr_dist.bin", primary_nbin) #labeled after offset only for iteration of experimental conditions (ideally outputs need to be uniquely packaged with the corresponding input data and the parameter file used)
    #print("troutdists.shape =", troutdists.shape)
    troutx = FeatureGeneration.delog_primary(cfg, rbfbinner_primary.decode(troutdists))
    #print("troutx.shape = ", troutx.shape)
    tetroutdists = Utils.loadbin("netw1_outpop_tetr_dist.bin", primary_nbin)
    tetroutx = FeatureGeneration.delog_primary(cfg, rbfbinner_primary.decode(tetroutdists))
    #print("tetroutx.shape = ", tetroutx.shape)

    if vanpat > offs:
        vanoutdists = Utils.loadbin("netw1_outpop_van_dist.bin", primary_nbin)
        vanoutx = FeatureGeneration.delog_primary(cfg, rbfbinner_primary.decode(vanoutdists))
    else:
        vanoutx = None

    teusoutdists = Utils.loadbin("netw1_outpop_teus_dist.bin", primary_nbin)
    teusoutx = FeatureGeneration.delog_primary(cfg, rbfbinner_primary.decode(teusoutdists))
    #print("teusoutx.shape =", teusoutx.shape)
    genoutdists = Utils.loadbin("netw1_outpop_gen_dist.bin", primary_nbin)
    genoutx = FeatureGeneration.delog_primary(cfg, rbfbinner_primary.decode(genoutdists))

    # Logger captures outpop_md1->act after updstate (via advance() in Globals.cpp),
    # so logger row i is the network's prediction targeting feats[input_time_at_iter_i + offs].
    # Therefore the feats slice on the LHS must be shifted forward by offs to align with the prediction.
    # Each metric is guarded so that empty windows return nan instead of triggering
    # numpy "mean of empty slice" warnings (e.g. when vanpat=0 or tenpat<=offs).
    pcol = cfg.primary_col  # decode-target column (ground truth for RMSE)
    truth_primary = FeatureGeneration.delog_primary(cfg, feats[:, pcol])
    train_skip = 100  # skip a training-transient prefix
    if trnpat - offs > train_skip:
        rmse_train = RMSE(
            truth_primary[train_skip + offs : trnpat],
            tetroutx[train_skip : trnpat - offs],
        )
    else:
        rmse_train = float("nan")
    if vanpat > offs:
        rmse_validation = RMSE(
            truth_primary[trnpat + offs : trnpat + vanpat],
            vanoutx[0 : vanpat - offs],
        )
    else:
        rmse_validation = float("nan")
    if tenpat > offs:
        rmse_unseen = RMSE(
            truth_primary[trnpat + vanpat + offs : trnpat + vanpat + tenpat],
            teusoutx[0 : tenpat - offs],
        )
    else:
        rmse_unseen = float("nan")
    if ngenstep > 0:
        # genoutx[i] is the i-th autoregressive prediction; aligned with feats[trnpat+vanpat+tenpat+i]
        rmse_generation = RMSE(
            truth_primary[trnpat + vanpat + tenpat : trnpat + vanpat + tenpat + ngenstep],
            genoutx[0 : ngenstep],
        )
    else:
        rmse_generation = float("nan")

    # Persistence baseline (naive y_{t+offs} = y_t forecaster) over the entire
    # post-training window (validation + unseen + generation). It depends only
    # on the data, not on the model, so it is a single comparable reference for
    # all three of rmse_validation/rmse_unseen/rmse_generation: any model worth
    # its salt should beat this number on at least the data-driven phases.
    rmse_persistence = persistence_rmse(
        truth_primary,
        offs=offs,
        start=trnpat,
        end=trnpat + vanpat + tenpat + ngenstep,
    )

    if figno > 0 :
        # Plotting lives in Display.plot_general_prediction so it can be invoked
        # standalone after a completed run (e.g. to iterate on the plot code
        # without re-running the simulator); we pass through the precomputed
        # arrays / RMSEs here to avoid redoing the feature + decode work.
        Display.plot_general_prediction(
            paramfile=paramfile,
            plot_range=plot_range,
            figno=figno,
            confidence_levels=confidence_levels,
            feats=feats,
            tetroutx=tetroutx,
            vanoutx=vanoutx,
            teusoutx=teusoutx,
            genoutx=genoutx,
            tetroutdists=tetroutdists,
            vanoutdists=vanoutdists if vanpat > offs else None,
            teusoutdists=teusoutdists,
            genoutdists=genoutdists,
            centers=rbfbinner_primary.centers,
            rmses=(rmse_train, rmse_validation, rmse_unseen,
                   rmse_generation, rmse_persistence),
        )

    if verbose:
        print(f"  rmse_train       = {rmse_train:.5g}")
        print(f"  rmse_validation  = {rmse_validation:.5g}")
        print(f"  rmse_unseen      = {rmse_unseen:.5g}")
        print(f"  rmse_generation  = {rmse_generation:.5g}")
        print(f"  rmse_persistence = {rmse_persistence:.5g}  "
              f"(naive y_t -> y_{{t+{offs}}}=y_t baseline over post-train window)")

    return rmse_train, rmse_validation, rmse_unseen, rmse_generation, rmse_persistence


def _require_schedule_file(paramfile) :
    """Return the ``schedule_file`` path declared in ``paramfile``."""
    return MultislicePlan.require_schedule_file(paramfile)


def run_multislice(paramfile="Parameters.par", exefile="./ltslmain",
                   schedule_file=None, rundir=None, verbose=True) :
    """
    Encode multislice inputs, write schedule sidecar, and invoke ``ltslmain``.

    Does **not** compute outcome metrics; use :func:`analyze_multislice` on the
  written artifacts.

    Parameters
    ----------
    rundir : str or None
        When set, all artifacts and a paramfile snapshot are written under this
        directory and ``ltslmain`` runs with ``cwd=rundir``. When ``None``,
        legacy CWD behaviour is preserved.

    Returns
    -------
    dict
        Run metadata (``plan``, ``artifacts``, ``stream_lengths``, …) without
        ``rmse`` / ``persistence`` cubes.
    """
    run_paramfile = paramfile
    if rundir :
        run_paramfile = MultislicePlan.prepare_run_directory(paramfile, rundir)

    plan = MultislicePlan.build_multislice_plan(
        run_paramfile, encode=True, rundir=rundir)
    plan["paramfile"] = run_paramfile
    plan["rundir"] = rundir

    if verbose :
        print(f"Mains.run_multislice: timeseries_file={plan['timeseries_file']}")
        print(f"  tickers={plan['ticker_labels']}")
        print(f"  trnpat={plan['trnpat']}  vanpat={plan['vanpat']}  "
              f"tenpat={plan['tenpat']}  warmup_pat={plan['warmup_pat']}  "
              f"block_step={plan['block_step']}")
        print(f"  calendar_start={plan['calendar_start']}  "
              f"max_calendar_end={plan['max_calendar_end']}  "
              f"n_blocks={plan['n_blocks']}")
        if rundir :
            print(f"  rundir={rundir}")
        print(f"  total feature rows={plan['feats'].shape[0]}  "
              f"(per-phase totals: trn={sum(p[0] for p in plan['block_pat_counts'])}, "
              f"van={sum(p[1] for p in plan['block_pat_counts'])}, "
              f"ten={sum(p[2] for p in plan['block_pat_counts'])})")

    artifacts = MultislicePlan.write_run_artifacts(
        plan, rundir=rundir, schedule_file=schedule_file)

    if exefile is not None :
        exe = MultislicePlan.resolve_exefile(exefile, rundir=rundir)
        par_arg = (os.path.basename(run_paramfile) if rundir
                   else run_paramfile)
        proc = subprocess.run(
            [exe, par_arg],
            cwd=rundir or None,
            capture_output=True,
            text=True,
        )
        if verbose and proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if verbose and proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n",
                  file=sys.stderr)
        if proc.returncode != 0:
            diag = Utils.diagnose_ltslmain_cuda(print_report=False)
            hint = ""
            if diag["issues"]:
                hint = (
                    "\n\nCUDA preflight issues:\n  - "
                    + "\n  - ".join(diag["issues"])
                )
                if any("mismatch" in s.lower() for s in diag["issues"]):
                    hint += (
                        "\n\nLikely fix: reboot to reload the NVIDIA kernel "
                        "module after a driver update."
                    )
            raise RuntimeError(
                f"ltslmain exited with status {proc.returncode} "
                f"(command: {[exe, par_arg]}).\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
                f"{hint}"
            ) from None

    expected = plan["stream_lengths"]
    if exefile is not None :
        streams = MultisliceAnalyze.load_decoded_streams(plan, rundir=rundir)
        got = streams["got"]
        if got != expected :
            sched = artifacts.get("schedule", schedule_file)
            raise RuntimeError(
                f"Decoded stream length mismatch (tetr/van/teus): expected "
                f"{expected} but got {got}. ltslmain likely ran a different "
                f"schedule than '{sched}'.")

    block_active_local = plan["block_active_local"]
    return {
        "paramfile"      : run_paramfile,
        "source_paramfile" : paramfile,
        "rundir"        : rundir,
        "plan"           : plan,
        "tickers"        : list(plan["ticker_labels"]),
        "block_active"   : [[lbl for lbl, _ in active]
                            for active in block_active_local],
        "trnpat"         : plan["trnpat"],
        "vanpat"         : plan["vanpat"],
        "tenpat"         : plan["tenpat"],
        "warmup_pat"     : plan["warmup_pat"],
        "block_step"     : plan["block_step"],
        "n_blocks"       : plan["n_blocks"],
        "stream_lengths" : expected,
        "artifacts"      : artifacts,
    }


def analyze_multislice(paramfile="Parameters.par", *, plan=None, rundir=None,
                       metrics=("rmse", "persistence"), var_level=0.95,
                       verbose=True, save_metrics=None) :
    """
    Compute multislice metrics from existing simulator output files.

    See :func:`MultisliceAnalyze.analyze_multislice` for details.
    """
    if plan is not None and rundir is None :
        rundir = plan.get("rundir")
    return MultisliceAnalyze.analyze_multislice(
        paramfile, plan=plan, rundir=rundir, metrics=metrics,
        var_level=var_level, verbose=verbose, save_metrics=save_metrics)


def multislice(paramfile = "Parameters.par", exefile = "./ltslmain",
               schedule_file = None, figno = 0, verbose = True,
               analyze = True, rundir = None,
               metrics = ("rmse", "persistence"),
               var_level = 0.95,
               save_metrics = None) :
    """
    Walk-forward multi-ticker continual-learning trainer.

    Instead of splitting the ticker universe into disjoint train / val / test
    sets (which leaks the test calendar window into training when held-out
    tickers span the same dates), this function trains, validates, and tests
    *all* selected tickers on consecutive time blocks. The model state is
    preserved across blocks because the whole schedule runs inside one
    ``ltslmain`` process (the C++ side reads ``schedule_file`` and loops).

    Required parameter keys in ``paramfile``:

    * ``timeseries_file`` - wide CSV in ``datasets/`` (e.g.
      ``sp100_IPO_sorted.csv``).
    * ``tickers``  - comma-separated column names from that CSV, e.g.
      ``AAPL,MSFT`` (whitespace-free). Omit or leave empty to use every ticker
      column in ``timeseries_file``.
    * ``trnpat``   - per-ticker, per-block feature rows used for training.
    * ``vanpat``   - per-ticker, per-block feature rows used for validation.
    * ``tenpat``   - per-ticker, per-block feature rows used for testing.
    * bin counts for the active ``feature_set`` (``mean_nbin`` +
      ``deriv1st_nbin`` for the default set, or ``return_nbin`` +
      ``variance_nbin`` for ``feature_set return_variance``), plus ``offs`` and
      ``infilename_md`` - as usual.

    Block schedule
    --------------
    Block ``b`` covers calendar feature rows
    ``[W - 1 + b * block_step, W - 1 + (b + 1) * block_step)`` where
    ``block_step = trnpat + vanpat + tenpat`` and ``W = ROLLING_WINDOW_N``.
    The anchor at ``W - 1`` matches the first valid feature row of any
    full-history ticker. Ticker ``t`` participates in block ``b`` iff its
    feature array fully covers that calendar range; otherwise the
    ``(t, phase, b)`` entries are left as ``NaN`` in the RMSE cube and
    ``False`` in the matching ``valid`` mask. Late-IPO tickers (e.g. ``META``,
    ``GOOGL``, ``DOW``, ``ABBV``) start
    participating once their leading-NaN run has elapsed.

    Within each block the patterns are concatenated as
    ``[train_t0, train_t1, ..., val_t0, val_t1, ..., test_t0, test_t1, ...]``
    so the C++ loop can run a single train -> test-on-train -> val -> test
    pass per block. The recurrent LTSL state and the BCPNN weights carry
    across both ticker boundaries (hard concatenation, with optional
    ``warmup_pat`` forward-only steps at ticker-to-ticker transitions)
    and block boundaries (this is the new continual-learning
    behaviour we want).

    ``ngenstep`` is forced to 0 (autoregression across stock/block boundaries
    is ill-defined).

    Returns
    -------
    dict with keys::

        "rmse"          - ndarray [n_tickers, 3, n_blocks]; phase axis is
                          (train, val, test). NaN for (ticker, block) entries
                          where the ticker is not active in that block.
        "valid"         - bool ndarray, same shape, True where rmse is finite.
        "persistence"   - ndarray [n_tickers, 3, n_blocks]; per-(ticker,
                          phase, block) persistence baseline RMSE
                          (``y_t -> y_{t+offs} = y_t``).
        "tickers"       - list[str] of ticker labels in row order (matches the
                          first axis of ``rmse``/``valid``/``persistence``).
        "block_active"  - list[list[str]] of length ``n_blocks``; tickers
                          active in each block (in the same order they
                          contributed to ``features.dat`` for that block).
        "trnpat"/"vanpat"/"tenpat"/"block_step"/"n_blocks" - echoed for
                          downstream plotting.

    When ``analyze=False``, returns the :func:`run_multislice` dict instead
    (no metric cubes). Pass ``rundir`` to isolate artifacts under a
    dedicated directory.
    """
    run = run_multislice(
        paramfile, exefile=exefile, schedule_file=schedule_file,
        rundir=rundir, verbose=verbose)
    if not analyze :
        return run
    return analyze_multislice(
        run["paramfile"], plan=run["plan"], rundir=rundir,
        metrics=metrics, var_level=var_level,
        verbose=verbose, save_metrics=save_metrics)