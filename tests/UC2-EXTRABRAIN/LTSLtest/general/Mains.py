import numpy as np
import pandas as pd
import os
import subprocess
import warnings
import pickle

import Utils
import FeatureGeneration
import Decode
import RBFBinner
import Display

from Utils import RMSE, persistence_rmse  # noqa: F401 (re-exported for back-compat)

def _min_raw_series_length(n_patterns: int, window_n: int) -> int:
    """Minimum len(x) so that after the rolling-feature window there are ``n_patterns`` rows."""
    return int(n_patterns) + int(window_n) - 1

#so this is the main function I will need to generalize out to any numpy timeseries from any dataset
def general(paramfile = "Parameters.par", exefile = "./ltslmain",
                     plot_range = None, figno = 0, verbose = True,
                     confidence_levels = None) : # plot_range only affects Display.plot_general_prediction
    #print("paramfile =", paramfile)
    mean_nbin = int(Utils.findparamval(paramfile, "mean_nbin"))
    deriv1st_nbin = int(Utils.findparamval(paramfile, "deriv1st_nbin"))
    nbin = mean_nbin * deriv1st_nbin
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
    W = FeatureGeneration.ROLLING_WINDOW_N
    n_needed = trnpat + vanpat + tenpat + ngenstep
    min_raw_length = _min_raw_series_length(n_needed, W)

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
    feats = FeatureGeneration.make_mean_derivative_features(x = x, figno = 0)
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

    rbfbinner_mean = RBFBinner.Encoder1D(np.min(feats[:,3]), np.max(feats[:,3]), mean_nbin, 1./mean_nbin)
    pats_mean = rbfbinner_mean.encode(feats[:,3])
    rbfbinner_deriv1st = RBFBinner.Encoder1D(np.min(feats[:,1]), np.max(feats[:,1]), deriv1st_nbin, 1./deriv1st_nbin)
    pats_deriv1st = rbfbinner_deriv1st.encode(feats[:,1])

    #not sure why pats have 3 axes, but this creates an outer product of mean and 1stderivative, after binning and RBF
    patterns_md = (pats_mean[:, :, None] * pats_deriv1st[:, None, :]).reshape(pats_mean.shape[0], -1) 

    #print(pats_mean.shape, pats_deriv1st.shape, patterns_md.shape)
    #print(savefile_m, infilename_md)

    Utils.savebin(patterns_md, infilename_md) #this is the actual input to the network at runtime, which already consists of mean AND 1st derivative combined 

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

    troutdists = Utils.loadbin("netw1_outpop_tr_dist.bin", mean_nbin) #labeled after offset only for iteration of experimental conditions (ideally outputs need to be uniquely packaged with the corresponding input data and the parameter file used)
    # dists = Decode.pats_to_dists(troutacts, mean_nbin, nbin//mean_nbin)
    #print("troutdists.shape =", troutdists.shape)
    troutx = rbfbinner_mean.decode(troutdists)
    #print("troutx.shape = ", troutx.shape)
    tetroutdists = Utils.loadbin("netw1_outpop_tetr_dist.bin", mean_nbin)
    # dists = Decode.pats_to_dists(tetroutacts, mean_nbin, nbin//mean_nbin)
    tetroutx = rbfbinner_mean.decode(tetroutdists)
    #print("tetroutx.shape = ", tetroutx.shape)

    if vanpat > offs:
        vanoutdists = Utils.loadbin("netw1_outpop_van_dist.bin", mean_nbin)
        vanoutx = rbfbinner_mean.decode(vanoutdists)
    else:
        vanoutx = None

    teusoutdists = Utils.loadbin("netw1_outpop_teus_dist.bin", mean_nbin)
    # dists = Decode.pats_to_dists(teusoutacts, mean_nbin, nbin//mean_nbin)
    teusoutx = rbfbinner_mean.decode(teusoutdists)
    #print("teusoutx.shape =", teusoutx.shape)
    genoutdists = Utils.loadbin("netw1_outpop_gen_dist.bin", mean_nbin)
    # dists = Decode.pats_to_dists(genoutacts, mean_nbin, nbin//mean_nbin)
    genoutx = rbfbinner_mean.decode(genoutdists)

    # Logger captures outpop_md1->act after updstate (via advance() in Globals.cpp),
    # so logger row i is the network's prediction targeting feats[input_time_at_iter_i + offs].
    # Therefore the feats slice on the LHS must be shifted forward by offs to align with the prediction.
    # Each metric is guarded so that empty windows return nan instead of triggering
    # numpy "mean of empty slice" warnings (e.g. when vanpat=0 or tenpat<=offs).
    train_skip = 100  # skip a training-transient prefix
    if trnpat - offs > train_skip:
        rmse_train = RMSE(
            feats[train_skip + offs : trnpat, 3],
            tetroutx[train_skip : trnpat - offs],
        )
    else:
        rmse_train = float("nan")
    if vanpat > offs:
        rmse_validation = RMSE(
            feats[trnpat + offs : trnpat + vanpat, 3],
            vanoutx[0 : vanpat - offs],
        )
    else:
        rmse_validation = float("nan")
    if tenpat > offs:
        rmse_unseen = RMSE(
            feats[trnpat + vanpat + offs : trnpat + vanpat + tenpat, 3],
            teusoutx[0 : tenpat - offs],
        )
    else:
        rmse_unseen = float("nan")
    if ngenstep > 0:
        # genoutx[i] is the i-th autoregressive prediction; aligned with feats[trnpat+vanpat+tenpat+i]
        rmse_generation = RMSE(
            feats[trnpat + vanpat + tenpat : trnpat + vanpat + tenpat + ngenstep, 3],
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
        feats[:, 3],
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
            centers=rbfbinner_mean.centers,
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


def _write_derived_paramfile(src_paramfile, dst_paramfile, overrides) :
    """
    Copy ``src_paramfile`` to ``dst_paramfile`` replacing only the values of
    keys listed in ``overrides`` (dict ``key -> value``). The ``findparamval``
    parser is whitespace-split on ``len(s)==2`` so values must remain
    whitespace-free. Comments (``# ...``) are preserved.
    """
    with open(src_paramfile, "r") as f :
        lines = f.read().splitlines()
    remaining = dict(overrides)
    out_lines = []
    for line in lines :
        stripped = line.strip()
        if not stripped or stripped.startswith("#") :
            out_lines.append(line)
            continue
        head = line
        comment = ""
        if "#" in line :
            head, comment = line.split("#", 1)
            comment = "#" + comment
        tokens = head.split()
        if len(tokens) == 2 and tokens[0] in remaining :
            key = tokens[0]
            new_val = remaining.pop(key)
            new_line = f"{key:<16s} {new_val}"
            if comment :
                new_line = f"{new_line:<32s} {comment}"
            out_lines.append(new_line)
        else :
            out_lines.append(line)
    for key, val in remaining.items() :
        out_lines.append(f"{key:<16s} {val}")
    with open(dst_paramfile, "w") as f :
        f.write("\n".join(out_lines) + "\n")


def _per_segment_rmse(feats_col, decoded_block, segments, block_start, offs) :
    """
    Compute per-segment RMSE for a phase block whose decoded output sits in
    ``decoded_block`` (length sum(s) for segment lengths s). Each segment
    ``(label, length)`` occupies a contiguous chunk in both the decoded array
    and in ``feats_col`` starting at ``block_start`` (index into feats[:,3]).
    Returns ``OrderedDict[label -> rmse_or_nan]``.
    """
    from collections import OrderedDict
    out = OrderedDict()
    cursor = 0
    for label, length in segments :
        if length - offs > 0 :
            truth = feats_col[block_start + cursor + offs :
                              block_start + cursor + length]
            pred = decoded_block[cursor : cursor + length - offs]
            out[label] = float(Utils.RMSE(truth, pred))
        else :
            out[label] = float("nan")
        cursor += length
    return out


def multi(paramfile = "Parameters.par", exefile = "./ltslmain",
          plot_range = None, figno = 0, verbose = True,
          confidence_levels = None) :
    """
    Multi-dataset variant of :func:`general`.

    Train / validation / test patterns are pulled from disjoint columns of one
    or more wide CSVs (e.g. ``sp_81_daily_prices.csv``) instead of being three
    contiguous time slices of a single series. Required parameter keys:

        train_files <stem>:<col1>,<col2>,...[;<stem2>:<colA>,...]
        val_files   <stem>:<col1>,<col2>,...
        test_files  <stem>:<col1>,<col2>,...

    The values must be whitespace-free (constraint of ``Utils.findparamval``).

    Hard concatenation: each phase's columns are loaded, max-normalized per
    column, fed through ``make_mean_derivative_features`` per column, and the
    resulting feature matrices are stacked row-wise. No warmup buffer is
    inserted between columns - the LTSL recurrent state will carry across the
    boundary, which is accepted as the simplest extension of the existing
    pipeline.

    ``trnpat`` / ``vanpat`` / ``tenpat`` from the param file are ignored and
    derived from the loaded data; ``ngenstep`` is forced to 0 (autoregressive
    generation across stock boundaries is ill-defined). A derived param file
    ``<paramfile>.derived.par`` is written next to ``paramfile`` and used to
    invoke ``ltslmain`` so the user's original file stays untouched.

    Returns the same 5-tuple as ``general`` plus a dict of per-segment test
    RMSEs:

        (rmse_train, rmse_validation, rmse_unseen, rmse_generation,
         rmse_persistence, per_segment_test_rmse)
    """
    mean_nbin = int(Utils.findparamval(paramfile, "mean_nbin"))
    deriv1st_nbin = int(Utils.findparamval(paramfile, "deriv1st_nbin"))
    offs = int(Utils.findparamval(paramfile, "offs"))
    infilename_md = Utils.findparamval(paramfile, "infilename_md")

    train_spec = Utils.findparamval(paramfile, "train_files")
    val_spec   = Utils.findparamval(paramfile, "val_files")
    test_spec  = Utils.findparamval(paramfile, "test_files")
    missing = [k for k, v in
               (("train_files", train_spec), ("val_files", val_spec), ("test_files", test_spec))
               if not v]
    if missing :
        raise ValueError(
            f"Mains.multi() requires {missing} in '{paramfile}'. "
            f"Each value is '<csv_stem>:<col1>,<col2>,...' (whitespace-free; "
            f"multiple stems separated by ';').")

    if verbose :
        print(f"Loading multi-dataset phases from '{paramfile}':")
        print(f"  train_files = {train_spec}")
        print(f"  val_files   = {val_spec}")
        print(f"  test_files  = {test_spec}")

    feats_train, train_segs = FeatureGeneration.phase_to_concat_features(train_spec)
    feats_val,   val_segs   = FeatureGeneration.phase_to_concat_features(val_spec)
    feats_test,  test_segs  = FeatureGeneration.phase_to_concat_features(test_spec)
    feats = np.concatenate([feats_train, feats_val, feats_test], axis = 0)
    trnpat = int(feats_train.shape[0])
    vanpat = int(feats_val.shape[0])
    tenpat = int(feats_test.shape[0])
    ngenstep = 0

    if trnpat <= offs or tenpat <= offs :
        raise ValueError(
            f"Too few patterns after feature generation: trnpat={trnpat}, "
            f"vanpat={vanpat}, tenpat={tenpat} with offs={offs}. "
            f"Each block must be longer than offs.")

    if verbose :
        print(f"  trnpat = {trnpat} (segments: {train_segs})")
        print(f"  vanpat = {vanpat} (segments: {val_segs})")
        print(f"  tenpat = {tenpat} (segments: {test_segs})")

    rbfbinner_mean = RBFBinner.Encoder1D(
        np.min(feats[:, 3]), np.max(feats[:, 3]), mean_nbin, 1. / mean_nbin)
    pats_mean = rbfbinner_mean.encode(feats[:, 3])
    rbfbinner_deriv1st = RBFBinner.Encoder1D(
        np.min(feats[:, 1]), np.max(feats[:, 1]), deriv1st_nbin, 1. / deriv1st_nbin)
    pats_deriv1st = rbfbinner_deriv1st.encode(feats[:, 1])
    patterns_md = (pats_mean[:, :, None] * pats_deriv1st[:, None, :]) \
        .reshape(pats_mean.shape[0], -1)
    Utils.savebin(patterns_md, infilename_md)

    derived_paramfile = paramfile + ".derived.par"
    _write_derived_paramfile(paramfile, derived_paramfile, {
        "trnpat":   trnpat,
        "vanpat":   vanpat,
        "tenpat":   tenpat,
        "ngenstep": ngenstep,
    })

    if exefile is not None :
        kwargs = {} if verbose else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        subprocess.run([exefile, derived_paramfile], check = True, **kwargs)

    tetroutdists = Utils.loadbin("netw1_outpop_tetr_dist.bin", mean_nbin)
    tetroutx = rbfbinner_mean.decode(tetroutdists)
    if vanpat > offs :
        vanoutdists = Utils.loadbin("netw1_outpop_van_dist.bin", mean_nbin)
        vanoutx = rbfbinner_mean.decode(vanoutdists)
    else :
        vanoutdists = None
        vanoutx = None
    teusoutdists = Utils.loadbin("netw1_outpop_teus_dist.bin", mean_nbin)
    teusoutx = rbfbinner_mean.decode(teusoutdists)
    genoutdists = Utils.loadbin("netw1_outpop_gen_dist.bin", mean_nbin)
    genoutx = rbfbinner_mean.decode(genoutdists)

    train_skip = 100
    if trnpat - offs > train_skip :
        rmse_train = RMSE(
            feats[train_skip + offs : trnpat, 3],
            tetroutx[train_skip : trnpat - offs],
        )
    else :
        rmse_train = float("nan")
    if vanpat > offs :
        rmse_validation = RMSE(
            feats[trnpat + offs : trnpat + vanpat, 3],
            vanoutx[0 : vanpat - offs],
        )
    else :
        rmse_validation = float("nan")
    if tenpat > offs :
        rmse_unseen = RMSE(
            feats[trnpat + vanpat + offs : trnpat + vanpat + tenpat, 3],
            teusoutx[0 : tenpat - offs],
        )
    else :
        rmse_unseen = float("nan")
    rmse_generation = float("nan")  # generation disabled in multi-mode

    rmse_persistence = persistence_rmse(
        feats[:, 3],
        offs = offs,
        start = trnpat,
        end = trnpat + vanpat + tenpat,
    )

    per_segment_test_rmse = _per_segment_rmse(
        feats_col = feats[:, 3],
        decoded_block = teusoutx,
        segments = test_segs,
        block_start = trnpat + vanpat,
        offs = offs,
    )

    if figno > 0 :
        Display.plot_general_prediction(
            paramfile = derived_paramfile,
            plot_range = plot_range,
            figno = figno,
            confidence_levels = confidence_levels,
            feats = feats,
            tetroutx = tetroutx,
            vanoutx = vanoutx,
            teusoutx = teusoutx,
            genoutx = genoutx,
            tetroutdists = tetroutdists,
            vanoutdists = vanoutdists if vanpat > offs else None,
            teusoutdists = teusoutdists,
            genoutdists = genoutdists,
            centers = rbfbinner_mean.centers,
            rmses = (rmse_train, rmse_validation, rmse_unseen,
                     rmse_generation, rmse_persistence),
        )

    if verbose :
        print(f"  rmse_train       = {rmse_train:.5g}")
        print(f"  rmse_validation  = {rmse_validation:.5g}")
        print(f"  rmse_unseen      = {rmse_unseen:.5g}")
        print(f"  rmse_persistence = {rmse_persistence:.5g}  "
              f"(naive y_t -> y_{{t+{offs}}}=y_t baseline over val+test window)")
        print(f"  per-segment test RMSE:")
        for label, val in per_segment_test_rmse.items() :
            print(f"    {label:<32s} = {val:.5g}")

    return (rmse_train, rmse_validation, rmse_unseen, rmse_generation,
            rmse_persistence, per_segment_test_rmse)