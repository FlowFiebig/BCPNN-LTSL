"""
Shared multislice schedule construction and artifact path resolution.

Single source of truth for block layout used by :func:`Mains.run_multislice`,
:func:`Mains.analyze_multislice`, and :func:`Display.stitch_multislice_ticker`.
"""
import os
import shutil

import numpy as np

import Utils
import FeatureGeneration

# Standard simulator output basenames (written by ltslmain in the run cwd).
_OUTPOP_TETR = "netw1_outpop_tetr_dist.bin"
_OUTPOP_VAN = "netw1_outpop_van_dist.bin"
_OUTPOP_TEUS = "netw1_outpop_teus_dist.bin"

_ARTIFACT_KEYS = ("features", "schedule", "centers", "tetr_dist", "van_dist", "teus_dist")


def require_schedule_file(paramfile) :
    """Return the ``schedule_file`` path declared in ``paramfile``."""
    schedule_file = Utils.findparamval(paramfile, "schedule_file")
    if not schedule_file :
        raise ValueError(
            f"'{paramfile}' must declare schedule_file (e.g. "
            f"schedule_file schedule.dat). Python writes the sidecar "
            f"before invoking ltslmain; the sidecar is the source of truth "
            f"for total trnpat/vanpat/tenpat.")
    return str(schedule_file)


def artifact_basenames(paramfile) :
    """Map logical artifact keys to basenames for a given paramfile."""
    infilename_md = Utils.findparamval(paramfile, "infilename_md")
    schedule_file = require_schedule_file(paramfile)
    return {
        "features"  : str(infilename_md),
        "schedule"  : schedule_file,
        "centers"   : FeatureGeneration.centers_sidecar_path(infilename_md),
        "tetr_dist" : _OUTPOP_TETR,
        "van_dist"  : _OUTPOP_VAN,
        "teus_dist" : _OUTPOP_TEUS,
    }


def multislice_artifact_path(name, *, rundir=None, paramfile=None,
                              plan=None) :
    """
    Resolve a multislice artifact path on disk.

    Parameters
    ----------
    name : str
        One of ``features``, ``schedule``, ``centers``, ``tetr_dist``,
        ``van_dist``, ``teus_dist``.
    rundir : str or None
        When set, artifacts live under this directory; otherwise CWD.
    paramfile : str or None
        Required when ``plan`` is None (to read ``infilename_md`` / schedule).
    plan : dict or None
        Optional precomputed plan; may contain ``artifact_basenames``.
    """
    if name not in _ARTIFACT_KEYS :
        raise KeyError(
            f"Unknown artifact '{name}'; expected one of {_ARTIFACT_KEYS}.")
    if plan is not None and "artifact_basenames" in plan :
        basename = plan["artifact_basenames"][name]
    elif paramfile is not None :
        basename = artifact_basenames(paramfile)[name]
    else :
        raise ValueError("multislice_artifact_path needs paramfile or plan.")
    base = rundir if rundir else "."
    return os.path.join(base, basename)


def _validate_pattern_counts(trnpat, vanpat, tenpat, offs, warmup_pat) :
    if trnpat <= offs or vanpat <= offs or tenpat <= offs :
        raise ValueError(
            f"Each of trnpat/vanpat/tenpat must be strictly greater than "
            f"offs (got trnpat={trnpat}, vanpat={vanpat}, tenpat={tenpat}, "
            f"offs={offs}).")
    if warmup_pat > 0 :
        for name, per_t_len in (("trnpat", trnpat), ("vanpat", vanpat),
                                ("tenpat", tenpat)) :
            if per_t_len <= warmup_pat + offs :
                raise ValueError(
                    f"When warmup_pat={warmup_pat}, {name}={per_t_len} must "
                    f"be > warmup_pat + offs ({warmup_pat + offs}).")


def build_multislice_plan(paramfile, *, encode=True,
                          prefer_saved_encoder=True, rundir=None) :
    """
    Build the multislice block schedule, feature stream, and decoder metadata.

    Parameters
    ----------
    paramfile : str
        Multislice ``.par`` file.
    encode : bool
        When True, run :func:`FeatureGeneration.build_patterns` and attach
        ``patterns_md`` / ``rbfbinner_primary`` for :func:`run_multislice`.
    prefer_saved_encoder : bool
        When True and ``encode`` is False, load centres from the artifact
        sidecar if present; otherwise recompute from concatenated features.
    rundir : str or None
        Directory where run artifacts live (for sidecar lookup when analyzing).

    Returns
    -------
    dict
        Plan consumed by run, analyze, and Display helpers.
    """
    cfg = FeatureGeneration.feature_config(paramfile)
    params = Utils.load_params(paramfile)
    offs = int(params["offs"])
    trnpat = int(params["trnpat"])
    vanpat = int(params["vanpat"])
    tenpat = int(params["tenpat"])
    warmup_pat = int(params.get("warmup_pat", 0))
    infilename_md = Utils.findparamval(paramfile, "infilename_md")
    block_step = trnpat + vanpat + tenpat
    _validate_pattern_counts(trnpat, vanpat, tenpat, offs, warmup_pat)

    tickers = Utils.findparamval(paramfile, "tickers")
    timeseries_file = Utils.findparamval(paramfile, "timeseries_file")
    if not timeseries_file :
        raise ValueError(
            f"build_multislice_plan() requires timeseries_file in "
            f"'{paramfile}'.")

    per_ticker, _csv_lens = FeatureGeneration.load_tickers_features(
        timeseries_file, tickers, feature_func=cfg.func,
        feature_align=FeatureGeneration.feature_align(cfg))
    ticker_labels = list(per_ticker.keys())
    ticker_idx = {label : i for i, label in enumerate(ticker_labels)}

    W = FeatureGeneration.feature_align(cfg)
    calendar_start = W - 1
    max_calendar_end = max(
        d["first_valid_feat_row"] + d["feats"].shape[0]
        for d in per_ticker.values())
    n_blocks_max = max(0, (max_calendar_end - calendar_start) // block_step)

    feats_chunks = []
    block_active_local = []
    block_pat_counts = []
    block_meta = []
    cursor_tetr = 0
    cursor_van = 0
    cursor_teus = 0

    for b in range(n_blocks_max) :
        block_calendar_start = calendar_start + b * block_step
        active = []
        for label in ticker_labels :
            d = per_ticker[label]
            start_local = block_calendar_start - d["first_valid_feat_row"]
            end_local = start_local + block_step
            if 0 <= start_local and end_local <= d["feats"].shape[0] :
                active.append((label, int(start_local)))
        if not active :
            continue
        N_b = len(active)
        for label, sl in active :
            feats_chunks.append(per_ticker[label]["feats"][sl : sl + trnpat])
        for label, sl in active :
            feats_chunks.append(per_ticker[label]["feats"][
                sl + trnpat : sl + trnpat + vanpat])
        for label, sl in active :
            feats_chunks.append(per_ticker[label]["feats"][
                sl + trnpat + vanpat : sl + block_step])

        block_pat_counts.append((N_b * trnpat, N_b * vanpat, N_b * tenpat))
        block_active_local.append(active)
        block_meta.append({
            "global_b"         : b,
            "active"           : active,
            "tetr_block_start" : cursor_tetr,
            "van_block_start"  : cursor_van,
            "teus_block_start" : cursor_teus,
        })
        cursor_tetr += (Utils.multislice_phase_logged_count(
            N_b, trnpat, offs, warmup_pat)
            if warmup_pat > 0 else N_b * trnpat - offs)
        cursor_van += (Utils.multislice_phase_logged_count(
            N_b, vanpat, offs, warmup_pat)
            if warmup_pat > 0 else N_b * vanpat - offs)
        cursor_teus += (Utils.multislice_phase_logged_count(
            N_b, tenpat, offs, warmup_pat)
            if warmup_pat > 0 else N_b * tenpat - offs)

    n_blocks = len(block_active_local)
    if n_blocks == 0 :
        raise ValueError(
            f"build_multislice_plan() found no usable blocks for "
            f"'{paramfile}'. block_step={block_step}, longest ticker "
            f"history={max_calendar_end - calendar_start} feature rows.")

    feats = np.concatenate(feats_chunks, axis=0)

    rbfbinner_primary = None
    patterns_md = None
    if encode :
        rbfbinner_primary, patterns_md = FeatureGeneration.build_patterns(
            feats, cfg)
    elif prefer_saved_encoder :
        centers_path = multislice_artifact_path(
            "centers", rundir=rundir, paramfile=paramfile)
        if os.path.exists(centers_path) :
            import RBFBinner
            rbfbinner_primary = RBFBinner.Encoder1D.from_centers(
                np.load(centers_path))
        if rbfbinner_primary is None :
            infilename_md_path = multislice_artifact_path(
                "features", rundir=rundir, paramfile=paramfile)
            rbfbinner_primary = (
                FeatureGeneration.load_primary_encoder(infilename_md_path)
                if infilename_md else None)
        if rbfbinner_primary is None :
            rbfbinner_primary = FeatureGeneration.primary_encoder(feats, cfg)
    else :
        rbfbinner_primary = FeatureGeneration.primary_encoder(feats, cfg)

    basenames = artifact_basenames(paramfile)
    return {
        "paramfile"           : paramfile,
        "rundir"             : rundir,
        "offs"                : offs,
        "trnpat"              : trnpat,
        "vanpat"              : vanpat,
        "tenpat"              : tenpat,
        "warmup_pat"          : warmup_pat,
        "block_step"          : block_step,
        "cfg"                 : cfg,
        "primary_col"         : cfg.primary_col,
        "primary_nbin"        : cfg.primary_nbin,
        "mean_nbin"           : cfg.primary_nbin,
        "secondary_nbin"      : cfg.secondary_nbin,
        "infilename_md"       : infilename_md,
        "timeseries_file"     : timeseries_file,
        "per_ticker"          : per_ticker,
        "ticker_labels"       : ticker_labels,
        "ticker_idx"          : ticker_idx,
        "n_tickers"           : len(ticker_labels),
        "block_meta"          : block_meta,
        "block_active_local"  : block_active_local,
        "block_pat_counts"    : block_pat_counts,
        "n_blocks"            : n_blocks,
        "feats"               : feats,
        "patterns_md"         : patterns_md,
        "rbfbinner_primary"   : rbfbinner_primary,
        "rbfbinner_mean"      : rbfbinner_primary,
        "centers"             : rbfbinner_primary.centers,
        "stream_lengths"      : (cursor_tetr, cursor_van, cursor_teus),
        "calendar_start"      : calendar_start,
        "max_calendar_end"    : max_calendar_end,
        "artifact_basenames"  : basenames,
    }


def prepare_run_directory(paramfile, rundir) :
    """
    Create ``rundir`` and write a paramfile snapshot for an isolated run.

    Returns the path to the snapshot paramfile inside ``rundir``.
    """
    os.makedirs(rundir, exist_ok=True)
    snapshot = os.path.join(rundir, "params.par")
    shutil.copy2(paramfile, snapshot)
    return snapshot


def write_run_artifacts(plan, *, rundir=None, schedule_file=None) :
    """
    Persist encoded features, centres sidecar, and schedule sidecar.

    Returns
    -------
    dict
        Resolved artifact paths (absolute when ``rundir`` is set).
    """
    paramfile = plan["paramfile"]
    basenames = plan["artifact_basenames"]
    features_path = multislice_artifact_path(
        "features", rundir=rundir, plan=plan)
    schedule_path = schedule_file or require_schedule_file(paramfile)
    if rundir :
        schedule_path = os.path.join(rundir, os.path.basename(schedule_path))

    Utils.savebin(plan["patterns_md"], features_path)
    FeatureGeneration.save_primary_centers(features_path, plan["rbfbinner_primary"])
    Utils.write_schedule(schedule_path, plan["block_pat_counts"])

    centers_path = FeatureGeneration.centers_sidecar_path(features_path)
    return {
        "features"  : features_path,
        "schedule"  : schedule_path,
        "centers"   : centers_path,
        "tetr_dist" : multislice_artifact_path("tetr_dist", rundir=rundir, plan=plan),
        "van_dist"  : multislice_artifact_path("van_dist", rundir=rundir, plan=plan),
        "teus_dist" : multislice_artifact_path("teus_dist", rundir=rundir, plan=plan),
    }


def resolve_exefile(exefile, *, rundir=None) :
    """Return an executable path that works when ``cwd=rundir``."""
    if rundir and not os.path.isabs(exefile) :
        return os.path.abspath(exefile)
    return exefile
