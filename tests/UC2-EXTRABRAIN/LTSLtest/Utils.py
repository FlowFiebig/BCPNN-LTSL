import sys, os, select
import socket
import math
import numpy as np
import random
import string
import copy
import scipy.spatial.distance as dist
import scipy.cluster.vq as vq
import time
import csv
from matplotlib import pyplot as plt
from numpy import array
import importlib
import pickle

hostname = socket.gethostname()

if "ubuntu" in hostname:
    import matplotlib.pyplot as plt

def reload(file) :

    importlib.reload(file)
    

def printarr(arr,ndec = 1,floatmode = 'fixed',suppress_small = False) :
    print(np.array2string(arr, precision=ndec,floatmode = floatmode, suppress_small = False))

def findparamval(paramfilename,paramstr) :

    with open(paramfilename, 'r') as file:
        
        paramnum = file.read()

    paramnum =  paramnum.split("\n")

    ss = []

    for s in paramnum :

        if 0<=s.find("#") :

            s = s[:s.find("#")]

        ss.append(s)

    sfound = False

    for s in ss :

        s = s.split()

        if len(s)==2 and s[0]==paramstr :

            if type(s[1])==str :

                sfound = s[1]

            else :

                sfound = float(s[1])

    return sfound
    

def modparamvals(parfile = "ihimain.par",modparfile = "ihimain.parx",
                 parmods = []) :

    file_ut = open(modparfile,"w")

    with open(parfile) as file_in:

        lines = []

        for line in file_in:

            file_ut.write(line)

    file_ut.write("\n\n######## Parameters modified ########\n\n")

    for parmod in parmods :

        blks = " " * (18 - len(parmod[0]))

        # print(parmod[0] + blks + str(parmod[1]))

        file_ut.write(parmod[0] + blks + str(parmod[1])+ "\n")

    file_ut.close()


def RMSE(y0, y1, offs = 0) :
    """Root-mean-square error between two arrays, with optional offset shift.

    With the default ``offs = 0`` this is the standard RMSE between two equal-
    length arrays. ``offs > 0`` trims the last ``offs`` samples off ``y0`` and
    the first ``offs`` samples off ``y1`` before comparing.
    """
    y0 = y0[-offs:]
    y1 = y1[offs:]
    return np.sqrt(np.mean((y0 - y1)**2))


def discrete_pmf_quantile(pmf, centers, q):
    """Return quantile ``q`` in ``[0, 1]`` from a discrete pmf over ``centers``."""
    p = np.asarray(pmf, dtype=float)
    c = np.asarray(centers, dtype=float)
    s = p.sum()
    if s > 0:
        p = p / s
    cdf = np.cumsum(p)
    return float(np.interp(float(q), cdf, c))


def persistence_rmse(y, offs, start, end):
    """
    Naive 'y_{t+offs} = y_t' baseline forecaster RMSE on y[start:end].

    A genuine model should beat this; if rmse_validation/rmse_unseen/rmse_generation
    are similar to or worse than this number, the model has not learned anything
    beyond the trivial persistence forecast.

    The window is automatically clipped to the available range (so that y[t+offs]
    is in-bounds for every t in the chosen window).
    """
    y = np.asarray(y, dtype=float)
    if offs <= 0 or len(y) <= offs:
        return float("nan")
    end = int(min(end, len(y) - offs))
    start = int(max(start, 0))
    if end <= start:
        return float("nan")
    diffs = y[start:end] - y[start + offs : end + offs]
    return float(np.sqrt(np.mean(diffs ** 2)))


_CUDA_ERROR_NAMES = {
    100: "cudaErrorNoDevice (no CUDA-capable device is detected)",
    101: "cudaErrorInvalidDevice",
    999: "cudaErrorUnknown",
}


def _read_proc_driver_nvidia_version():
    path = "/proc/driver/nvidia/version"
    try:
        with open(path) as fh:
            return fh.read().strip().split("\n")[0]
    except OSError:
        return None


def _modinfo_nvidia_version():
    import subprocess
    try:
        out = subprocess.run(
            ["modinfo", "-F", "version", "nvidia"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def _cuda_device_count():
    """Return (error_code, device_count). error_code 0 means success."""
    import ctypes
    try:
        lib = ctypes.CDLL("libcudart.so.12")
    except OSError:
        try:
            lib = ctypes.CDLL("libcudart.so")
        except OSError as exc:
            return -1, 0, f"libcudart not found: {exc}"
    count = ctypes.c_int(0)
    err = int(lib.cudaGetDeviceCount(ctypes.byref(count)))
    return err, count.value, _CUDA_ERROR_NAMES.get(err, f"cuda error {err}")


def diagnose_ltslmain_cuda(*, print_report=True):
    """
    Preflight diagnostics for ``./ltslmain`` (CUDA-only BCPNN simulator).

    Returns a dict with ``ok`` (bool), ``issues`` (list of str), and detail
    fields. When ``print_report`` is True, prints a human-readable summary.
    """
    import subprocess

    issues = []
    info = {}

    # --- GPU hardware ---------------------------------------------------------
    try:
        lspci = subprocess.run(
            ["lspci"], capture_output=True, text=True, check=False,
        )
        gpus = [
            ln for ln in lspci.stdout.splitlines()
            if "nvidia" in ln.lower() or "vga" in ln.lower() or "3d" in ln.lower()
        ]
        info["lspci_gpu"] = gpus
        if not any("nvidia" in ln.lower() for ln in gpus):
            issues.append("No NVIDIA GPU reported by lspci.")
    except OSError:
        info["lspci_gpu"] = None

    # --- kernel module for running kernel -------------------------------------
    running_kern = os.uname().release
    info["running_kernel"] = running_kern
    mod_dirs = [
        f"/lib/modules/{running_kern}/kernel/nvidia",
        f"/lib/modules/{running_kern}/updates/dkms",
        f"/lib/modules/{running_kern}/updates",
    ]
    nvidia_ko = []
    for d in mod_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                nvidia_ko.extend(
                    os.path.join(root, f)
                    for f in files if f.startswith("nvidia") and f.endswith(".ko")
                )
    info["nvidia_ko_for_running_kernel"] = nvidia_ko
    if not nvidia_ko and not os.path.isdir(f"/proc/driver/nvidia"):
        pkg = f"linux-modules-nvidia-580-open-{running_kern}"
        issues.append(
            f"No NVIDIA kernel module for running kernel {running_kern}. "
            f"The driver userspace (580.x) is installed but the matching "
            f"kernel module package is missing — install with:\n"
            f"      sudo apt install {pkg}\n"
            f"    or: sudo apt install linux-modules-nvidia-580-open-generic-hwe-24.04\n"
            f"    then reboot (or sudo modprobe nvidia)."
        )

    # --- nvidia-smi / NVML ----------------------------------------------------
    smi = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False,
    )
    info["nvidia_smi_rc"] = smi.returncode
    info["nvidia_smi_stdout"] = smi.stdout.strip()
    info["nvidia_smi_stderr"] = smi.stderr.strip()
    if smi.returncode != 0:
        smi_err = smi.stderr.strip() or smi.stdout.strip() or f"exit {smi.returncode}"
        issues.append(f"nvidia-smi failed: {smi_err}")

    # --- driver version consistency -------------------------------------------
    info["proc_driver_version"] = _read_proc_driver_nvidia_version()
    info["modinfo_version"] = _modinfo_nvidia_version()
    if info["proc_driver_version"] and info["modinfo_version"]:
        import re
        proc_ver = info["proc_driver_version"]
        mod_ver = info["modinfo_version"]
        loaded_m = re.search(r"\b(\d+\.\d+\.\d+)\b", proc_ver)
        loaded_tag = loaded_m.group(1) if loaded_m else proc_ver
        if mod_ver != loaded_tag and mod_ver not in proc_ver:
            issues.append(
                "NVIDIA driver mismatch: loaded kernel module "
                f"({loaded_tag}) ≠ installed module ({mod_ver}). "
                "Typical after a driver update without reboot — CUDA sees no device."
            )

    # --- CUDA runtime device count --------------------------------------------
    err, count, err_msg = _cuda_device_count()
    info["cuda_get_device_count_err"] = err
    info["cuda_device_count"] = count
    info["cuda_get_device_count_msg"] = err_msg
    if err != 0 or count == 0:
        issues.append(
            f"cudaGetDeviceCount failed or returned 0: {err_msg} (err={err})."
        )

    # --- ltslmain binary ------------------------------------------------------
    for candidate in ("./ltslmain", "../ltslmain"):
        if os.path.isfile(candidate) or os.path.islink(candidate):
            info["ltslmain_path"] = os.path.realpath(candidate)
            break
    else:
        issues.append("ltslmain executable not found in ./ or ../")

    ok = len(issues) == 0
    report = {
        "ok": ok,
        "issues": issues,
        **info,
    }

    if print_report:
        print("=== ltslmain / CUDA preflight ===")
        if info.get("lspci_gpu"):
            for ln in info["lspci_gpu"]:
                if "nvidia" in ln.lower():
                    print(f"  GPU: {ln.strip()}")
        if info.get("running_kernel"):
            print(f"  Running kernel: {info['running_kernel']}")
        if info.get("proc_driver_version"):
            print(f"  Loaded driver: {info['proc_driver_version']}")
        elif not info.get("nvidia_ko_for_running_kernel"):
            print("  Loaded driver: (none — module not loaded)")
        if info.get("modinfo_version"):
            print(f"  Installed module (modinfo): {info['modinfo_version']}")
        if info.get("nvidia_ko_for_running_kernel"):
            print(f"  nvidia.ko for this kernel: yes ({len(info['nvidia_ko_for_running_kernel'])} file(s))")
        else:
            print("  nvidia.ko for this kernel: MISSING")
        print(f"  cudaGetDeviceCount: err={err}  devices={count}  ({err_msg})")
        if info.get("ltslmain_path"):
            print(f"  ltslmain: {info['ltslmain_path']}")
        if ok:
            print("  Status: OK — CUDA ready for ltslmain.")
        else:
            print("  Status: NOT READY")
            for i, msg in enumerate(issues, 1):
                print(f"    {i}. {msg}")
            print("\n  Remediation (most common fix on this machine):")
            print("    • Reboot after NVIDIA driver updates, or")
            print("    • sudo systemctl reboot  (reloads kernel module 580.x)")
            print("    • Verify: nvidia-smi && python -c \"import ctypes; ...\"")
            print("    • Workaround: Mains.run_multislice(..., exefile=None) if")
            print("      netw1_outpop_*_dist.bin artifacts already exist.")

    return report


def multislice_warmup_len(k, warmup_pat) :
    """Forward-only warmup steps before ticker ``k``'s scored segment (``k > 0`` only)."""
    return int(warmup_pat) if k > 0 and warmup_pat > 0 else 0


def multislice_phase_logged_count(n_tickers, per_t_len, offs, warmup_pat = 0) :
    """Logged output rows per block-phase (matches ltslmain multislice loggers)."""
    n_tickers = int(n_tickers)
    per_t_len = int(per_t_len)
    offs = int(offs)
    warmup_pat = int(warmup_pat)
    if n_tickers <= 0 :
        return 0
    if warmup_pat <= 0 :
        return n_tickers * per_t_len - offs
    return ((per_t_len - offs)
            + (n_tickers - 1) * (per_t_len - warmup_pat - offs))


def multislice_scored_len(k, per_t_len, offs, warmup_pat = 0) :
    """Number of logged predictions for ticker index ``k`` in one phase."""
    w = multislice_warmup_len(k, warmup_pat)
    return per_t_len - w - offs


def multislice_stream_offset(k, per_t_len, offs, warmup_pat = 0) :
    """Start index of ticker ``k`` in the packed logged stream (warmup enabled)."""
    return sum(multislice_scored_len(j, per_t_len, offs, warmup_pat)
               for j in range(k))


def _parse_param_value(s):
    """Convert string to int, float, or keep as str."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load_params(paramfilename):
    """
    Load a .par file into a dictionary of parameter name -> value.
    Values are parsed as int, float, or str as appropriate.
    """
    params = {}
    with open(paramfilename, 'r') as f:
        for line in f:
            if '#' in line:
                line = line[:line.index('#')]
            parts = line.split()
            if len(parts) >= 2:
                name, val = parts[0], parts[1]
                params[name] = _parse_param_value(val)
    return params


def _ltsl_tau_ladder_values(K, taumin, taumax, K_mode="logarithmic"):
    """
    Core LTSL ladder (same as TDPop / ltslmain).

    K_mode == "logarithmic": geometric progression τ_k = taumin * c^k.
    K_mode == "linear":      uniform spacing τ_k = linspace(taumin, taumax, K).

    Returns (taus, c) where c is None for linear mode.
    """
    if K < 2:
        raise ValueError("K must be >= 2 (ltslmain uses (K - 1) in the exponent).")
    if taumax <= taumin:
        raise ValueError("taumax must be greater than taumin.")
    if K_mode == "linear":
        taus = np.linspace(float(taumin), float(taumax), K)
        c = None
    else:
        c = (float(taumax) / float(taumin)) ** (1.0 / (K - 1))
        taus = float(taumin) * (c ** np.arange(K, dtype=float))
    return taus, c


def ltsl_tau_ladder(paramfilename):
    """
    LTSL delay time constants from a .par file.

    Reads K, taumin, taumax and (optionally) K_mode from a parameter file and
    builds the corresponding ladder, matching TDPop / ltslmain.

    Parameters
    ----------
    paramfilename : str
        Path to parameter file (must contain K, taumin, taumax).

    Returns
    -------
    taus : ndarray, shape (K,)
    c : float or None
        Multiplicative ratio between adjacent stages (None for linear mode).
    """
    params = load_params(paramfilename)
    missing = [k for k in ("K", "taumin", "taumax") if k not in params]
    if missing:
        raise KeyError(f"{paramfilename!r} missing required keys: {missing}")
    K = int(params["K"])
    taumin = float(params["taumin"])
    taumax = float(params["taumax"])
    K_mode = str(params.get("K_mode", "logarithmic"))
    return _ltsl_tau_ladder_values(K, taumin, taumax, K_mode)


def write_params(paramfilename, params, preserve_comments=True):
    """
    Write a parameter dictionary to a .par file.
    If preserve_comments=True (default), re-reads the file to keep existing
    comments; otherwise writes only param value pairs.
    """
    comments = {}
    if preserve_comments:
        try:
            with open(paramfilename, 'r') as f:
                for line in f:
                    if '#' in line:
                        before_comment = line[:line.index('#')]
                        comment_part = line[line.index('#'):].rstrip()
                        parts = before_comment.split()
                        if len(parts) >= 2:
                            comments[parts[0]] = comment_part
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            comments[parts[0]] = ""
        except FileNotFoundError:
            pass

    with open(paramfilename, 'w') as f:
        for name, val in params.items():
            line = f"{name:<16} {val}"
            if name in comments and comments[name]:
                line += f"  {comments[name]}"
            f.write(line + "\n")


def write_schedule(schedule_path, block_counts) :
    """
    Write a sidecar schedule file for ``ltslmain``.

    Each row is ``trnpat vanpat tenpat`` for one contiguous block. One row
    covers ``Mains.general()``; many rows cover ``Mains.multislice()``.
    """
    with open(schedule_path, "w") as fp :
        for tp, vp, ep in block_counts :
            fp.write(f"{int(tp)} {int(vp)} {int(ep)}\n")


def read_schedule_totals(schedule_path) :
    """
    Sum phase pattern counts from a schedule sidecar file.

    Returns
    -------
    trnpat, vanpat, tenpat, n_blocks : int
    """
    sum_trn = sum_van = sum_ten = 0
    n_blocks = 0
    with open(schedule_path, "r") as fp :
        for line in fp :
            if "#" in line :
                line = line[:line.index("#")]
            parts = line.split()
            if len(parts) < 3 :
                continue
            sum_trn += int(parts[0])
            sum_van += int(parts[1])
            sum_ten += int(parts[2])
            n_blocks += 1
    return sum_trn, sum_van, sum_ten, n_blocks


def phase_totals_from_paramfile(paramfile) :
    """
    Resolve total ``trnpat`` / ``vanpat`` / ``tenpat`` for plotting.

    When ``schedule_file`` is set in ``paramfile``, totals are summed from
    that sidecar (which is what ``ltslmain`` uses). Otherwise the values are
    read directly from the parameter file (``Mains.general()`` without a
    schedule sidecar).
    """
    params = load_params(paramfile)
    schedule_file = params.get("schedule_file")
    if schedule_file and os.path.exists(str(schedule_file)) :
        trnpat, vanpat, tenpat, _ = read_schedule_totals(str(schedule_file))
        return int(trnpat), int(vanpat), int(tenpat)
    return int(params["trnpat"]), int(params["vanpat"]), int(params["tenpat"])


# Metrics requested from Mains.multislice / analyze_multislice during sweeps.
SWEEP_EVAL_METRICS = (
    "rmse",
    "persistence",
    "quadratic_loss",
    "smooth_loss",
    "tick_loss",
    "firm_loss",
)

# Selectable scalar outcomes for param_sweep(metric=...).
SWEEP_RMSE_METRICS = (
    "rmse_train",
    "rmse_validation",
    "rmse_unseen",
    "rmse_persistence",
)

# Legacy Mains.general / 5-tuple adapter indices for RMSE-named metrics.
_SWEEP_METRIC_TO_INDEX = {
    "rmse_train": 0,
    "rmse_validation": 1,
    "rmse_unseen": 2,
    "rmse_persistence": 4,
}


def _sweep_metrics() :
    """Lazy SWEEP_METRICS so DeepVaRLosses need not import at module load."""
    import DeepVaRLosses
    return SWEEP_RMSE_METRICS + tuple(DeepVaRLosses.DEEPVAR_LOSS_NAMES)


def _nanmean_vector(arr) :
    """NaN-mean over a 1-D (or raveled) metric vector."""
    flat = np.asarray(arr, dtype=float).ravel()
    m = np.isfinite(flat)
    return float(np.mean(flat[m])) if m.any() else float("nan")


def scalar_from_analyze(result, metric) :
    """
    Collapse an ``analyze_multislice`` / ``Mains.multislice`` result dict to one float.

    Same nan-mean rules as ``evolve_hyperparams_v4._populate_metrics``.
    """
    valid = _sweep_metrics()
    if metric not in valid :
        raise ValueError(
            f"Unknown metric {metric!r}; choose from {valid}")

    if metric == "rmse_train" :
        cube = np.asarray(result["rmse"], dtype=float)
        return _nanmean_cube(cube[:, 0, :])
    if metric == "rmse_validation" :
        cube = np.asarray(result["rmse"], dtype=float)
        return _nanmean_cube(cube[:, 1, :])
    if metric == "rmse_unseen" :
        cube = np.asarray(result["rmse"], dtype=float)
        return _nanmean_cube(cube[:, 2, :])
    if metric == "rmse_persistence" :
        cube = np.asarray(result["persistence"], dtype=float)
        return _nanmean_cube(cube[:, 1, :])

    # DeepVaR per-ticker vectors
    if metric not in result :
        raise KeyError(
            f"Metric {metric!r} missing from analyze result; "
            f"pass metrics={SWEEP_EVAL_METRICS!r} to Mains.multislice "
            f"(or use Utils.multislice_for_sweep).")
    return _nanmean_vector(result[metric])


def multislice_for_sweep(**kwargs) :
    """
    Run ``Mains.multislice`` with the full sweep metric set and return the
    analyze dict (for ``param_sweep(..., metric=...)``).
    """
    import Mains
    kw = dict(kwargs)
    kw.setdefault("metrics", SWEEP_EVAL_METRICS)
    return Mains.multislice(**kw)


def param_sweep(
    sweep_spec,
    default_paramfile,
    paramfile,
    run_fn,
    run_kwargs=None,
    metric="rmse_unseen",
    result_index=None,
):
    """
    Run all combinations of parameter values and collect results.

    Parameters
    ----------
    sweep_spec : dict  {param_name: [val1, val2, ...]}
        Parameters to sweep. Single-value lists are kept constant.
    default_paramfile : str
        Path to the baseline .par file to start from.
    paramfile : str
        Path to the working .par file that gets overwritten each run.
    run_fn : callable
        Function called as run_fn(**run_kwargs). Prefer
        ``Utils.multislice_for_sweep``, which returns the
        ``analyze_multislice`` dict. Legacy callables that return a sequence
        (e.g. ``Mains.general``) are still supported via ``result_index`` /
        RMSE ``metric`` name mapping.
    run_kwargs : dict or None
        Extra keyword arguments forwarded to run_fn.
    metric : str, default ``rmse_unseen``
        Outcome to store. For analyze-dict returns, one of
        ``rmse_train``, ``rmse_validation``, ``rmse_unseen``,
        ``rmse_persistence``, ``quadratic_loss``, ``smooth_loss``,
        ``tick_loss``, ``firm_loss``. For sequence returns, RMSE names map to
        tuple indices unless ``result_index`` is set.
    result_index : int or None, default None
        If ``run_fn`` returns a sequence, override which element to store.
        When None, RMSE ``metric`` names map to indices (unseen→2, etc.);
        other metrics require a dict return from ``run_fn``.

    Returns
    -------
    results : dict  {tuple of param values -> float}
        Keys are tuples in the same order as sweep_spec keys. Values are the
        chosen metric (default: rmse_unseen).
    """
    from itertools import product

    if run_kwargs is None:
        run_kwargs = {}

    names = list(sweep_spec.keys())
    value_lists = [sweep_spec[n] for n in names]
    combos = list(product(*value_lists))

    n_total = len(combos)
    results = {}
    elapsed_times = []
    print(f"Parameter sweep: {n_total} combinations to run (metric={metric})")
    for i, combo in enumerate(combos):
        t_start = time.time()

        params = load_params(default_paramfile)
        for name, val in zip(names, combo):
            params[name] = val
        write_params(paramfile, params)

        out = run_fn(**run_kwargs)
        if isinstance(out, dict):
            score = scalar_from_analyze(out, metric)
        else:
            if result_index is not None:
                idx = result_index
            elif metric in _SWEEP_METRIC_TO_INDEX:
                idx = _SWEEP_METRIC_TO_INDEX[metric]
            else:
                raise TypeError(
                    f"metric={metric!r} requires run_fn to return an analyze "
                    f"dict (use Utils.multislice_for_sweep); got "
                    f"{type(out).__name__!r}")
            try:
                score = out[idx]
            except (TypeError, IndexError) as e:
                raise TypeError(
                    f"run_fn must return a sequence with index {idx}; "
                    f"got {type(out).__name__!r}"
                ) from e
        results[combo] = score

        runtime = time.time() - t_start
        elapsed_times.append(runtime)
        avg_dt = sum(elapsed_times) / len(elapsed_times)
        remaining = avg_dt * (n_total - (i + 1))
        mins, secs = divmod(int(remaining), 60)
        print(f"  [{i+1}/{n_total}] {dict(zip(names, combo))} "
              f"-> {metric} = {score:.4e}  ({runtime:.1f}s, ~{mins}m{secs:02d}s left)")

    return results


def probe_calendar_span(timeseries_file, tickers, paramfile = None) :
    """
    Calendar feature-row span for ``Mains.multislice`` block sizing.

    Uses the same anchor as ``Mains.multislice``: ``calendar_start = W - 1`` where
    ``W = feature_align(cfg)``. Pass ``paramfile`` when the active feature set is not
    ``mean_derivative`` (e.g. ``return_variance`` with ``variance_window``).
    """
    import FeatureGeneration as FG
    if paramfile is not None :
        cfg = FG.feature_config(paramfile)
        feature_func = cfg.func
        W = FG.feature_align(cfg)
    else :
        feature_func = None
        W = FG.ROLLING_WINDOW_N
    per_ticker, _ = FG.load_tickers_features(
        timeseries_file, tickers, feature_func=feature_func, feature_align=W)
    calendar_start = W - 1
    max_calendar_end = max(
        d["first_valid_feat_row"] + d["feats"].shape[0]
        for d in per_ticker.values())
    calendar_span = int(max_calendar_end - calendar_start)
    return {
        "calendar_start"   : calendar_start,
        "max_calendar_end" : int(max_calendar_end),
        "calendar_span"    : calendar_span,
    }


def scale_pats_for_n_blocks(n_blocks, trnpat0, vanpat0, tenpat0,
                            calendar_span, offs = 1, warmup_pat = 0) :
    """
    Scale per-ticker per-block ``trnpat/vanpat/tenpat`` to fit ``n_blocks``
    blocks in ``calendar_span`` feature rows, preserving phase ratios.

    Returns (trnpat, vanpat, tenpat, block_step).
    """
    n_blocks = int(n_blocks)
    if n_blocks < 1 :
        raise ValueError(f"n_blocks must be >= 1 (got {n_blocks})")
    offs = int(offs)
    warmup_pat = int(warmup_pat)
    min_phase = offs + 1
    if warmup_pat > 0 :
        min_phase = max(min_phase, warmup_pat + offs + 1)
    block_step = int(calendar_span) // n_blocks
    if block_step < 3 * min_phase :
        raise ValueError(
            f"n_blocks={n_blocks} gives block_step={block_step}, but each of "
            f"trnpat/vanpat/tenpat must be > offs={offs} (need block_step >= "
            f"{3 * min_phase}).")

    total = int(trnpat0) + int(vanpat0) + int(tenpat0)
    if total <= 0 :
        raise ValueError("trnpat0 + vanpat0 + tenpat0 must be positive")

    trnpat = max(min_phase, int(round(block_step * int(trnpat0) / total)))
    vanpat = max(min_phase, int(round(block_step * int(vanpat0) / total)))
    tenpat = block_step - trnpat - vanpat
    if tenpat < min_phase :
        raise ValueError(
            f"n_blocks={n_blocks}: scaled tenpat={tenpat} is < min_phase={min_phase} "
            f"(trnpat={trnpat}, vanpat={vanpat}, block_step={block_step}).")
    return trnpat, vanpat, tenpat, block_step


def _nanmean_cube(arr) :
    """NaN-mean over all finite entries in an RMSE/persistence cube slice."""
    flat = np.asarray(arr, dtype = float).ravel()
    m = np.isfinite(flat)
    return float(np.mean(flat[m])) if m.any() else float("nan")


def block_sweep(
    n_blocks_list,
    default_paramfile,
    paramfile,
    run_fn = None,
    run_kwargs = None,
    result_phase = 2,
) :
    """
    Sweep target block count by scaling ``trnpat/vanpat/tenpat`` together.

    For each ``n_blocks``, ``block_step = calendar_span // n_blocks`` and
    phase lengths keep the ratios from ``default_paramfile``. Each run calls
    ``Mains.multislice`` (by default) with the scaled pattern counts.

    Parameters
    ----------
    n_blocks_list : sequence of int
        Target number of walk-forward blocks (e.g. ``[1, 2, 4, 8, 16, 32]``).
    default_paramfile : str
        Baseline ``.par`` (provides ratios, tickers, timeseries, hyperparams).
    paramfile : str
        Scratch ``.par`` overwritten on every evaluation.
    run_fn : callable or None
        ``run_fn(paramfile=..., **run_kwargs)``; defaults to ``Mains.multislice``.
    run_kwargs : dict or None
        Forwarded to ``run_fn`` (e.g. ``exefile``, ``verbose``).
    result_phase : int
        Phase index for ``results`` dict (0=train, 1=val, 2=test).

    Returns
    -------
    dict with keys ``results``, ``details``, ``calendar_span``.
        ``results`` maps ``(n_blocks,)`` tuples to scalar RMSE for
        ``result_phase``, compatible with ``Display.plot_sweep_heatmaps``.
    """
    if run_fn is None :
        import Mains
        run_fn = Mains.multislice
    if run_kwargs is None :
        run_kwargs = {"exefile" : "./ltslmain", "verbose" : False}

    base = load_params(default_paramfile)
    import FeatureGeneration as FG
    if "tickers" not in base:
        base["tickers"]=FG.csv_ticker_columns(base["timeseries_file"])
    
    for required in ("timeseries_file", "tickers", "trnpat", "vanpat", "tenpat", "offs") :
        if required not in base :
            raise KeyError(
                f"'{default_paramfile}' missing '{required}' required by block_sweep().")

    trn0 = int(base["trnpat"])
    van0 = int(base["vanpat"])
    ten0 = int(base["tenpat"])
    offs = int(base["offs"])
    warmup_pat = int(base.get("warmup_pat", 0))
    span_info = probe_calendar_span(
        base["timeseries_file"], base["tickers"], paramfile=default_paramfile)
    calendar_span = span_info["calendar_span"]

    targets = [int(n) for n in n_blocks_list]
    n_total = len(targets)
    results = {}
    details = []
    elapsed_times = []

    print(f"Block sweep: {n_total} n_blocks values, calendar_span={calendar_span} "
          f"(base ratios {trn0}:{van0}:{ten0})")

    for i, n_blocks in enumerate(targets) :
        t_start = time.time()
        try :
            trnpat, vanpat, tenpat, block_step = scale_pats_for_n_blocks(
                n_blocks, trn0, van0, ten0, calendar_span, offs = offs,
                warmup_pat = warmup_pat)
        except ValueError as e :
            print(f"  [{i+1}/{n_total}] n_blocks={n_blocks} -> SKIP ({e})")
            continue

        params = dict(base)
        params["trnpat"] = trnpat
        params["vanpat"] = vanpat
        params["tenpat"] = tenpat
        write_params(paramfile, params, preserve_comments = False)

        try :
            out = run_fn(paramfile = paramfile, **run_kwargs)
        except Exception as e :
            print(f"  [{i+1}/{n_total}] n_blocks={n_blocks} -> FAILED ({e})")
            continue

        rmse_cube = np.asarray(out["rmse"], dtype = float)
        pers_cube = np.asarray(out["persistence"], dtype = float)
        rmse_train = _nanmean_cube(rmse_cube[:, 0, :])
        rmse_val = _nanmean_cube(rmse_cube[:, 1, :])
        rmse_test = _nanmean_cube(rmse_cube[:, 2, :])
        rmse_pers = _nanmean_cube(pers_cube[:, 1, :])
        phase_vals = (rmse_train, rmse_val, rmse_test)
        metric = phase_vals[result_phase]
        runtime = time.time() - t_start

        results[(n_blocks,)] = metric
        details.append({
            "n_blocks"        : n_blocks,
            "actual_n_blocks" : int(out["n_blocks"]),
            "block_step"      : int(block_step),
            "trnpat"          : trnpat,
            "vanpat"          : vanpat,
            "tenpat"          : tenpat,
            "rmse_train"      : rmse_train,
            "rmse_val"        : rmse_val,
            "rmse_test"       : rmse_test,
            "rmse_persistence": rmse_pers,
            "elapsed_s"       : float(runtime),
        })

        elapsed_times.append(runtime)
        avg_runtime = sum(elapsed_times) / len(elapsed_times)
        remaining = avg_runtime * (n_total - (i + 1))
        mins, secs = divmod(int(remaining), 60)
        print(f"  [{i+1}/{n_total}] n_blocks={n_blocks}  "
              f"step={block_step}  pats={trnpat}/{vanpat}/{tenpat}  "
              f"actual={out['n_blocks']}  "
              f"rmse_test={rmse_test:.4e}  ({runtime:.1f}s, ~{mins}m{secs:02d}s left)")

    return {
        "results"       : results,
        "details"       : details,
        "calendar_span" : calendar_span,
    }

def ticker_count_sweep(
    n_tickers_list,
    default_paramfile,
    paramfile,
    run_fn = None,
    run_kwargs = None,
    result_phase = 2,
    ticker_pool = None,
) :
    """
    Sweep how many tickers are included in ``Mains.multislice``.

    For each ``n_tickers``, the first ``n`` symbols from ``ticker_pool`` are
    written to the ``tickers`` parameter. By default ``ticker_pool`` is the
    column order of ``timeseries_file`` (use ``sp100_IPO_sorted.csv`` so
    full-history names precede late-IPO names). ``trnpat/vanpat/tenpat`` stay fixed.

    Parameters
    ----------
    n_tickers_list : sequence of int
        Target ticker counts (e.g. ``list(range(2, 81))``).
    default_paramfile : str
        Baseline ``.par`` (provides pattern counts, timeseries, hyperparams).
    paramfile : str
        Scratch ``.par`` overwritten on every evaluation.
    run_fn : callable or None
        ``run_fn(paramfile=..., **run_kwargs)``; defaults to ``Mains.multislice``.
    run_kwargs : dict or None
        Forwarded to ``run_fn`` (e.g. ``exefile``, ``verbose``).
    result_phase : int
        Phase index for ``results`` dict (0=train, 1=val, 2=test).
    ticker_pool : sequence of str or None
        Ordered ticker universe; defaults to ``FeatureGeneration.csv_ticker_columns``
        for ``timeseries_file`` in ``default_paramfile``.

    Returns
    -------
    dict with keys ``results``, ``details``, ``ticker_pool``, ``n_pool``.
        ``results`` maps ``(n_tickers,)`` tuples to scalar RMSE for
        ``result_phase``, compatible with ``Display.plot_sweep_heatmaps``.
    """
    if run_fn is None :
        import Mains
        run_fn = Mains.multislice
    if run_kwargs is None :
        run_kwargs = {"exefile" : "./ltslmain", "verbose" : False}

    base = load_params(default_paramfile)
    for required in ("timeseries_file", "trnpat", "vanpat", "tenpat") :
        if required not in base :
            raise KeyError(
                f"'{default_paramfile}' missing '{required}' required by "
                f"ticker_count_sweep().")

    import FeatureGeneration as FG
    if ticker_pool is None :
        pool = FG.csv_ticker_columns(base["timeseries_file"])
    else :
        pool = list(ticker_pool)
    n_pool = len(pool)

    targets = [int(n) for n in n_tickers_list]
    n_total = len(targets)
    results = {}
    details = []
    elapsed_times = []

    print(f"Ticker-count sweep: {n_total} values, pool={n_pool} tickers "
          f"(fixed pats {base['trnpat']}/{base['vanpat']}/{base['tenpat']})")

    for i, n_tickers in enumerate(targets) :
        t_start = time.time()
        if n_tickers < 1 :
            print(f"  [{i+1}/{n_total}] n_tickers={n_tickers} -> SKIP (must be >= 1)")
            continue
        if n_tickers > n_pool :
            print(f"  [{i+1}/{n_total}] n_tickers={n_tickers} -> SKIP "
                  f"(pool has only {n_pool} tickers)")
            continue

        tickers = pool[:n_tickers]
        params = dict(base)
        params["tickers"] = ",".join(tickers)
        write_params(paramfile, params, preserve_comments = False)

        try :
            out = run_fn(paramfile = paramfile, **run_kwargs)
        except Exception as e :
            print(f"  [{i+1}/{n_total}] n_tickers={n_tickers} -> FAILED ({e})")
            continue

        rmse_cube = np.asarray(out["rmse"], dtype = float)
        pers_cube = np.asarray(out["persistence"], dtype = float)
        rmse_train = _nanmean_cube(rmse_cube[:, 0, :])
        rmse_val = _nanmean_cube(rmse_cube[:, 1, :])
        rmse_test = _nanmean_cube(rmse_cube[:, 2, :])
        rmse_pers = _nanmean_cube(pers_cube[:, 1, :])
        phase_vals = (rmse_train, rmse_val, rmse_test)
        metric = phase_vals[result_phase]
        runtime = time.time() - t_start

        results[(n_tickers,)] = metric
        actual_n = len(out["tickers"])
        details.append({
            "n_tickers"       : n_tickers,
            "actual_n_tickers": actual_n,
            "tickers"         : ",".join(tickers),
            "rmse_train"      : rmse_train,
            "rmse_val"        : rmse_val,
            "rmse_test"       : rmse_test,
            "rmse_persistence": rmse_pers,
            "elapsed_s"       : float(runtime),
        })

        elapsed_times.append(runtime)
        avg_runtime = sum(elapsed_times) / len(elapsed_times)
        remaining = avg_runtime * (n_total - (i + 1))
        mins, secs = divmod(int(remaining), 60)
        print(f"  [{i+1}/{n_total}] n_tickers={n_tickers}  "
              f"rmse_test={rmse_test:.4e}  ({runtime:.1f}s, ~{mins}m{secs:02d}s left)")

    return {
        "results"     : results,
        "details"     : details,
        "ticker_pool" : pool,
        "n_pool"      : n_pool,
    }


def loadbin(filename,N = None,npat = None,dtype = np.float32) :
    # Loads binary file (with header soon)

    if N==None :

        data = np.fromfile(filename,dtype)

    else :

        if npat==None : 

            data = np.fromfile(filename,dtype)

            if len(data)%N!=0 :
                print(len(data))
                raise (AssertionError("len(data) -- N mismatch"))

            npat = len(data)/N

        else :

            data = np.fromfile(filename,dtype,count = N * npat)
        
        npat = int(npat)

        N = int(N)

        data = data.reshape(npat,N)

    return data


def savebin(arr,filename,dtype = np.float32) :

    # a = array(arr,type(arr[0,0]))
    # output_file = open(filename,'wb')
    # a.tofile(output_file)
    # output_file.close()

    with open(filename,'wb') as fp: np.array(arr,dtype=dtype).tofile(fp)



def loadcsv(filename) :
    datarows = []
    with open(filename,'rb') as csvfile:
        csvreader = csv.reader(csvfile) # ,delimiter=' ',quotechar='#')
        for row in csvreader:
            datarows.append(row)
    return np.array(datarows).astype(np.float32)


def loadandsaveMNISThid() :

#     trdata = loadcsv("train-20-30-10000.csv")
#     tedata = loadcsv("test-20-30-10000.csv")

    trdata = loadcsv("patt.train-classifier.csv")
    tedata = loadcsv("patt.test-classifier.csv")

    np.savetxt("mnist_trhid_60k.txt",trdata)
    np.savetxt("mnist_tehid_10k.txt",tedata)


"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

import struct

def loadMNIST(dataset = "training",what = "lbl",path = ".",savefname = None) :
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.

    But skipping img here ... and iterator as well.

    """

    if dataset=="training":
        if what=="all" or what=="both" :
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif what=="img" :
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = None
        elif what=="lbl" :
            fname_img = None
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset=="testing":
        if what=="all" or what=="both" :
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        elif what=="img" :
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = None
        elif what=="lbl" :
            fname_img = None
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    flbl = None
    if fname_lbl!=None :
        with open(fname_lbl, 'rb') as flblfp:
            magic, num = struct.unpack(">II", flblfp.read(8))
            lbl = np.fromfile(flblfp, dtype=np.int8).astype(int)

        flbl = np.zeros((len(lbl),10))

        for i,x in zip(lbl,flbl) : x[i] = 1

    fimg = None
    if fname_img!=None :
        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        fimg = img.reshape(img.shape[0],img.shape[1]*img.shape[2]).astype(float)

        fimg = fimg/256.0

    if savefname!=None :
        if dataset=="training" :
            if fname_img!=None : savebin(fimg,savefname)
            if fname_lbl!=None : savebin(flbl,savefname)
        else :
            if fname_img!=None : savebin(fimg,savefname)
            if fname_lbl!=None : savebin(flbl,savefname)

    return fimg,flbl


def loadandsaveMNISTlbl() :

    trxlbl = loadMNIST("training")
    texlbl = loadMNIST("testing")

    np.savetxt("mnist_trlbl.txt",trxlbl)
    np.savetxt("mnist_telbl.txt",texlbl)


def loaddata(csvfile,savefile = None) :

    data = loadcsv(csvfile).astype(int)
    
    xlbl = np.zeros((len(data),10))

    for i,x in zip(data,xlbl) : x[i] = 1

    if savefile!=None : 
        np.savetxt(savefile,data)

    return xlbl


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def getncore1(ipopnpart,hpopnpart,opopnpart = 1,useih = True,usehi = False,usehhi = False,usehhe = False,
              useho = True,usehno = False) :

    ncore = ipopnpart + hpopnpart + opopnpart
    if useih : ncore += ipopnpart * hpopnpart
    if usehi : ncore += ipopnpart + ipopnpart * hpopnpart
    if usehhi : ncore += hpopnpart * hpopnpart
    if usehhe : ncore += hpopnpart * hpopnpart
    if useho : ncore += hpopnpart * opopnpart
    if usehno : ncore += hpopnpart * opopnpart

    nnode = int(np.ceil(ncore/32.))

    return nnode,ncore


def getncore2(ipopnpart = 1,hpopnpart = 1) :

    nproc = ipopnpart + hpopnpart + 1;
    nproc += ipopnpart * hpopnpart;
    nproc += hpopnpart * 1;

    print("Minimum n:o cores = %d, n:o nodes = %d (maxnproc = %d)" % \
        (nproc,np.ceil(nproc/32.),np.ceil(nproc/32.)*32))


def factors(n):    
    result = set()
    for i in range(1, int(n ** 0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result |= {i, div}
    return sorted(result)


def checknproc(ipopnpart = 1,hpopnpart = 1,hhiprjp = True,hheprjp = True,avtrgain = 1) :

    nproc = ipopnpart + hpopnpart + 1
    nproc += ipopnpart * hpopnpart
    if hheprjp : nproc += hpopnpart * hpopnpart
    if hhiprjp : nproc += hpopnpart * hpopnpart
    nproc += hpopnpart * 1;
    if avtrgain!=0 :
        nproc += 1
        nproc += hpopnpart * 1

    print("Minimum n:o cores = %d and nodes = %d" % (nproc,np.ceil(nproc/32.)))


def bellrf(x,m,s) :

    x = (x - m)/s * (x - m)/s

    return np.exp(-x)


def intrfcode(data2,nint,kdx = 1,savefname = "") :

    xmin = np.min(data2,0)
    xmax = np.max(data2,0)

    nrow = data2.shape[0]
    ncol = data2.shape[1]

    rfdata2 = np.zeros((nrow,ncol*(nint+1)))

    for d in range(ncol) :

        dx = 1./nint

        x = (data2[:,d] - xmin[d])/(xmax[d] - xmin[d]);

        for b in range(nint+1) :

            rfdata2[:,d*(nint+1) + b] = bellrf(x,b*dx,kdx*dx)

    if savefname!="" : savebin(rfdata2,savefname)

    return rfdata2;



def imshow(data2,interpolation='none',aspect='auto',axes = None,cmap = 'jet',origin = 'upper',extent = None,figno = 1,
           vmin = None,vmax = None,clr = True) :

    if figno<=0 : return

    modulename = 'matplotlib'
    if modulename not in sys.modules:
        print("Module has not been imported".format(modulename))
        return

    if axes!=None :

        im = axes.imshow(data2,interpolation = interpolation,aspect = aspect,cmap = cmap,
                         origin = origin, extent = extent, vmin = vmin,vmax = vmax)

    else :
        
        plt.figure(figno)

        if clr : plt.clf()

        im = plt.imshow(data2,interpolation = interpolation,aspect = aspect,cmap = cmap,
                        extent = extent, vmin = vmin,vmax = vmax)

    return im

    
def plot(data,r0 = 0,r1 = -1,figno = 1,axes = None,clr = True) :

    if r1<0 : r1 = len(data)

    plt.figure(figno)

    if axes!=None :

        if len(data.shape)==1 :

            res = axes.plot(data[r0:r1])

        elif len(data.shape)==2 :

            res = axes.plot(data[r0:r1,0],data[r0:r1,1])

        else : raise(AssertionError, "data.dim must be in {1,2}")

    else :

        if clr : plt.clf()

        if len(data.shape)==1 :

            res = plt.plot(data[r0:r1])

        elif len(data.shape)==2 :

            res = plt.plot(data[r0:r1,0],data[r0:r1,1])

        else : raise(AssertionError, "data.dim must be in {1,2}")


def plotxy(xdata,ydata,r0 = 0,r1 = -1,figno = 1,axes = None,clr = True) :

    if len(xdata.shape)>1 : raise(AssertionError, "xdata.dim >1")

    if len(xdata)!=len(ydata) : raise(AssertionError, "xdata - ydata len mismatch")

    if r1<0 : r1 = len(ydata)

    plt.figure(figno)
    if clr : plt.clf()

    if axes!=None :

        res = axes.plot(xdata[r0:r1],ydata[r0:r1])

    else :

        res = plt.plot(xdata[r0:r1],ydata[r0:r1])


def histo(data,nbin = 40,figno = 1,axes = None,clr = True) :

    plt.figure(figno)
    
    if axes!=None :

        axes.hist(data.flatten(),bins = nbin);

    else :

        if clr : plt.clf()

        plt.hist(data.flatten(),bins = nbin);


def prnintarr(x, fmt = '{:d}', maxlinewidth = 100, endstr = "\n") :
    print(np.array2string(x, formatter={'int_kind': fmt.format}, max_line_width = maxlinewidth), end = endstr)

def prnfltarr(x,fmt = '{:6.3f}', endstr = "\n") :
    print(np.array2string(x, formatter={'float_kind': fmt.format}), end = endstr)


def savefig(fig, filename) :
    pieces = filename.split(".")
    if pieces[-1] != "pkl" :
        filename += ".pkl"

    # Save the figure as a Python object
    with open(filename, "wb") as f:
        pickle.dump(fig, f)


def loadfig(filename) :
    pieces = filename.split(".")
    if pieces[-1] != "pkl" :
        filename += ".pkl"

    with open(filename, "rb") as f:
        fig = pickle.load(f)

    return fig
