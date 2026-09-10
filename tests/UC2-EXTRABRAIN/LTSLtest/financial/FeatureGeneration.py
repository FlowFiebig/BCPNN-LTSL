import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from functools import lru_cache, partial
from dataclasses import dataclass
from typing import Callable

import Utils

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# Rolling window size for finite-difference features in ``make_mean_derivative_features``.
ROLLING_WINDOW_N = 7


def dataset_path(name) :
    """
    Resolve a timeseries filename to a path inside the ``datasets`` folder that
    sits next to this module.

    If ``name`` already contains a directory separator it is treated as an
    explicit path and returned unchanged (so callers can still override the
    location). Otherwise the datasets folder is ensured to exist and the
    composed path is returned.
    """
    name = str(name)
    if os.path.sep in name or (os.altsep and os.altsep in name) :
        return name
    os.makedirs(DATASETS_DIR, exist_ok = True)
    return os.path.join(DATASETS_DIR, name)


def timeseries_from_dataset(data) :
    """
    Return the 1D value series from a dataset pickle (``ndarray``).

    If ``x_norm`` is present it is returned (same convention as OHLCV and
    Mackey-Glass pickles from ``ohlcv_close`` / ``mackey_glass`` with default
    ``normalize=True``): values scaled so ``max(x_norm) <= 1`` with ``x_max``
    stored for de-scaling.

    Legacy pickles without ``x_norm`` (old Mackey-Glass or sum-of-sinusoids files)
    fall back to raw ``x``.
    """
    if "x_norm" in data :
        return np.asarray(data["x_norm"], dtype = float)
    if "x" in data :
        return np.asarray(data["x"], dtype = float)
    return None


def mackey_glass(beta = 0.2, gamma = 0.1, n = 10, tau = 17, x0 = 1.2, tmax = 10000, dt = 1.,
                 savefile = "mgdata.pkl", normalize = True) :
    """
    Generate Mackey-Glass time series using Euler integration.
    Note that I cut off some transients (first 500 steps) by default, but then add that number to tmax as well, such that i still retrive the desired length

    Parameters:
        beta, gamma, n, tau: Mackey-Glass parameters
        #n sometimes referred to as c in the literature. 
        #Beta is sometimes referred to as alpha. gamma is sometimes referred to as beta.
        x0      : initial value
        tmax   : total simulation time
        dt      : time step
        normalize : if True (default), also store ``x_norm = x / max(x)`` and
            ``x_max`` (same max scaling as ``ohlcv_close``). ``timeseries_from_dataset``
            prefers ``x_norm`` when present.

    Writes:
        savefile : pickle with raw ``t``, ``x``, MG parameters, and when
            ``normalize`` is True also ``x_norm`` and ``x_max``.
    """
    #Transient cutoff
    transient_cutoff = 500
    # Number of steps
    steps = int((tmax+transient_cutoff) / dt) # generating extra steps, so we can remove the initial transient
    # Delay in number of steps
    delay_steps = int(tau / dt)

    # Storage for time series
    x = np.zeros(steps)
    x[:delay_steps] = x0   # initialize history

    # Time array
    t = np.linspace(0, tmax, int(tmax / dt)) #time array still counts from 0 to tmax, even if we drop the transient entries at the start

    # Generate series
    for i in range(delay_steps, steps - 1):
        x_tau = x[i - delay_steps]  # delayed value
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x[i]
        x[i+1] = x[i] + dx * dt

    t = t[:tmax]
    x = x[transient_cutoff:tmax+transient_cutoff]

    mg_data = {"beta": beta, "gamma": gamma, "n": n, "tau":tau, "x0": x0, "tmax": tmax, "dt": dt, "t": t, "x": x}
    if normalize:
        x_max = float(np.max(x))
        if x_max <= 0.0:
            raise ValueError("Mackey-Glass series has non-positive max; cannot normalize")
        mg_data["x_norm"] = x / x_max
        mg_data["x_max"] = x_max
    savefile = dataset_path(savefile)
    pickle.dump(mg_data, open(savefile, "wb"))
    # mgdata = pickle.load( open( savefile, "rb" ))
    # mgdata["n"]

    return


def sum_of_sinusoids(omegas = None, tmax = 1000, dt = 1., savefile = "sosdata.pkl") :
    """
    Sum of sinusoids x(t) = (1/N) * sum_k sin(omega_k * t) on a uniform time grid.

    Parameters:
        omegas: angular frequencies (same units as 1/time as dt). Non-empty list or array.
        tmax: duration; samples span [0, tmax) with step dt (see np.arange(0, tmax, dt)).
        dt: sample interval; must be positive.
    Writes:
        savefile: pickle with keys omegas, tmax, dt, t, x (len(x) == len(t)).
    """
    if omegas is None:
        omegas = [1]
    if dt <= 0:
        raise ValueError("dt must be positive")
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    omegas = [float(w) for w in omegas]
    if len(omegas) == 0:
        raise ValueError("omegas must be non-empty")

    t = np.arange(0, tmax, dt, dtype = float)
    N = len(omegas)
    x = np.zeros_like(t)
    for omega in omegas:
        x += (1.0 / N) * np.sin(omega * t)

    sos_data = {"omegas": omegas, "tmax": float(tmax), "dt": float(dt), "t": t, "x": x}
    savefile = dataset_path(savefile)
    pickle.dump(sos_data, open(savefile, "wb"))


def ohlcv_close(csv_file, savefile = None, sep = "\t", dt = 1.0) :
    """
    Extract the Close column from an OHLCV CSV and save it as a pickle whose
    time axis matches ``mackey_glass`` / ``sum_of_sinusoids`` style; use
    ``timeseries_from_dataset`` to read the value series (``x_norm`` here).

    Expected CSV layout (no header):
        Date, Open, High, Low, Close, Volume
    with ``sep`` as the field separator (tab by default).

    Parameters:
        csv_file: path to the source CSV. A bare filename is looked up via
            ``dataset_path`` (i.e. inside the ``datasets`` folder).
        savefile: output pickle filename; defaults to ``<csv stem>.pkl``.
            Also resolved via ``dataset_path``.
        sep: field separator in the CSV.
        dt: nominal timestep used for the generated time axis.

    Writes:
        savefile: pickle dict with keys ``source_csv``, ``dates``, ``tmax``,
            ``dt``, ``t``, ``x_norm``, ``x_max``. ``x_norm`` is the Close series
            divided by its maximum (so ``max(x_norm) == 1``); ``x_max`` is that
            maximum (original scale). ``t`` is an evenly spaced time axis with
            spacing ``dt``.

    Returns:
        Path to the written pickle.
    """
    csv_path = dataset_path(csv_file)
    if savefile is None :
        savefile = os.path.splitext(os.path.basename(csv_file))[0] + ".pkl"
    df = pd.read_csv(csv_path, sep = sep, header = None,
                     names = ["Date", "Open", "High", "Low", "Close", "Volume"])
    x_raw = np.asarray(df["Close"].values, dtype = float)
    x_max = float(np.max(x_raw))
    if x_max <= 0.0 :
        raise ValueError("Close series has non-positive max; cannot normalize")
    x_norm = x_raw / x_max
    t = np.arange(len(x_norm), dtype = float) * float(dt)
    dates = df["Date"].tolist()
    data = {"source_csv": os.path.basename(csv_file),
            "dates": dates,
            "tmax": float(len(x_norm)) * float(dt),
            "dt": float(dt),
            "t": t,
            "x_norm": x_norm,
            "x_max": x_max}
    savefile = dataset_path(savefile)
    pickle.dump(data, open(savefile, "wb"))
    return savefile


def mixed_forex_pkl(components = ("AUDUSD", "GBPUSD", "JPYUSD", "EURUSD"),
                    savefile = "MixedFOREX.pkl", dt = 1.0) :
    """
    Build a pickle by concatenating each pair's ``x_norm`` (per-pair max-normalized
    closes) in order—no de-normalization and no global re-scaling. ``dates`` are
    concatenated the same way. ``x_max`` is ``max`` of the combined series (1.0
    when each segment attains 1). Extra key ``components`` lists the stems used.
    """
    chunks = []
    dates_all = []
    for stem in components :
        path = dataset_path(stem + ".pkl")
        with open(path, "rb") as f :
            d = pickle.load(f)
        chunks.append(np.asarray(d["x_norm"], dtype = float))
        dates_all.extend(d["dates"])
    x_norm = np.concatenate(chunks, axis = 0)
    x_max = float(np.max(x_norm))
    if x_max <= 0.0 :
        raise ValueError("Combined series has non-positive max")
    t = np.arange(len(x_norm), dtype = float) * float(dt)
    data = {"source_csv": "MixedFOREX",
            "components": tuple(components),
            "dates": dates_all,
            "tmax": float(len(x_norm)) * float(dt),
            "dt": float(dt),
            "t": t,
            "x_norm": x_norm,
            "x_max": x_max}
    savefile = dataset_path(savefile)
    pickle.dump(data, open(savefile, "wb"))
    return savefile


def parse_ticker_list(tickers_string) :
    """
    Parse a comma-separated ticker/column list ``"AAPL,MSFT,..."``.

    Whitespace around names is tolerated. The value must be whitespace-free in
    the ``.par`` file (``Utils.findparamval`` constraint).
    """
    if not tickers_string or not str(tickers_string).strip() :
        raise ValueError("Empty tickers list")
    cols = [c.strip() for c in str(tickers_string).split(",") if c.strip()]
    if not cols :
        raise ValueError(f"Empty tickers list parsed from '{tickers_string}'")
    return cols


def wide_csv_stem(timeseries_file) :
    """
    Return the wide-CSV stem referenced by ``timeseries_file``.

    Accepts ``sp100_daily_prices.csv``, ``sp100_daily_prices``, or a pickle
    name such as ``sp100_daily_prices.pkl`` (stem only; the CSV sibling is
    resolved separately via :func:`dataset_path`).
    """
    return os.path.splitext(os.path.basename(str(timeseries_file)))[0]


def wide_csv_path(timeseries_file) :
    """Resolve ``timeseries_file`` to an on-disk wide CSV under ``datasets/``."""
    stem = wide_csv_stem(timeseries_file)
    csv_path = dataset_path(stem + ".csv")
    if not os.path.exists(csv_path) :
        raise FileNotFoundError(
            f"Wide CSV not found for timeseries_file '{timeseries_file}': "
            f"expected {csv_path}")
    return csv_path, stem


def csv_ticker_columns(timeseries_file) :
    """
    Ticker column names from a wide CSV, in file column order (excluding ``Date``).

    For ``sp100_IPO_sorted.csv``, full-history names come first and late-IPO
    names are appended at the end — use ``columns[:n]`` to take the top ``n`` tickers.
    """
    csv_path, _ = wide_csv_path(timeseries_file)
    df = pd.read_csv(csv_path, nrows = 0)
    return [c for c in df.columns if c != "Date"]


def build_ipo_sorted_csv(src_timeseries_file = "sp100_daily_prices.csv",
                         dst_timeseries_file = "sp100_IPO_sorted.csv") :
    """
    Write ``datasets/<dst>`` with columns reordered: full-history tickers first
    (alphabetical), then late-IPO tickers (by first valid row, then alphabetical).

    Returns the ordered ticker list written.
    """
    src_path, _ = wide_csv_path(src_timeseries_file)
    dst_path = dataset_path(dst_timeseries_file)

    df = pd.read_csv(src_path)
    if "Date" not in df.columns :
        raise ValueError(f"{src_path} has no 'Date' column")

    def _sort_key(col) :
        s = df[col]
        first = s.first_valid_index()
        if first is None :
            return (2, 10**9, col)
        first = int(first)
        if first == 0 and not s.isna().any() :
            return (0, 0, col)
        return (1, first, col)

    tickers = sorted([c for c in df.columns if c != "Date"], key = _sort_key)
    out = df[["Date"] + tickers]
    out.to_csv(dst_path, index = False)
    return tickers


def build_forex_wide_csv(pair_stems = ("AUDUSD", "EURUSD", "GBPUSD", "JPYUSD"),
                         dst_timeseries_file = "FOREX.csv",
                         sep = "\t") :
    """
    Write ``datasets/<dst>`` as a wide comma-separated CSV (``Date`` + Close
    columns) from tab-separated OHLCV pair files, matching the contract of
    ``sp100_IPO_sorted.csv`` for ``load_tickers_features`` / ``Mains.multislice``.

    Each source file must have layout ``Date, Open, High, Low, Close, Volume``
    with no header row. All pair calendars must match exactly.

    Returns the ordered ticker list written.
    """
    ohlcv_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
    dst_path = dataset_path(dst_timeseries_file)
    master_dates = None
    closes = {}

    for stem in pair_stems :
        csv_path = dataset_path(stem + ".csv")
        df = pd.read_csv(csv_path, sep = sep, header = None, names = ohlcv_names)
        dates = pd.to_datetime(df["Date"])
        if master_dates is None :
            master_dates = dates
        elif not (dates.values == master_dates.values).all() :
            raise ValueError(
                f"{stem}.csv calendar does not match {pair_stems[0]}.csv; "
                f"run align_forex_ohlcv_csvs_on_master_calendar() first.")
        if df["Close"].isna().any() :
            raise ValueError(f"{stem}.csv Close column contains NaNs")
        closes[stem] = df["Close"].values

    out = pd.DataFrame({"Date": master_dates.dt.strftime("%Y-%m-%d")})
    for stem in pair_stems :
        out[stem] = closes[stem]
    out.to_csv(dst_path, index = False)
    return list(pair_stems)


def load_tickers_features(timeseries_file, tickers, cutoff = 0,
                          feature_func = None, feature_align = None) :
    """
    Load every ticker from a wide CSV and return per-ticker rolling features
    aligned on the CSV's calendar.

    Parameters
    ----------
    timeseries_file : str
        Wide CSV filename (e.g. ``sp100_daily_prices.csv``) or stem /
        pickle-style name whose stem maps to ``datasets/<stem>.csv``.
    tickers : str
        Comma-separated column names to load, e.g. ``"AAPL,MSFT"``. If missing,
        empty, or whitespace-only, every ticker column in the wide CSV is used.
    cutoff : int
        Forwarded to the feature generator.
    feature_func : callable or None
        Feature generator applied per ticker (signature ``(x, cutoff, figno)``
        returning ``[T-n+1, 4]``). Defaults to
        :func:`make_mean_derivative_features` for backward compatibility; pass
        ``feature_config(paramfile).func`` to honour the ``feature_set`` key.
    feature_align : int or None
        Leading raw rows consumed before ``feats[0]`` (``first_valid_feat_row =
        first_raw + feature_align - 1``). Defaults to ``ROLLING_WINDOW_N``; pass
        :func:`feature_align` of the matching :class:`FeatureConfig` when using a
        non-default generator (e.g. ``return_variance`` with ``variance_window``).

    Returns
    -------
    per_ticker : ``OrderedDict[label -> dict]``
        Keys are ticker symbols (column names). Each value holds ``feats``,
        ``first_valid_feat_row``, ``x_max``, ``stem``, and ``col``.
    csv_lens : ``dict[stem -> int]``
        Length of the loaded CSV in dated rows.
    """
    from collections import OrderedDict
    if feature_func is None :
        feature_func = make_mean_derivative_features
    if feature_align is None :
        feature_align = ROLLING_WINDOW_N
    csv_path, stem = wide_csv_path(timeseries_file)
    if not tickers or not str(tickers).strip() :
        cols = csv_ticker_columns(timeseries_file)
    else :
        cols = parse_ticker_list(tickers)
    df = pd.read_csv(csv_path)
    csv_lens = {stem : int(len(df))}
    missing = [c for c in cols if c not in df.columns]
    if missing :
        raise KeyError(
            f"Columns missing from {csv_path}: {missing}. "
            f"Available (excluding Date): "
            f"{[c for c in df.columns if c != 'Date'][:10]}...")
    per_ticker = OrderedDict()
    for col in cols :
        s = df[col]
        first = s.first_valid_index()
        if first is None :
            raise ValueError(
                f"Column '{col}' in {csv_path} has no valid (non-NaN) values.")
        trimmed = s.loc[first:]
        x_raw = np.asarray(trimmed.values, dtype = float)
        if np.isnan(x_raw).any() :
            n_nan = int(np.isnan(x_raw).sum())
            raise ValueError(
                f"Column '{col}' in {csv_path} contains {n_nan} interior "
                f"NaNs (gap inside the series); fix the source data.")
        x_max = float(np.max(x_raw))
        if x_max <= 0.0 :
            raise ValueError(
                f"Column '{col}' has non-positive max ({x_max}); "
                f"cannot normalize.")
        x_norm = x_raw / x_max
        feats = feature_func(
            x = x_norm, cutoff = cutoff, figno = 0)
        first_valid_feat_row = int(first) + (feature_align - 1)
        per_ticker[col] = {
            "feats"               : feats,
            "first_valid_feat_row": first_valid_feat_row,
            "x_max"               : x_max,
            "stem"                : stem,
            "col"                 : col,
        }
    return per_ticker, csv_lens


def _write_ohlcv_tab(csv_path, df, sep = "\t", date_fmt = "%Y-%m-%d %H:%M",
                     ohlc_decimals = 8) :
    """Write Date, Open, High, Low, Close, Volume without header (tab-separated)."""
    fmt = "{:." + str(int(ohlc_decimals)) + "f}"
    with open(csv_path, "w") as f :
        for _, r in df.iterrows() :
            f.write(
                f"{r['Date']}{sep}"
                f"{fmt.format(float(r['Open']))}{sep}"
                f"{fmt.format(float(r['High']))}{sep}"
                f"{fmt.format(float(r['Low']))}{sep}"
                f"{fmt.format(float(r['Close']))}{sep}"
                f"{int(r['Volume'])}\n")


def jpyusd_dataframe_from_usdjpy(df) :
    """
    OHLCV USDJPY (yen per dollar) -> JPYUSD (dollars per yen), same row order.
    High/Low swap under inversion. Volume unchanged.
    """
    o = np.asarray(df["Open"], dtype = float)
    h = np.asarray(df["High"], dtype = float)
    l = np.asarray(df["Low"], dtype = float)
    c = np.asarray(df["Close"], dtype = float)
    out = pd.DataFrame({
        "Date":   df["Date"].values,
        "Open":   1.0 / o,
        "High":   1.0 / l,
        "Low":    1.0 / h,
        "Close":  1.0 / c,
        "Volume": df["Volume"].astype(int).values,
    })
    bad = (out["High"] < out["Low"]) | (out["High"] < out[["Open", "Close"]].max(axis = 1)) \
        | (out["Low"] > out[["Open", "Close"]].min(axis = 1))
    if bad.any() :
        mx = out[["Open", "High", "Low", "Close"]].max(axis = 1)
        mn = out[["Open", "High", "Low", "Close"]].min(axis = 1)
        out.loc[bad, "High"] = mx[bad]
        out.loc[bad, "Low"] = mn[bad]
    return out


def align_forex_ohlcv_csvs_on_master_calendar(
        master_stem = "EURUSD",
        interpolate_stems = ("AUDUSD", "GBPUSD", "EURUSD", "USDJPY"),
        sep = "\t",
        ohlc_decimals = 8,
        regenerate_jpyusd_from_usdjpy = True,
        regenerate_pkls = True,
        regenerate_mixed_forex = True) :
    """
    Reindex each OHLCV CSV onto the master's date index and fill gaps with
    time-based linear interpolation (Open, High, Low, Close, Volume). On rows
    that were missing, High/Low are clipped so they bracket Open/Close.

    Defaults use ``EURUSD`` as master (full union of dates in this bundle).
    After aligning ``USDJPY``, optionally rebuild ``JPYUSD.csv`` by exact
    inversion so the pair stays consistent.

    If ``regenerate_pkls`` is True (default), each touched CSV is passed through
    ``ohlcv_close`` so ``.pkl`` caches get ``x_norm`` / ``x_max`` again (raw
    closes live only in the CSV). If ``regenerate_mixed_forex`` is True,
    ``mixed_forex_pkl`` is run last.

    Returns:
        List of written CSV paths (excluding JPYUSD unless regenerated).
    """
    names = ["Date", "Open", "High", "Low", "Close", "Volume"]
    master_path = dataset_path(master_stem + ".csv")
    master_df = pd.read_csv(master_path, sep = sep, header = None, names = names)
    master_df["Date"] = pd.to_datetime(master_df["Date"])
    master_idx = pd.DatetimeIndex(pd.unique(master_df["Date"].sort_values()))
    if len(master_idx) != len(master_df) :
        raise ValueError(f"{master_stem}.csv must have unique dates to be master calendar")

    written = []
    for stem in interpolate_stems :
        csv_path = dataset_path(stem + ".csv")
        df = pd.read_csv(csv_path, sep = sep, header = None, names = names)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        full = df.reindex(master_idx)
        missing = full["Close"].isna()
        full = full.interpolate(method = "time", axis = 0)
        ohlc = full[["Open", "High", "Low", "Close"]]
        mx = ohlc.max(axis = 1)
        mn = ohlc.min(axis = 1)
        full.loc[missing, "High"] = mx[missing].values
        full.loc[missing, "Low"] = mn[missing].values
        full["Volume"] = np.round(full["Volume"]).astype(np.int64)
        if full.isna().any().any() :
            raise RuntimeError(f"Interpolation left NaNs in {stem}.csv")
        out = full.reset_index()
        out = out.rename(columns = {out.columns[0]: "Date"})
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d %H:%M")
        _write_ohlcv_tab(csv_path, out, sep = sep, ohlc_decimals = ohlc_decimals)
        written.append(csv_path)

    if regenerate_jpyusd_from_usdjpy :
        u_path = dataset_path("USDJPY.csv")
        j_names = names
        u_df = pd.read_csv(u_path, sep = sep, header = None, names = j_names)
        jpy = jpyusd_dataframe_from_usdjpy(u_df)
        j_path = dataset_path("JPYUSD.csv")
        _write_ohlcv_tab(j_path, jpy, sep = sep, ohlc_decimals = ohlc_decimals)
        written.append(j_path)

    if regenerate_pkls :
        for stem in interpolate_stems :
            ohlcv_close(stem + ".csv", sep = sep)
        if regenerate_jpyusd_from_usdjpy :
            ohlcv_close("JPYUSD.csv", sep = sep)
    if regenerate_mixed_forex :
        mixed_forex_pkl()

    return written


def fornberg_weights(x0: float, x: np.ndarray, m: int) -> np.ndarray:
    n = len(x)
    c = np.zeros((m + 1, n), dtype=float)
    c1 = 1.0
    c4 = x[0] - x0
    c[0, 0] = 1.0

    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0

        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3

            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = (c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1])) / c2
                c[0, i] = (-c1 * c5 * c[0, i - 1]) / c2

            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = (c4 * c[0, j]) / c3

        c1 = c2

    return c[m]


@lru_cache(maxsize=None)
def get_endpoint_weights(n: int, h: float):
    """
    Cached weights for value, 1st and 2nd derivative
    using last n uniformly spaced samples.
    """
    if n < 3:
        raise ValueError("Need n >= 3")
    x0 = 0.0
    x = -h * np.arange(n)  # [0, -h, -2h, ...]
    w0 = fornberg_weights(x0, x, 0)
    w1 = fornberg_weights(x0, x, 1)
    w2 = fornberg_weights(x0, x, 2)
    return w0, w1, w2


def trailing_mean_weights(n, m=None) :
    """
    Weights for simple trailing mean at endpoint using last m points (<= n).
    Applies to reversed window [x_t, x_{t-1}, ...].
    """
    if m is None:
        m = n
    if not (1 <= m <= n):
        raise ValueError("m must satisfy 1 <= m <= n")
    w = np.zeros(n, dtype=float)
    w[:m] = 1.0 / m
    return w


def ewma_weights(n: int, alpha: float) -> np.ndarray:
    """
    One-sided EWMA weights at endpoint over last n points.
    Applies to reversed window [x_t, x_{t-1}, ...].
    alpha in (0,1], larger -> more weight on most recent.
    Normalized to sum to 1 over the truncated n-window.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    j = np.arange(n, dtype=float)         # 0 is most recent
    w = (1.0 - alpha) ** j
    w *= alpha
    w /= w.sum()
    return w


def rolling_features_numpy(y: np.ndarray, w0, w1, w2, wmean) -> np.ndarray:
    """
    y: [T] #timeseries of length T. Previously x, as returned from mackey_glass generator function
    w*: length n weights that apply to reversed window [x_t, x_{t-1}, ...]
    returns: [T-n+1, 4] columns [value, d1, d2, mean]
    """
    y = np.asarray(y, dtype=float)
    n = len(w0)
    if y.size < n:
        raise ValueError("Need at least n samples")
    W = sliding_window_view(y, window_shape=n)  # [T-n+1, n], time order
    Wr = W[:, ::-1]                             # reversed to match weights

    w0 = np.asarray(w0, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    wm = np.asarray(wmean, dtype=float)
    val  = Wr @ w0
    d1   = Wr @ w1
    d2   = Wr @ w2
    mean = Wr @ wm
    return np.stack([val, d1, d2, mean], axis=-1) #note that these are plotted out in a different order


def make_mean_derivative_features(x: np.ndarray, cutoff = 0, figno = 1) :
    """
    Generate rolling features (value, 1st deriv, 2nd deriv, mean) from an input time-series.
    Parameters:
        x      : complete 1D numpy time-series
        cutoff : optional transient cutoff used for robust feature scaling/clamping
                 of the 1st-derivative range. The model pipeline (Mains.general)
                 passes the default cutoff = 0, since the Mackey-Glass generator
                 already strips its own transient internally and the
                 sum-of-sinusoids / on-disk OHLCV series have no transient phase.
                 The notebook exploration cells still pass an explicit cutoff
                 (e.g. 400) when they want robust scaling for visualization.
        figno   : if >0, plot the generated features
    Returns:
        feats : [T-n+1, 4] with columns [value, d1, d2, mean]
    """
    x = np.asarray(x, dtype=float)
    n = ROLLING_WINDOW_N
    h = 0.1 # some other unknown parameter for the endpoint weights

    w0, w1, w2 = get_endpoint_weights(n, h)  # Fornberg finite-difference weights

    # Option A: simple trailing mean over last m points (often best baseline)
    wm = trailing_mean_weights(n, m=n)

    feats = rolling_features_numpy(x, w0, w1, w2, wm)  # [T-n+1, 4]
    
    mean_min = np.min(feats[cutoff:,3]) #cutting off some transients to avoid binns generated for the transients
    mean_max = np.max(feats[cutoff:,3])
    deriv1st_min = np.min(feats[cutoff:,1])
    deriv1st_max = np.max(feats[cutoff:,1])
    idxs = np.where(feats[:,1] < deriv1st_min)
    feats[idxs] = deriv1st_min
    idxs = np.where(feats[:,1] > deriv1st_max)
    feats[idxs] = deriv1st_max
    if figno > 0 :
        fig = plt.figure(figno) # , figsize = (14, 10))
        plt.clf()
        axs = fig.subplots(4,1)

        axs[0].plot(feats[:,0])
        axs[0].plot(feats[:,3]) #ploting the mean on top of te he original timeseries
        axs[1].plot(feats[:,3]) #mean only plot
        axs[1].set_ylim(mean_min, mean_max)
        
        axs[2].plot(feats[:,1]) #1st derivative plot
        axs[2].set_ylim(deriv1st_min, deriv1st_max)
        
        axs[3].plot(feats[:,2]) #2nd derivative plot
        
        axs[0].set_title("vals, mean")
        axs[1].set_title("mean")
        axs[2].set_title("1st deriv")
        axs[3].set_title("2nd deriv")
        plt.suptitle("Mackey-Glass data set", fontsize = 14)

        plt.tight_layout()

    return feats


def make_return_variance_features(x: np.ndarray, cutoff = 0, figno = 1,
                                  variance_window = 7) :
    """
    Generate simple return/variance features from an input time-series.

    Unlike ``make_mean_derivative_features`` this uses no Fornberg / trailing-mean
    filtering; everything is computed inline so the four columns are:
        0 value        : raw value at the window endpoint
        1 delta        : simple finite difference x[t] - x[t-1] (no filtering)
        2 log_return   : log(x[t] / x[t-1]) — RBF-encoded primary channel
        3 variance     : left-sided variance of log returns over the trailing
                         ``variance_window`` (also sets feature/calendar alignment)

    The primary channel is stored in log space; use :func:`delog_primary`
    to map it to the daily return ratio ``exp(log_return)`` for plots and RMSE.

    The output is endpoint-aligned to [T-variance_window+1, 4] (same convention
    and shape as ``make_mean_derivative_features``) so it stays drop-in compatible
    with the Mains / Display pipeline.

    Parameters:
        x               : complete 1D numpy time-series
        cutoff          : optional transient cutoff used only for robust y-limits when
                          plotting (figno > 0); the default cutoff = 0 is a no-op.
        figno           : if >0, plot the generated features
        variance_window : trailing window for log-return variance and endpoint align
    Returns:
        feats : [T-variance_window+1, 4] with columns [value, delta, log_return, variance]
    """
    x = np.asarray(x, dtype=float)
    n = int(variance_window)
    if n < 2 :
        raise ValueError(f"variance_window must be >= 2 (got {n}).")

    # Endpoint-aligned over the last (T - n + 1) days.
    value = x[n - 1:]        # raw value at endpoint t
    prev  = x[n - 2:-1]      # value at t-1, aligned to ``value``

    delta      = value - prev                 # col 1: simple delta, no temporal filtering
    log_return = np.log(value / prev)         # col 2: log return (encoded primary)

    # col 3: trailing variance of log returns; the window is left-sided
    # (uses only the current and previous n-1 days).
    log_ret = np.empty_like(x)
    log_ret[0]  = 0.0                         # synthetic seed; only touches the first window row
    log_ret[1:] = np.log(x[1:] / x[:-1])
    Wlog = sliding_window_view(log_ret, n)    # [T-n+1, n], window ends at endpoint t
    variance = np.var(Wlog, axis=1)           # population variance over the n-day window

    feats = np.stack([value, delta, log_return, variance], axis=-1)  # [T-n+1, 4]

    if figno > 0 :
        delta_min = np.min(feats[cutoff:, 1])
        delta_max = np.max(feats[cutoff:, 1])
        var_min   = np.min(feats[cutoff:, 3])
        var_max   = np.max(feats[cutoff:, 3])

        fig = plt.figure(figno)
        plt.clf()
        axs = fig.subplots(4, 1)

        axs[0].plot(feats[:, 0])  # raw value

        axs[1].plot(np.exp(feats[:, 2]))  # daily return ratio (display)

        axs[2].plot(feats[:, 1])  # simple delta
        axs[2].set_ylim(delta_min, delta_max)

        axs[3].plot(feats[:, 3])  # trailing log-return variance
        axs[3].set_ylim(var_min, var_max)

        axs[0].set_title("value")
        axs[1].set_title("daily return (ratio)")
        axs[2].set_title("delta")
        axs[3].set_title("log-return variance")
        plt.suptitle("return / variance features", fontsize = 14)

        plt.tight_layout()

    return feats


# ---------------------------------------------------------------------------
# Feature-set registry
# ---------------------------------------------------------------------------
# A "feature set" bundles a generator function (returning [T-n+1, 4]) with the
# two columns that get RBF-encoded and the .par keys that hold their bin counts.
#
# The model always encodes exactly two channels into an outer-product pattern:
#   * primary   : the decode / prediction target. It is the OUTER factor of the
#                 pattern and the dimension the C++ ``outpop`` marginalizes down
#                 to (``mean_nbin`` on the C++ side). RMSE / persistence / plots
#                 use this column as ground truth.
#   * secondary : the INNER factor (``deriv1st_nbin`` on the C++ side).
#
# The C++ ``ltslmain`` only consumes two integers (output dim and inner dim) and
# is agnostic to their meaning; it accepts both the legacy ``mean_nbin`` /
# ``deriv1st_nbin`` keys and the ``return_nbin`` / ``variance_nbin`` aliases.

DEFAULT_FEATURE_SET = "mean_derivative"

_FEATURE_SET_SPECS = {
    # columns of make_mean_derivative_features: [value, d1, d2, mean]
    "mean_derivative": dict(
        func               = make_mean_derivative_features,
        primary_col        = 3,   # mean
        secondary_col      = 1,   # 1st derivative
        primary_nbin_key   = "mean_nbin",
        secondary_nbin_key = "deriv1st_nbin",
    ),
    # columns of make_return_variance_features: [value, delta, log_return, variance]
    "return_variance": dict(
        func                 = make_return_variance_features,
        primary_col          = 2,   # log return (prediction target; display via exp)
        secondary_col        = 3,   # trailing log-return variance
        primary_nbin_key     = "return_nbin",
        secondary_nbin_key   = "variance_nbin",
        primary_log_encoded  = True,
    ),
}


VALID_BINMODES = ("rbf", "rbf-quantile")
RBF_WINSOR_TAIL_PCT = 0.25


@dataclass(frozen=True)
class FeatureConfig:
    """Resolved feature-set configuration for one parameter file."""
    name                 : str
    func                 : Callable
    primary_col          : int
    secondary_col        : int
    primary_nbin         : int
    secondary_nbin       : int
    primary_nbin_key     : str
    secondary_nbin_key   : str
    primary_log_encoded  : bool = False
    binmode              : str = "rbf"
    variance_window      : int | None = None


def feature_align(cfg: FeatureConfig) -> int:
    """Raw-row warm-up before ``feats[0]`` for calendar / multislice stitching."""
    if cfg.name == "return_variance":
        return int(cfg.variance_window)
    return ROLLING_WINDOW_N


def feature_config(paramfile) :
    """
    Resolve the :class:`FeatureConfig` for ``paramfile``.

    Reads the optional ``feature_set`` key (default ``mean_derivative`` so legacy
    parameter files keep working), the matching ``*_nbin`` keys, the ``binmode``
    key that selects the RBF centre layout (``rbf`` linear on a 0.25% winsor range vs
    ``rbf-quantile``),
    and for ``return_variance`` the ``variance_window`` key (default 7).
    """
    params = Utils.load_params(paramfile)
    name = str(params.get("feature_set", DEFAULT_FEATURE_SET))
    if name not in _FEATURE_SET_SPECS :
        raise ValueError(
            f"Unknown feature_set '{name}' in '{paramfile}'. "
            f"Valid options: {sorted(_FEATURE_SET_SPECS)}.")
    spec = _FEATURE_SET_SPECS[name]
    for key in (spec["primary_nbin_key"], spec["secondary_nbin_key"]) :
        if key not in params :
            raise KeyError(
                f"feature_set '{name}' requires '{key}' in '{paramfile}'.")
    binmode = str(params.get("binmode", "rbf"))
    if binmode not in VALID_BINMODES :
        raise ValueError(
            f"Unknown binmode '{binmode}' in '{paramfile}'. "
            f"Valid options: {list(VALID_BINMODES)}.")
    variance_window = None
    func = spec["func"]
    if name == "return_variance" :
        variance_window = int(params.get("variance_window", 7))
        if variance_window < 2 :
            raise ValueError(
                f"variance_window must be >= 2 in '{paramfile}' "
                f"(got {variance_window}).")
        func = partial(spec["func"], variance_window=variance_window)
    return FeatureConfig(
        name                 = name,
        func                 = func,
        primary_col          = spec["primary_col"],
        secondary_col        = spec["secondary_col"],
        primary_nbin         = int(params[spec["primary_nbin_key"]]),
        secondary_nbin       = int(params[spec["secondary_nbin_key"]]),
        primary_nbin_key     = spec["primary_nbin_key"],
        secondary_nbin_key   = spec["secondary_nbin_key"],
        primary_log_encoded  = bool(spec.get("primary_log_encoded", False)),
        binmode              = binmode,
        variance_window      = variance_window,
    )


def delog_primary(cfg, values) :
    """Map stored primary-channel values to display units (ratio for log-encoded return)."""
    v = np.asarray(values, dtype=float)
    if cfg.primary_log_encoded :
        return np.exp(v)
    return v


def _make_channel_encoder(col, nbin, binmode) :
    """Build a 1-D RBF encoder for one channel honouring ``binmode``.

    ``rbf``          : centres linearly spaced over the 0.25%/99.75% percentile range,
                       width 1/nbin; raw values are encoded without clipping.
    ``rbf-quantile`` : centres at the empirical (midpoint) quantiles of the data,
                       with adaptive per-centre width.
    """
    import RBFBinner
    if binmode == "rbf" :
        return RBFBinner.Encoder1D.from_linear_winsor(
            col, nbin, RBF_WINSOR_TAIL_PCT)
    if binmode == "rbf-quantile" :
        return RBFBinner.Encoder1D.from_quantiles(col, nbin)
    raise ValueError(
        f"Unknown binmode '{binmode}'. Valid options: {list(VALID_BINMODES)}.")


def primary_encoder(feats, cfg) :
    """Build the RBF encoder/decoder for the primary (decode-target) channel."""
    return _make_channel_encoder(feats[:, cfg.primary_col], cfg.primary_nbin, cfg.binmode)


def build_patterns(feats, cfg) :
    """
    Encode ``feats`` into the outer-product input patterns for ``ltslmain``.

    Returns ``(rbf_primary, patterns_md)`` where ``patterns_md`` has shape
    ``[N, primary_nbin * secondary_nbin]`` with the primary channel as the outer
    factor (so the C++ marginalization recovers the primary distribution). Both
    channels honour ``cfg.binmode``.
    """
    rbf_primary = primary_encoder(feats, cfg)
    sec = feats[:, cfg.secondary_col]
    rbf_secondary = _make_channel_encoder(sec, cfg.secondary_nbin, cfg.binmode)
    pats_primary = rbf_primary.encode(feats[:, cfg.primary_col])
    pats_secondary = rbf_secondary.encode(sec)
    patterns_md = (pats_primary[:, :, None] * pats_secondary[:, None, :]
                   ).reshape(pats_primary.shape[0], -1)
    return rbf_primary, patterns_md


def centers_sidecar_path(infilename_md) :
    """Path of the sidecar storing the primary decode-target bin centres."""
    return os.path.splitext(str(infilename_md))[0] + ".centers.npy"


def save_primary_centers(infilename_md, encoder) :
    """Persist the primary channel's bin centres next to ``features.dat`` (float64)."""
    path = centers_sidecar_path(infilename_md)
    np.save(path, np.asarray(encoder.centers, dtype=float))
    return path


def load_primary_encoder(infilename_md) :
    """Reconstruct the primary decoder from saved centres, or ``None`` if absent.

    The decode path only needs the centres array, so this rebuilds an
    :class:`RBFBinner.Encoder1D` from the persisted centres (works for both linear
    and quantile layouts) and guarantees the decode grid matches what was encoded.
    """
    import RBFBinner
    path = centers_sidecar_path(infilename_md)
    if not os.path.exists(path) :
        return None
    return RBFBinner.Encoder1D.from_centers(np.load(path))
