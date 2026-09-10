import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from functools import lru_cache

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


def load_wide_csv_columns(stem, columns, normalize = True, skip_leading_nan = False) :
    """
    Load one or more columns from a wide CSV (e.g. ``sp_81_daily_prices.csv``)
    sitting in the ``datasets`` folder. The CSV is expected to have a ``Date``
    column (or any other first column that is not a numeric series) plus one
    column per series. Comma-separated by default.

    Parameters:
        stem             : CSV filename stem (no ``.csv``), resolved via ``dataset_path``.
        columns          : iterable of column names to load.
        normalize        : if True (default), each column is divided by its own max
            (same convention as ``ohlcv_close`` / ``mackey_glass``: ``x_norm = x / max(x)``).
        skip_leading_nan : if True, each column is trimmed to start at its first
            non-NaN row. This lets late-IPO tickers (e.g. ``DOW``, ``META``,
            ``GOOGL`` in ``sp100_daily_prices.csv``) be used without crashing.
            Returned arrays then have **different lengths** per column. Interior
            NaNs are still treated as an error regardless of this flag.

    Returns:
        dict mapping ``column_name`` -> ``{"x": raw 1D ndarray, "x_norm":
        normalized 1D ndarray (or copy of x when normalize=False), "x_max":
        per-column maximum used for scaling}``. Insertion order matches the
        ``columns`` argument.
    """
    csv_path = dataset_path(str(stem) + ".csv")
    if not os.path.exists(csv_path) :
        raise FileNotFoundError(f"Wide CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing :
        raise KeyError(
            f"Columns missing from {csv_path}: {missing}. "
            f"Available (excluding Date): {[c for c in df.columns if c != 'Date'][:10]}...")
    out = {}
    for c in cols :
        s = df[c]
        if skip_leading_nan :
            first = s.first_valid_index()
            if first is None :
                raise ValueError(
                    f"Column '{c}' in {csv_path} has no valid (non-NaN) values.")
            s = s.loc[first:]
        x_raw = np.asarray(s.values, dtype = float)
        if np.isnan(x_raw).any() :
            # Interior NaNs (or any NaNs when skip_leading_nan=False) remain a hard error.
            n_nan = int(np.isnan(x_raw).sum())
            hint = (" Use skip_leading_nan=True if these are pre-listing leading NaNs."
                    if not skip_leading_nan else
                    " These are interior NaNs (gap inside the series); fix the source data.")
            raise ValueError(
                f"Column '{c}' in {csv_path} contains {n_nan} NaNs.{hint}")
        if normalize :
            x_max = float(np.max(x_raw))
            if x_max <= 0.0 :
                raise ValueError(
                    f"Column '{c}' has non-positive max ({x_max}); cannot normalize.")
            x_norm = x_raw / x_max
        else :
            x_max = 1.0
            x_norm = x_raw.copy()
        out[c] = {"x" : x_raw, "x_norm" : x_norm, "x_max" : x_max}
    return out


def parse_phase_spec(spec_string) :
    """
    Parse a phase spec string of the form

        "<stem1>:<col1>,<col2>,...;<stem2>:<colA>,<colB>,..."

    into a list of ``(stem, [col, ...])`` tuples preserving order. Whitespace
    around stems/columns is tolerated. Empty groups raise ``ValueError``.

    Constraints (enforced upstream by ``Utils.findparamval`` which splits on
    whitespace): the entire spec value in the param file must not contain
    whitespace, so semicolons and commas are the only separators.
    """
    if not spec_string or not str(spec_string).strip() :
        raise ValueError("Empty phase spec")
    groups = []
    for raw_group in str(spec_string).split(";") :
        group = raw_group.strip()
        if not group :
            continue
        if ":" not in group :
            raise ValueError(
                f"Invalid phase spec group '{group}'. "
                f"Expected '<csv_stem>:<col1>,<col2>,...'.")
        stem, cols_part = group.split(":", 1)
        stem = stem.strip()
        cols = [c.strip() for c in cols_part.split(",") if c.strip()]
        if not stem or not cols :
            raise ValueError(
                f"Invalid phase spec group '{group}': "
                f"need a non-empty stem and at least one column.")
        groups.append((stem, cols))
    if not groups :
        raise ValueError(f"Phase spec '{spec_string}' parsed to no groups")
    return groups


def phase_to_concat_features(spec_string, cutoff = 0, skip_leading_nan = True) :
    """
    Resolve a phase spec to a single feature matrix by:
      1. parsing the spec into (stem, [columns]) groups,
      2. loading each requested column from its CSV (max-normalized),
      3. computing rolling mean/derivative features per column via
         ``make_mean_derivative_features``,
      4. concatenating row-wise in the order columns were listed.

    Hard concatenation only: no warmup buffer is inserted between segments
    (recurrent LTSL state will carry across boundaries; this is by design).

    Leading NaNs (e.g. late-IPO tickers in ``sp100_daily_prices.csv``) are
    trimmed per column by default so each column starts at its first valid
    sample. Interior NaNs still raise an error.

    Parameters:
        spec_string      : phase spec passed through ``parse_phase_spec``.
        cutoff           : forwarded to ``make_mean_derivative_features`` for
                           derivative clamping; default 0 matches ``Mains.general``.
        skip_leading_nan : forwarded to ``load_wide_csv_columns`` (default True
                           so late-IPO tickers contribute their valid range
                           rather than crashing).

    Returns:
        feats   : ndarray, concatenated [T_total, 4] feature matrix
        segments: list of ``(label, length)`` tuples, one per loaded column,
                  where ``label`` is ``"<stem>:<col>"`` and ``length`` is the
                  number of feature rows that column contributed.
    """
    groups = parse_phase_spec(spec_string)
    feats_parts = []
    segments = []
    for stem, cols in groups :
        loaded = load_wide_csv_columns(stem, cols, normalize = True,
                                       skip_leading_nan = skip_leading_nan)
        for col in cols :
            x = loaded[col]["x_norm"]
            seg_feats = make_mean_derivative_features(x = x, cutoff = cutoff, figno = 0)
            feats_parts.append(seg_feats)
            segments.append((f"{stem}:{col}", int(seg_feats.shape[0])))
    feats = np.concatenate(feats_parts, axis = 0)
    return feats, segments


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
