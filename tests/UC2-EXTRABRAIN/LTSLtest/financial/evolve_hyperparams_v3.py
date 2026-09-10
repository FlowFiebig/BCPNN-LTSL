#!/usr/bin/env python3
"""
Evolutionary search (v3) for K, K_mode, return_nbin, variance_nbin,
variance_window, binmode, and taumax to minimize ``rmse_validation`` from
:func:`Mains.multislice`. Runs until a wall-clock time budget (default 72 h).

Differences from ``evolve_hyperparams_v2.py`` (v2):

- Search space uses the ``return_variance`` feature-set parameter names
  (``return_nbin``, ``variance_nbin``, ``variance_window``, ``binmode``) instead
  of the legacy ``mean_nbin`` / ``deriv1st_nbin`` pair.
- Adds ``variance_window`` (trailing log-return variance window and calendar
  align) and ``binmode`` (``rbf`` vs ``rbf-quantile`` RBF centre layout).
- Default ``--base`` is ``Parameters_multislice_return_variance.par`` (expects
  ``feature_set return_variance``).

Inherited from v2:

- Drives ``Mains.multislice`` (multi-ticker walk-forward).
- Objective is fixed to rmse_validation (NaN-mean over the validation phase of
  the per-(ticker, phase, block) RMSE cube).
- ``timeseries_file`` / ``tickers`` / ``trnpat`` / ``vanpat`` / ``tenpat`` are
  read from the base .par and held fixed.

Run from the ``financial`` directory (or pass --chdir). Search bounds below
mirror the §7 1-D sweeps in ``notebook-multislice_return_variance.ipynb``
(``K`` 2–30, ``taumax`` 10–800, ``return_nbin`` 10–80, ``variance_nbin`` 2–16,
``variance_window`` 3–21, both ``K_mode`` / ``binmode`` choices). Example
72-hour run:

    python3 evolve_hyperparams_v3.py \\
        --time 259200 \\
        --base Parameters_multislice_return_variance.par \\
        --work-par Parameters_multislice_return_variance_ea_work.par \\
        --K 2 30 \\
        --K_mode linear logarithmic \\
        --return_nbin 10 80 \\
        --variance_nbin 2 16 \\
        --variance_window 3 21 \\
        --binmode rbf rbf-quantile \\
        --taumax 10 800 \\
        --pop 12 \\
        --history ea_return_variance_trace.jsonl \\
        --out-json ea_return_variance_best.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import FeatureGeneration
import Mains
import Utils


VALID_K_MODES = ("linear", "logarithmic")
VALID_BINMODES = FeatureGeneration.VALID_BINMODES
OBJECTIVE = "rmse_validation"  # fixed in v3


@dataclass
class Bounds:
    K: Tuple[int, int]
    K_mode: Sequence[str]
    return_nbin: Tuple[int, int]
    variance_nbin: Tuple[int, int]
    variance_window: Tuple[int, int]
    binmode: Sequence[str]
    taumax: Tuple[float, float]

    def clip(self, genes: "Genes") -> "Genes":
        K0, K1 = self.K
        r0, r1 = self.return_nbin
        v0, v1 = self.variance_nbin
        w0, w1 = self.variance_window
        t0, t1 = self.taumax
        km = genes.K_mode if genes.K_mode in self.K_mode else self.K_mode[0]
        bm = genes.binmode if genes.binmode in self.binmode else self.binmode[0]
        return Genes(
            K=int(np.clip(genes.K, K0, K1)),
            K_mode=km,
            return_nbin=int(np.clip(genes.return_nbin, r0, r1)),
            variance_nbin=int(np.clip(genes.variance_nbin, v0, v1)),
            variance_window=int(np.clip(genes.variance_window, w0, w1)),
            binmode=bm,
            taumax=float(np.clip(genes.taumax, t0, t1)),
        )


@dataclass
class Genes:
    K: int
    K_mode: str
    return_nbin: int
    variance_nbin: int
    variance_window: int
    binmode: str
    taumax: float
    fitness: Optional[float] = None  # rmse_validation (lower is better)
    rmse_train: Optional[float] = None
    rmse_validation: Optional[float] = None
    rmse_unseen: Optional[float] = None
    rmse_persistence: Optional[float] = None

    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.K,
            self.K_mode,
            self.return_nbin,
            self.variance_nbin,
            self.variance_window,
            self.binmode,
            round(float(self.taumax), 8),
        )


def random_genes(rng: np.random.Generator, b: Bounds) -> Genes:
    K0, K1 = b.K
    r0, r1 = b.return_nbin
    v0, v1 = b.variance_nbin
    w0, w1 = b.variance_window
    t0, t1 = b.taumax
    return b.clip(
        Genes(
            K=int(rng.integers(K0, K1 + 1)),
            K_mode=rng.choice(b.K_mode),
            return_nbin=int(rng.integers(r0, r1 + 1)),
            variance_nbin=int(rng.integers(v0, v1 + 1)),
            variance_window=int(rng.integers(w0, w1 + 1)),
            binmode=rng.choice(b.binmode),
            taumax=float(rng.uniform(t0, t1)),
        )
    )


def crossover(rng: np.random.Generator, a: Genes, b: Genes, bounds: Bounds) -> Genes:
    g = Genes(
        K=a.K if rng.random() < 0.5 else b.K,
        K_mode=a.K_mode if rng.random() < 0.5 else b.K_mode,
        return_nbin=a.return_nbin if rng.random() < 0.5 else b.return_nbin,
        variance_nbin=a.variance_nbin if rng.random() < 0.5 else b.variance_nbin,
        variance_window=a.variance_window if rng.random() < 0.5 else b.variance_window,
        binmode=a.binmode if rng.random() < 0.5 else b.binmode,
        taumax=a.taumax if rng.random() < 0.5 else b.taumax,
    )
    return bounds.clip(g)


def mutate(rng: np.random.Generator, g: Genes, bounds: Bounds, p: float = 0.25) -> Genes:
    K0, K1 = bounds.K
    r0, r1 = bounds.return_nbin
    v0, v1 = bounds.variance_nbin
    w0, w1 = bounds.variance_window
    t0, t1 = bounds.taumax

    K, km = g.K, g.K_mode
    rn, vn, vw, bm, t = g.return_nbin, g.variance_nbin, g.variance_window, g.binmode, g.taumax
    if rng.random() < p:
        step = int(rng.integers(-2, 3))
        K = int(np.clip(K + step, K0, K1))
    if rng.random() < p and len(bounds.K_mode) > 1:
        km = rng.choice(bounds.K_mode)
    if rng.random() < p:
        step = int(rng.integers(-5, 6))
        rn = int(np.clip(rn + step, r0, r1))
    if rng.random() < p:
        step = int(rng.integers(-2, 3))
        vn = int(np.clip(vn + step, v0, v1))
    if rng.random() < p:
        step = int(rng.integers(-3, 4))
        vw = int(np.clip(vw + step, w0, w1))
    if rng.random() < p and len(bounds.binmode) > 1:
        bm = rng.choice(bounds.binmode)
    if rng.random() < p:
        span = t1 - t0
        t = float(np.clip(t + rng.normal(0, 0.15 * max(span, 1e-9)), t0, t1))

    return bounds.clip(
        Genes(
            K=K,
            K_mode=km,
            return_nbin=rn,
            variance_nbin=vn,
            variance_window=vw,
            binmode=bm,
            taumax=t,
        )
    )


def tournament(rng: np.random.Generator, pop: List[Genes], k: int = 3) -> Genes:
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    contestants = [pop[i] for i in idx]
    return min(contestants, key=lambda ind: ind.fitness if ind.fitness is not None else np.inf)


def apply_genes(base_params: Dict[str, Any], g: Genes) -> Dict[str, Any]:
    """Return a copy of base_params with the 7 EA-controlled keys overridden."""
    p = dict(base_params)
    p["K"] = g.K
    p["K_mode"] = g.K_mode
    p["return_nbin"] = g.return_nbin
    p["variance_nbin"] = g.variance_nbin
    p["variance_window"] = g.variance_window
    p["binmode"] = g.binmode
    p["taumax"] = g.taumax
    return p


def _nanmean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    m = np.isfinite(arr)
    return float(np.mean(arr[m])) if m.any() else float("nan")


def _nanmean_cube(arr: np.ndarray) -> float:
    """NaN-mean over an arbitrary-shaped slice of a multislice RMSE cube."""
    flat = np.asarray(arr, dtype=float).ravel()
    m = np.isfinite(flat)
    return float(np.mean(flat[m])) if m.any() else float("nan")


def evaluate(
    g: Genes,
    base_params: Dict[str, Any],
    work_par: str,
    exefile: Optional[str],
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    verbose_eval: bool,
    n_repeats: int = 1,
) -> float:
    key = g.cache_key()
    if key in cache:
        entry = cache[key]
        g.fitness = entry["fitness"]
        g.rmse_train = entry["rmse_train"]
        g.rmse_validation = entry["rmse_validation"]
        g.rmse_unseen = entry["rmse_unseen"]
        g.rmse_persistence = entry["rmse_persistence"]
        return g.fitness

    Utils.write_params(work_par, apply_genes(base_params, g), preserve_comments=False)
    tr_samples: List[float] = []
    val_samples: List[float] = []
    u_samples: List[float] = []
    pers_samples: List[float] = []
    for _ in range(max(1, int(n_repeats))):
        try:
            result = Mains.multislice(paramfile=work_par, exefile=exefile, verbose=verbose_eval)
        except Exception as e:
            if verbose_eval:
                print(f"  eval failed: {e}", file=sys.stderr)
            tr_samples.append(float("nan"))
            val_samples.append(float("nan"))
            u_samples.append(float("nan"))
            pers_samples.append(float("nan"))
            continue
        rmse_cube = np.asarray(result["rmse"], dtype=float)
        pers_cube = np.asarray(result["persistence"], dtype=float)
        tr_samples.append(_nanmean_cube(rmse_cube[:, 0, :]))
        val_samples.append(_nanmean_cube(rmse_cube[:, 1, :]))
        u_samples.append(_nanmean_cube(rmse_cube[:, 2, :]))
        pers_samples.append(_nanmean_cube(pers_cube[:, 1, :]))

    g.rmse_train = _nanmean(tr_samples)
    g.rmse_validation = _nanmean(val_samples)
    g.rmse_unseen = _nanmean(u_samples)
    g.rmse_persistence = _nanmean(pers_samples)

    fit = g.rmse_validation if np.isfinite(g.rmse_validation) else float("inf")
    g.fitness = fit
    cache[key] = {
        "fitness": fit,
        "rmse_train": g.rmse_train,
        "rmse_validation": g.rmse_validation,
        "rmse_unseen": g.rmse_unseen,
        "rmse_persistence": g.rmse_persistence,
    }
    return fit


def _clone(g: Genes) -> Genes:
    return Genes(
        K=g.K,
        K_mode=g.K_mode,
        return_nbin=g.return_nbin,
        variance_nbin=g.variance_nbin,
        variance_window=g.variance_window,
        binmode=g.binmode,
        taumax=g.taumax,
        fitness=g.fitness,
        rmse_train=g.rmse_train,
        rmse_validation=g.rmse_validation,
        rmse_unseen=g.rmse_unseen,
        rmse_persistence=g.rmse_persistence,
    )


def _genes_to_history_row(
    ind: Genes,
    *,
    t: float,
    generation: int,
    best_fitness: Optional[float],
) -> Dict[str, Any]:
    return {
        "t": t,
        "gen": generation,
        "objective": OBJECTIVE,
        "K": ind.K,
        "K_mode": ind.K_mode,
        "return_nbin": ind.return_nbin,
        "variance_nbin": ind.variance_nbin,
        "variance_window": ind.variance_window,
        "binmode": ind.binmode,
        "taumax": ind.taumax,
        "rmse_train": ind.rmse_train,
        "rmse_validation": ind.rmse_validation,
        "rmse_unseen": ind.rmse_unseen,
        "rmse_persistence": ind.rmse_persistence,
        "fitness": ind.fitness,
        "best_fitness": best_fitness,
    }


def run_ea(
    time_budget_s: float,
    bounds: Bounds,
    base_param_path: str,
    work_par: str,
    exefile: Optional[str],
    pop_size: int,
    rng_seed: Optional[int],
    verbose_eval: bool,
    log_every: int,
    n_repeats: int,
    history_path: Optional[str] = None,
) -> Tuple[Genes, List[Dict[str, Any]]]:
    rng = np.random.default_rng(rng_seed)
    base_params = Utils.load_params(base_param_path)
    if "timeseries_file" not in base_params:
        raise KeyError(
            f"Base param file '{base_param_path}' is missing required key "
            f"'timeseries_file'. evolve_hyperparams_v3.py drives "
            f"Mains.multislice(), which requires timeseries_file (wide CSV); "
            f"tickers is optional (comma-separated column list, or omit for "
            f"all columns). Use a Parameters_multislice-style file.")
    feature_set = str(base_params.get("feature_set", "mean_derivative"))
    if feature_set != "return_variance":
        raise ValueError(
            f"evolve_hyperparams_v3.py expects feature_set return_variance in "
            f"'{base_param_path}' (got '{feature_set}'). Use v2 for "
            f"mean_derivative / mean_nbin searches.")
    w0, _ = bounds.variance_window
    if w0 < 2:
        raise ValueError("variance_window lower bound must be >= 2.")

    cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []

    history_file = open(history_path, "w", encoding="utf-8") if history_path else None
    try:
        t_start = time.perf_counter()
        n_eval = 0

        population: List[Genes] = [random_genes(rng, bounds) for _ in range(pop_size)]
        best_ever: Optional[Genes] = None

        def tick_log(force: bool = False) -> None:
            if not force and log_every > 0 and n_eval % log_every != 0:
                return
            elapsed = time.perf_counter() - t_start
            bf = best_ever.fitness if best_ever and best_ever.fitness is not None else float("nan")
            print(
                f"[{elapsed:7.1f}s / {time_budget_s:.0f}s] evals={n_eval} "
                f"best_{OBJECTIVE}={bf:.6g}",
                flush=True,
            )

        generation = 0
        while time.perf_counter() - t_start < time_budget_s:
            for ind in population:
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                if ind.fitness is not None:
                    continue
                evaluate(ind, base_params, work_par, exefile, cache, verbose_eval, n_repeats=n_repeats)
                n_eval += 1
                if best_ever is None or (ind.fitness is not None and ind.fitness < best_ever.fitness):
                    best_ever = _clone(ind)
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                row = _genes_to_history_row(
                    ind,
                    t=time.perf_counter() - t_start,
                    generation=generation,
                    best_fitness=best_ever.fitness,
                )
                history.append(row)
                if history_file is not None:
                    history_file.write(json.dumps(row) + "\n")
                    history_file.flush()
                tick_log()

            if time.perf_counter() - t_start >= time_budget_s:
                break

            population.sort(key=lambda x: x.fitness if x.fitness is not None else np.inf)
            elites = [_clone(p) for p in population[:2]]

            new_pop: List[Genes] = list(elites)
            while len(new_pop) < pop_size:
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                pa = tournament(rng, population)
                pb = tournament(rng, population)
                child = mutate(rng, crossover(rng, pa, pb, bounds), bounds)
                child.fitness = None
                new_pop.append(child)
            population = new_pop
            generation += 1

        tick_log(force=True)
        if best_ever is None:
            raise RuntimeError("No successful evaluations; check ltslmain and parameter files.")
        return best_ever, history
    finally:
        if history_file is not None:
            history_file.close()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="EA minimizing rmse_validation from Mains.multislice() over "
                    "K, K_mode, return_nbin, variance_nbin, variance_window, "
                    "binmode, taumax (return_variance feature set). "
                    "timeseries_file/tickers are read from --base and held fixed.")
    parser.add_argument("--chdir", default=here, help="Working directory (default: script dir)")
    parser.add_argument(
        "--base", default="Parameters_multislice_return_variance.par",
        help="Base .par template (must contain timeseries_file and "
             "feature_set return_variance); "
             "default Parameters_multislice_return_variance.par",
    )
    parser.add_argument(
        "--work-par", default="Parameters_multislice_return_variance_ea_work.par",
        help="Scratch .par path written for every evaluation",
    )
    parser.add_argument(
        "--time", type=float, default=259200.0,
        help="Time budget in seconds (default 259200 = 72 h)",
    )
    parser.add_argument("--pop", type=int, default=12, help="Population size")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exefile", default="./ltslmain", help='Executable (use "" to skip sim)')
    parser.add_argument("--verbose-eval", action="store_true", help="Forward verbose Mains.multislice")
    parser.add_argument("--log-every", type=int, default=1, help="Print progress every N evals (0=only final)")
    parser.add_argument(
        "--n-repeats", type=int, default=1,
        help="Independent simulator runs per genotype; per-metric RMSEs are "
             "averaged across runs to reduce noise (default 1).",
    )
    parser.add_argument(
        "--history", default=None,
        help="Optional JSONL path for search trace (truncated at start; one line appended per evaluation)",
    )
    parser.add_argument("--out-json", default=None, help="Optional JSON path for best solution")
    parser.add_argument(
        "--K", nargs=2, type=int, metavar=("MIN", "MAX"), default=[2, 30],
        help="Search bounds for K (notebook §7 sweep: 2,4,8,12,20,30)",
    )
    parser.add_argument(
        "--K_mode", nargs="+", choices=VALID_K_MODES,
        default=list(VALID_K_MODES),
        help="K_mode values to explore (default: both linear and logarithmic)",
    )
    parser.add_argument(
        "--return_nbin", nargs=2, type=int, metavar=("MIN", "MAX"), default=[10, 80],
        help="Search bounds for decode-target (log return) bins "
             "(notebook §7 sweep: 10,20,40,80)",
    )
    parser.add_argument(
        "--variance_nbin", nargs=2, type=int, metavar=("MIN", "MAX"), default=[2, 16],
        help="Search bounds for secondary (trailing return variance) bins "
             "(notebook §7 sweep: 2,4,8,16)",
    )
    parser.add_argument(
        "--variance_window", nargs=2, type=int, metavar=("MIN", "MAX"), default=[3, 21],
        help="Search bounds for trailing variance window, minimum 2 "
             "(notebook §7 sweep: 3,5,7,14,21)",
    )
    parser.add_argument(
        "--binmode", nargs="+", choices=VALID_BINMODES,
        default=list(VALID_BINMODES),
        help="binmode values to explore (default: rbf and rbf-quantile)",
    )
    parser.add_argument(
        "--taumax", nargs=2, type=float, metavar=("MIN", "MAX"), default=[10.0, 800.0],
        help="Search bounds for taumax (slowest LTSL time constant; "
             "notebook §7 sweep: 10,25,50,100,200,400,800)",
    )

    args = parser.parse_args()
    os.chdir(args.chdir)

    vw_min, vw_max = min(args.variance_window), max(args.variance_window)
    if vw_min < 2:
        parser.error("variance_window lower bound must be >= 2.")

    b = Bounds(
        K=(min(args.K), max(args.K)),
        K_mode=list(dict.fromkeys(args.K_mode)),
        return_nbin=(min(args.return_nbin), max(args.return_nbin)),
        variance_nbin=(min(args.variance_nbin), max(args.variance_nbin)),
        variance_window=(vw_min, vw_max),
        binmode=list(dict.fromkeys(args.binmode)),
        taumax=(min(args.taumax), max(args.taumax)),
    )

    exefile = args.exefile.strip()
    exefile_arg: Optional[str] = exefile if exefile else None

    best, _ = run_ea(
        time_budget_s=args.time,
        bounds=b,
        base_param_path=args.base,
        work_par=args.work_par,
        exefile=exefile_arg,
        pop_size=args.pop,
        rng_seed=args.seed,
        verbose_eval=args.verbose_eval,
        log_every=args.log_every,
        n_repeats=args.n_repeats,
        history_path=args.history,
    )

    print("\n=== Best after time budget ===")
    print(f"  objective         = {OBJECTIVE} (fixed in v3)")
    print(f"  K                 = {best.K}")
    print(f"  K_mode            = {best.K_mode}")
    print(f"  return_nbin       = {best.return_nbin}")
    print(f"  variance_nbin     = {best.variance_nbin}")
    print(f"  variance_window   = {best.variance_window}")
    print(f"  binmode           = {best.binmode}")
    print(f"  taumax            = {best.taumax}")
    print(f"  {OBJECTIVE} (minimized) = {best.fitness}")
    if best.rmse_train is not None:
        print(f"  rmse_train         = {best.rmse_train}")
    if best.rmse_validation is not None:
        print(f"  rmse_validation    = {best.rmse_validation}")
    if best.rmse_unseen is not None:
        print(f"  rmse_unseen        = {best.rmse_unseen}")
    if best.rmse_persistence is not None:
        print(f"  rmse_persistence   = {best.rmse_persistence}  "
              f"(naive y_t -> y_{{t+offs}}=y_t baseline; "
              f"any model worth its salt should beat this)")

    if args.out_json:
        out = {
            "objective": OBJECTIVE,
            "fitness": best.fitness,
            "K": best.K,
            "K_mode": best.K_mode,
            "return_nbin": best.return_nbin,
            "variance_nbin": best.variance_nbin,
            "variance_window": best.variance_window,
            "binmode": best.binmode,
            "taumax": best.taumax,
            "rmse_train": best.rmse_train,
            "rmse_validation": best.rmse_validation,
            "rmse_unseen": best.rmse_unseen,
            "rmse_persistence": best.rmse_persistence,
            "base_paramfile": args.base,
            "time_budget_s": args.time,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
