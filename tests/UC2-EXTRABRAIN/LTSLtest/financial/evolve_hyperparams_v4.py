#!/usr/bin/env python3
"""
Evolutionary search (v4) for K, K_mode, return_nbin, variance_nbin,
variance_window, binmode, taumin, taumax, and bdebias to minimize a selectable fitness
criterion from :func:`Mains.multislice` (default: ``tick_loss``). Runs until
a wall-clock time budget (default 72 h).

Differences from ``evolve_hyperparams_v3.py`` (v3):

- Default ``--base`` is ``Parameters_FOREX.par``; all other keys are read from
  that file and held fixed.
- Fitness criterion is selectable via ``--objective`` (default ``tick_loss``).
  Also supports ``quadratic_loss``, ``smooth_loss``, ``firm_loss``, and the
  scalar RMSE aggregates ``rmse_train``, ``rmse_validation``, ``rmse_unseen``,
  ``rmse_persistence``.
- Every evaluation requests RMSE cubes plus all four DeepVaR losses; the trace
  records every metric so objectives can be correlated offline.
- ``--var-level`` sets the VaR confidence level passed to
  :func:`Mains.multislice` (default 0.99, matching ``nb-FOREX.ipynb``).
- ``n_repeats`` removed (deterministic simulator; genotype cache avoids reruns).

Inherited from v3:

- Search space: K, K_mode, return_nbin, variance_nbin, variance_window,
  binmode, taumin, taumax, bdebias (``return_variance`` feature set).
  ``taumax`` is always strictly greater than ``taumin``.
- Drives ``Mains.multislice`` (multi-ticker walk-forward).

Run from the ``financial`` directory (or pass --chdir). Example 72-hour run
optimizing tick loss:

    python3 evolve_hyperparams_v4.py \
        --time 259200 \
        --objective tick_loss \
        --var-level 0.99 \
        --base Parameters_FOREX.par \
        --work-par Parameters_FOREX_ea_work.par \
        --K 2 30 \
        --K_mode linear logarithmic \
        --return_nbin 10 80 \
        --variance_nbin 2 16 \
        --variance_window 3 21 \
        --binmode rbf rbf-quantile \
        --taumin 0.1 100 \
        --taumax 10 800 \
        --bdebias 0 1 \
        --pop 12 \
        --history ea_forex_trace.jsonl \
        --out-json ea_forex_best.json
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

import DeepVaRLosses
import FeatureGeneration
import Mains
import Utils


VALID_K_MODES = ("linear", "logarithmic")
VALID_BINMODES = FeatureGeneration.VALID_BINMODES
VALID_BDEBIAS = (0, 1)
TAU_STRICT_GAP = 1e-6

DEEPVAR_OBJECTIVES = tuple(DeepVaRLosses.DEEPVAR_LOSS_NAMES)
RMSE_OBJECTIVES = (
    "rmse_train",
    "rmse_validation",
    "rmse_unseen",
    "rmse_persistence",
)
VALID_OBJECTIVES = DEEPVAR_OBJECTIVES + RMSE_OBJECTIVES
DEFAULT_OBJECTIVE = "tick_loss"

EVAL_METRICS = (
    "rmse",
    "persistence",
    "quadratic_loss",
    "smooth_loss",
    "tick_loss",
    "firm_loss",
)


@dataclass
class Bounds:
    K: Tuple[int, int]
    K_mode: Sequence[str]
    return_nbin: Tuple[int, int]
    variance_nbin: Tuple[int, int]
    variance_window: Tuple[int, int]
    binmode: Sequence[str]
    taumin: Tuple[float, float]
    taumax: Tuple[float, float]
    bdebias: Sequence[int]

    def _clip_tau_pair(self, taumin: float, taumax: float) -> Tuple[float, float]:
        """Clip taumin/taumax to bounds and enforce taumax > taumin."""
        tmin0, tmin1 = self.taumin
        tmax0, tmax1 = self.taumax
        gap = TAU_STRICT_GAP
        tmin = float(np.clip(taumin, tmin0, tmin1))
        tmax_lo = max(tmax0, tmin + gap)
        if tmax_lo > tmax1:
            tmin = float(np.clip(tmax1 - gap, tmin0, tmin1))
            tmax_lo = max(tmax0, tmin + gap)
        tmax = float(np.clip(taumax, tmax_lo, tmax1))
        if tmax <= tmin:
            tmax = min(tmax1, tmin + gap)
        if tmax <= tmin:
            tmin = max(tmin0, tmax - gap)
        return tmin, tmax

    def clip(self, genes: "Genes") -> "Genes":
        K0, K1 = self.K
        r0, r1 = self.return_nbin
        v0, v1 = self.variance_nbin
        w0, w1 = self.variance_window
        km = genes.K_mode if genes.K_mode in self.K_mode else self.K_mode[0]
        bm = genes.binmode if genes.binmode in self.binmode else self.binmode[0]
        bd = int(genes.bdebias) if int(genes.bdebias) in self.bdebias else int(self.bdebias[0])
        tmin, tmax = self._clip_tau_pair(genes.taumin, genes.taumax)
        return Genes(
            K=int(np.clip(genes.K, K0, K1)),
            K_mode=km,
            return_nbin=int(np.clip(genes.return_nbin, r0, r1)),
            variance_nbin=int(np.clip(genes.variance_nbin, v0, v1)),
            variance_window=int(np.clip(genes.variance_window, w0, w1)),
            binmode=bm,
            taumin=tmin,
            taumax=tmax,
            bdebias=bd,
        )


@dataclass
class Genes:
    K: int
    K_mode: str
    return_nbin: int
    variance_nbin: int
    variance_window: int
    binmode: str
    taumin: float
    taumax: float
    bdebias: int
    fitness: Optional[float] = None
    rmse_train: Optional[float] = None
    rmse_validation: Optional[float] = None
    rmse_unseen: Optional[float] = None
    rmse_persistence: Optional[float] = None
    quadratic_loss: Optional[float] = None
    smooth_loss: Optional[float] = None
    tick_loss: Optional[float] = None
    firm_loss: Optional[float] = None
    violation_rate: Optional[float] = None
    n_violations: Optional[float] = None

    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.K,
            self.K_mode,
            self.return_nbin,
            self.variance_nbin,
            self.variance_window,
            self.binmode,
            round(float(self.taumin), 8),
            round(float(self.taumax), 8),
            int(self.bdebias),
        )


def random_genes(rng: np.random.Generator, b: Bounds) -> Genes:
    K0, K1 = b.K
    r0, r1 = b.return_nbin
    v0, v1 = b.variance_nbin
    w0, w1 = b.variance_window
    tmin0, tmin1 = b.taumin
    tmax0, tmax1 = b.taumax
    taumin = float(rng.uniform(tmin0, tmin1))
    taumax_lo = max(tmax0, taumin + TAU_STRICT_GAP)
    taumax = float(rng.uniform(taumax_lo, tmax1))
    return b.clip(
        Genes(
            K=int(rng.integers(K0, K1 + 1)),
            K_mode=rng.choice(b.K_mode),
            return_nbin=int(rng.integers(r0, r1 + 1)),
            variance_nbin=int(rng.integers(v0, v1 + 1)),
            variance_window=int(rng.integers(w0, w1 + 1)),
            binmode=rng.choice(b.binmode),
            taumin=taumin,
            taumax=taumax,
            bdebias=int(rng.choice(b.bdebias)),
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
        taumin=a.taumin if rng.random() < 0.5 else b.taumin,
        taumax=a.taumax if rng.random() < 0.5 else b.taumax,
        bdebias=a.bdebias if rng.random() < 0.5 else b.bdebias,
    )
    return bounds.clip(g)


def mutate(rng: np.random.Generator, g: Genes, bounds: Bounds, p: float = 0.25) -> Genes:
    K0, K1 = bounds.K
    r0, r1 = bounds.return_nbin
    v0, v1 = bounds.variance_nbin
    w0, w1 = bounds.variance_window
    tmin0, tmin1 = bounds.taumin
    tmax0, tmax1 = bounds.taumax

    K, km = g.K, g.K_mode
    rn, vn, vw, bm = g.return_nbin, g.variance_nbin, g.variance_window, g.binmode
    tmin, tmax, bd = g.taumin, g.taumax, g.bdebias
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
        span = tmin1 - tmin0
        tmin = float(np.clip(tmin + rng.normal(0, 0.15 * max(span, 1e-9)), tmin0, tmin1))
    if rng.random() < p:
        span = tmax1 - tmax0
        tmax = float(np.clip(tmax + rng.normal(0, 0.15 * max(span, 1e-9)), tmax0, tmax1))
    if rng.random() < p and len(bounds.bdebias) > 1:
        bd = int(rng.choice(bounds.bdebias))

    return bounds.clip(
        Genes(
            K=K,
            K_mode=km,
            return_nbin=rn,
            variance_nbin=vn,
            variance_window=vw,
            binmode=bm,
            taumin=tmin,
            taumax=tmax,
            bdebias=bd,
        )
    )


def tournament(rng: np.random.Generator, pop: List[Genes], k: int = 3) -> Genes:
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    contestants = [pop[i] for i in idx]
    return min(contestants, key=lambda ind: ind.fitness if ind.fitness is not None else np.inf)


def apply_genes(base_params: Dict[str, Any], g: Genes) -> Dict[str, Any]:
    """Return a copy of base_params with the 9 EA-controlled keys overridden."""
    p = dict(base_params)
    p["K"] = g.K
    p["K_mode"] = g.K_mode
    p["return_nbin"] = g.return_nbin
    p["variance_nbin"] = g.variance_nbin
    p["variance_window"] = g.variance_window
    p["binmode"] = g.binmode
    p["taumin"] = g.taumin
    p["taumax"] = g.taumax
    p["bdebias"] = g.bdebias
    return p


def _nanmean_cube(arr: np.ndarray) -> float:
    """NaN-mean over an arbitrary-shaped slice of a multislice RMSE cube."""
    flat = np.asarray(arr, dtype=float).ravel()
    m = np.isfinite(flat)
    return float(np.mean(flat[m])) if m.any() else float("nan")


def _nanmean_vector(arr: np.ndarray) -> float:
    """NaN-mean over a per-ticker metric vector."""
    return _nanmean_cube(arr)


def _populate_metrics(g: Genes, result: Dict[str, Any]) -> None:
    rmse_cube = np.asarray(result["rmse"], dtype=float)
    pers_cube = np.asarray(result["persistence"], dtype=float)
    g.rmse_train = _nanmean_cube(rmse_cube[:, 0, :])
    g.rmse_validation = _nanmean_cube(rmse_cube[:, 1, :])
    g.rmse_unseen = _nanmean_cube(rmse_cube[:, 2, :])
    g.rmse_persistence = _nanmean_cube(pers_cube[:, 1, :])

    for name in DEEPVAR_OBJECTIVES:
        setattr(g, name, _nanmean_vector(np.asarray(result[name], dtype=float)))
    g.n_violations = _nanmean_vector(np.asarray(result["n_violations"], dtype=float))
    g.violation_rate = _nanmean_vector(np.asarray(result["violation_rate"], dtype=float))


def _metrics_cache_entry(g: Genes, fitness: float) -> Dict[str, Any]:
    return {
        "fitness": fitness,
        "rmse_train": g.rmse_train,
        "rmse_validation": g.rmse_validation,
        "rmse_unseen": g.rmse_unseen,
        "rmse_persistence": g.rmse_persistence,
        "quadratic_loss": g.quadratic_loss,
        "smooth_loss": g.smooth_loss,
        "tick_loss": g.tick_loss,
        "firm_loss": g.firm_loss,
        "n_violations": g.n_violations,
        "violation_rate": g.violation_rate,
    }


def _apply_cache_entry(g: Genes, entry: Dict[str, Any]) -> float:
    g.fitness = entry["fitness"]
    g.rmse_train = entry["rmse_train"]
    g.rmse_validation = entry["rmse_validation"]
    g.rmse_unseen = entry["rmse_unseen"]
    g.rmse_persistence = entry["rmse_persistence"]
    g.quadratic_loss = entry["quadratic_loss"]
    g.smooth_loss = entry["smooth_loss"]
    g.tick_loss = entry["tick_loss"]
    g.firm_loss = entry["firm_loss"]
    g.n_violations = entry["n_violations"]
    g.violation_rate = entry["violation_rate"]
    return g.fitness


def evaluate(
    g: Genes,
    base_params: Dict[str, Any],
    work_par: str,
    exefile: Optional[str],
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    objective: str,
    var_level: float,
    verbose_eval: bool,
) -> float:
    key = g.cache_key()
    if key in cache:
        return _apply_cache_entry(g, cache[key])

    Utils.write_params(work_par, apply_genes(base_params, g), preserve_comments=False)
    try:
        result = Mains.multislice(
            paramfile=work_par,
            exefile=exefile,
            verbose=verbose_eval,
            metrics=EVAL_METRICS,
            var_level=var_level,
        )
    except Exception as e:
        if verbose_eval:
            print(f"  eval failed: {e}", file=sys.stderr)
        g.fitness = float("inf")
        cache[key] = _metrics_cache_entry(g, g.fitness)
        return g.fitness

    _populate_metrics(g, result)
    metric = getattr(g, objective) if objective in DEEPVAR_OBJECTIVES else {
        "rmse_train": g.rmse_train,
        "rmse_validation": g.rmse_validation,
        "rmse_unseen": g.rmse_unseen,
        "rmse_persistence": g.rmse_persistence,
    }[objective]
    fit = float(metric) if np.isfinite(metric) else float("inf")
    g.fitness = fit
    cache[key] = _metrics_cache_entry(g, fit)
    return fit


def _clone(g: Genes) -> Genes:
    return Genes(
        K=g.K,
        K_mode=g.K_mode,
        return_nbin=g.return_nbin,
        variance_nbin=g.variance_nbin,
        variance_window=g.variance_window,
        binmode=g.binmode,
        taumin=g.taumin,
        taumax=g.taumax,
        bdebias=g.bdebias,
        fitness=g.fitness,
        rmse_train=g.rmse_train,
        rmse_validation=g.rmse_validation,
        rmse_unseen=g.rmse_unseen,
        rmse_persistence=g.rmse_persistence,
        quadratic_loss=g.quadratic_loss,
        smooth_loss=g.smooth_loss,
        tick_loss=g.tick_loss,
        firm_loss=g.firm_loss,
        violation_rate=g.violation_rate,
        n_violations=g.n_violations,
    )


def _genes_to_history_row(
    ind: Genes,
    *,
    t: float,
    generation: int,
    objective: str,
    var_level: float,
    best_fitness: Optional[float],
) -> Dict[str, Any]:
    return {
        "t": t,
        "gen": generation,
        "objective": objective,
        "var_level": var_level,
        "K": ind.K,
        "K_mode": ind.K_mode,
        "return_nbin": ind.return_nbin,
        "variance_nbin": ind.variance_nbin,
        "variance_window": ind.variance_window,
        "binmode": ind.binmode,
        "taumin": ind.taumin,
        "taumax": ind.taumax,
        "bdebias": ind.bdebias,
        "rmse_train": ind.rmse_train,
        "rmse_validation": ind.rmse_validation,
        "rmse_unseen": ind.rmse_unseen,
        "rmse_persistence": ind.rmse_persistence,
        "quadratic_loss": ind.quadratic_loss,
        "smooth_loss": ind.smooth_loss,
        "tick_loss": ind.tick_loss,
        "firm_loss": ind.firm_loss,
        "n_violations": ind.n_violations,
        "violation_rate": ind.violation_rate,
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
    objective: str,
    var_level: float,
    verbose_eval: bool,
    log_every: int,
    history_path: Optional[str] = None,
) -> Tuple[Genes, List[Dict[str, Any]]]:
    rng = np.random.default_rng(rng_seed)
    base_params = Utils.load_params(base_param_path)
    if "timeseries_file" not in base_params:
        raise KeyError(
            f"Base param file '{base_param_path}' is missing required key "
            f"'timeseries_file'. evolve_hyperparams_v4.py drives "
            f"Mains.multislice(), which requires timeseries_file (wide CSV); "
            f"tickers is optional (comma-separated column list, or omit for "
            f"all columns).")
    feature_set = str(base_params.get("feature_set", "mean_derivative"))
    if feature_set != "return_variance":
        raise ValueError(
            f"evolve_hyperparams_v4.py expects feature_set return_variance in "
            f"'{base_param_path}' (got '{feature_set}').")
    w0, _ = bounds.variance_window
    if w0 < 2:
        raise ValueError("variance_window lower bound must be >= 2.")
    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective}'; choose from {VALID_OBJECTIVES}")
    if not 0.0 < var_level < 1.0:
        raise ValueError(f"var_level must lie strictly in (0, 1); got {var_level}")
    tmin0, tmin1 = bounds.taumin
    tmax0, tmax1 = bounds.taumax
    if tmin0 >= tmin1 or tmax0 >= tmax1:
        raise ValueError("taumin and taumax bounds must each have MIN < MAX.")
    if max(tmax0, tmin0 + TAU_STRICT_GAP) > tmax1:
        raise ValueError(
            "Infeasible taumin/taumax bounds: no value pair with taumax > taumin "
            f"exists within [{tmin0}, {tmin1}] x [{tmax0}, {tmax1}].")

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
                f"best_{objective}={bf:.6g}",
                flush=True,
            )

        generation = 0
        while time.perf_counter() - t_start < time_budget_s:
            for ind in population:
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                if ind.fitness is not None:
                    continue
                evaluate(
                    ind, base_params, work_par, exefile, cache,
                    objective, var_level, verbose_eval,
                )
                n_eval += 1
                if best_ever is None or (ind.fitness is not None and ind.fitness < best_ever.fitness):
                    best_ever = _clone(ind)
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                row = _genes_to_history_row(
                    ind,
                    t=time.perf_counter() - t_start,
                    generation=generation,
                    objective=objective,
                    var_level=var_level,
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
        description="EA minimizing a selectable multislice metric (default "
                    "tick_loss) over K, K_mode, return_nbin, variance_nbin, "
                    "variance_window, binmode, taumin, taumax, bdebias "
                    "(return_variance feature set; taumax > taumin enforced). "
                    "Other keys are read from --base and held fixed.")
    parser.add_argument("--chdir", default=here, help="Working directory (default: script dir)")
    parser.add_argument(
        "--base", default="Parameters_FOREX.par",
        help="Base .par template (must contain timeseries_file and "
             "feature_set return_variance); default Parameters_FOREX.par",
    )
    parser.add_argument(
        "--work-par", default="Parameters_FOREX_ea_work.par",
        help="Scratch .par path written for every evaluation",
    )
    parser.add_argument(
        "--objective", default=DEFAULT_OBJECTIVE, choices=VALID_OBJECTIVES,
        help=f"Fitness criterion to minimize (default: {DEFAULT_OBJECTIVE})",
    )
    parser.add_argument(
        "--var-level", type=float, default=0.99,
        help="VaR confidence level for DeepVaR losses (default 0.99)",
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
        "--history", default=None,
        help="Optional JSONL path for search trace (truncated at start; one line appended per evaluation)",
    )
    parser.add_argument("--out-json", default=None, help="Optional JSON path for best solution")
    parser.add_argument(
        "--K", nargs=2, type=int, metavar=("MIN", "MAX"), default=[2, 30],
        help="Search bounds for K",
    )
    parser.add_argument(
        "--K_mode", nargs="+", choices=VALID_K_MODES,
        default=list(VALID_K_MODES),
        help="K_mode values to explore (default: both linear and logarithmic)",
    )
    parser.add_argument(
        "--return_nbin", nargs=2, type=int, metavar=("MIN", "MAX"), default=[10, 80],
        help="Search bounds for decode-target (log return) bins",
    )
    parser.add_argument(
        "--variance_nbin", nargs=2, type=int, metavar=("MIN", "MAX"), default=[2, 16],
        help="Search bounds for secondary (trailing return variance) bins",
    )
    parser.add_argument(
        "--variance_window", nargs=2, type=int, metavar=("MIN", "MAX"), default=[3, 21],
        help="Search bounds for trailing variance window, minimum 2",
    )
    parser.add_argument(
        "--binmode", nargs="+", choices=VALID_BINMODES,
        default=list(VALID_BINMODES),
        help="binmode values to explore (default: rbf and rbf-quantile)",
    )
    parser.add_argument(
        "--taumin", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.1, 100.0],
        help="Search bounds for taumin (fastest LTSL time constant)",
    )
    parser.add_argument(
        "--taumax", nargs=2, type=float, metavar=("MIN", "MAX"), default=[10.0, 800.0],
        help="Search bounds for taumax (slowest LTSL time constant; must exceed taumin)",
    )
    parser.add_argument(
        "--bdebias", nargs="+", type=int, choices=VALID_BDEBIAS,
        default=list(VALID_BDEBIAS),
        help="bdebias values to explore (default: 0 and 1)",
    )

    args = parser.parse_args()
    os.chdir(args.chdir)

    vw_min, vw_max = min(args.variance_window), max(args.variance_window)
    if vw_min < 2:
        parser.error("variance_window lower bound must be >= 2.")
    if not 0.0 < args.var_level < 1.0:
        parser.error("var_level must lie strictly in (0, 1).")
    tmin_lo, tmin_hi = min(args.taumin), max(args.taumin)
    tmax_lo, tmax_hi = min(args.taumax), max(args.taumax)
    if tmin_lo >= tmin_hi or tmax_lo >= tmax_hi:
        parser.error("taumin and taumax bounds must each have MIN < MAX.")
    if max(tmax_lo, tmin_lo + TAU_STRICT_GAP) > tmax_hi:
        parser.error(
            "Infeasible taumin/taumax bounds: no value pair with taumax > taumin "
            f"exists within [{tmin_lo}, {tmin_hi}] x [{tmax_lo}, {tmax_hi}].")

    b = Bounds(
        K=(min(args.K), max(args.K)),
        K_mode=list(dict.fromkeys(args.K_mode)),
        return_nbin=(min(args.return_nbin), max(args.return_nbin)),
        variance_nbin=(min(args.variance_nbin), max(args.variance_nbin)),
        variance_window=(vw_min, vw_max),
        binmode=list(dict.fromkeys(args.binmode)),
        taumin=(tmin_lo, tmin_hi),
        taumax=(tmax_lo, tmax_hi),
        bdebias=list(dict.fromkeys(args.bdebias)),
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
        objective=args.objective,
        var_level=args.var_level,
        verbose_eval=args.verbose_eval,
        log_every=args.log_every,
        history_path=args.history,
    )

    print("\n=== Best after time budget ===")
    print(f"  objective          = {args.objective}")
    print(f"  var_level          = {args.var_level}")
    print(f"  K                  = {best.K}")
    print(f"  K_mode             = {best.K_mode}")
    print(f"  return_nbin        = {best.return_nbin}")
    print(f"  variance_nbin      = {best.variance_nbin}")
    print(f"  variance_window    = {best.variance_window}")
    print(f"  binmode            = {best.binmode}")
    print(f"  taumin             = {best.taumin}")
    print(f"  taumax             = {best.taumax}")
    print(f"  bdebias            = {best.bdebias}")
    print(f"  {args.objective} (minimized) = {best.fitness}")
    print(f"  rmse_train         = {best.rmse_train}")
    print(f"  rmse_validation    = {best.rmse_validation}")
    print(f"  rmse_unseen        = {best.rmse_unseen}")
    print(f"  rmse_persistence   = {best.rmse_persistence}")
    print(f"  quadratic_loss     = {best.quadratic_loss}")
    print(f"  smooth_loss        = {best.smooth_loss}")
    print(f"  tick_loss          = {best.tick_loss}")
    print(f"  firm_loss          = {best.firm_loss}")
    print(f"  violation_rate     = {best.violation_rate}")
    print(f"  n_violations       = {best.n_violations}")

    if args.out_json:
        out = {
            "objective": args.objective,
            "var_level": args.var_level,
            "fitness": best.fitness,
            "K": best.K,
            "K_mode": best.K_mode,
            "return_nbin": best.return_nbin,
            "variance_nbin": best.variance_nbin,
            "variance_window": best.variance_window,
            "binmode": best.binmode,
            "taumin": best.taumin,
            "taumax": best.taumax,
            "bdebias": best.bdebias,
            "rmse_train": best.rmse_train,
            "rmse_validation": best.rmse_validation,
            "rmse_unseen": best.rmse_unseen,
            "rmse_persistence": best.rmse_persistence,
            "quadratic_loss": best.quadratic_loss,
            "smooth_loss": best.smooth_loss,
            "tick_loss": best.tick_loss,
            "firm_loss": best.firm_loss,
            "violation_rate": best.violation_rate,
            "n_violations": best.n_violations,
            "base_paramfile": args.base,
            "time_budget_s": args.time,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
