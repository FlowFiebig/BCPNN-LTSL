#!/usr/bin/env python3
"""
Evolutionary search (v2) for K, K_mode, mean_nbin, deriv1st_nbin, and taumax to
minimize ``rmse_validation`` from :func:`Mains.multi`. Runs until a wall-clock
time budget (default 1 hour).

Differences from ``evolve_hyperparams.py`` (v1):

- Drives ``Mains.multi`` instead of ``Mains.general``.
- Search space restricted to K, K_mode, mean_nbin, deriv1st_nbin, taumax.
  outagain_gen is dropped (free generation disabled in multi-mode) and trnpat
  is dropped (derived from ``train_files`` and ignored by ``Mains.multi``).
- Objective is fixed to rmse_validation (no --objective flag).
- ``train_files`` / ``val_files`` / ``test_files`` from the base .par file are
  preserved on every evaluation; the EA does not mutate them.

Run from the ``general`` directory (or pass --chdir). Default --base is
``Parameters_sp100.par`` (78 train / 10 val / 10 test SP100 split with late-IPO
trimming handled by ``phase_to_concat_features`` via ``skip_leading_nan=True``).
Example:

    python evolve_hyperparams_v2.py --time 3600 \\
        --base Parameters_sp100.par \\
        --K 5 25 --K_mode linear logarithmic \\
        --mean_nbin 15 50 --deriv1st_nbin 15 50 \\
        --taumax 10.0 1000.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import Mains
import Utils


VALID_K_MODES = ("linear", "logarithmic")
OBJECTIVE = "rmse_validation"  # fixed in v2


@dataclass
class Bounds:
    K: Tuple[int, int]
    K_mode: Sequence[str]
    mean_nbin: Tuple[int, int]
    deriv1st_nbin: Tuple[int, int]
    taumax: Tuple[float, float]

    def clip(self, genes: "Genes") -> "Genes":
        K0, K1 = self.K
        m0, m1 = self.mean_nbin
        d0, d1 = self.deriv1st_nbin
        t0, t1 = self.taumax
        km = genes.K_mode if genes.K_mode in self.K_mode else self.K_mode[0]
        return Genes(
            K=int(np.clip(genes.K, K0, K1)),
            K_mode=km,
            mean_nbin=int(np.clip(genes.mean_nbin, m0, m1)),
            deriv1st_nbin=int(np.clip(genes.deriv1st_nbin, d0, d1)),
            taumax=float(np.clip(genes.taumax, t0, t1)),
        )


@dataclass
class Genes:
    K: int
    K_mode: str
    mean_nbin: int
    deriv1st_nbin: int
    taumax: float
    fitness: Optional[float] = None  # rmse_validation (lower is better)
    rmse_train: Optional[float] = None
    rmse_validation: Optional[float] = None
    rmse_unseen: Optional[float] = None
    rmse_persistence: Optional[float] = None
    per_segment_test_rmse: Dict[str, float] = field(default_factory=dict)

    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.K,
            self.K_mode,
            self.mean_nbin,
            self.deriv1st_nbin,
            round(float(self.taumax), 8),
        )


def random_genes(rng: np.random.Generator, b: Bounds) -> Genes:
    K0, K1 = b.K
    m0, m1 = b.mean_nbin
    d0, d1 = b.deriv1st_nbin
    t0, t1 = b.taumax
    return b.clip(
        Genes(
            K=int(rng.integers(K0, K1 + 1)),
            K_mode=rng.choice(b.K_mode),
            mean_nbin=int(rng.integers(m0, m1 + 1)),
            deriv1st_nbin=int(rng.integers(d0, d1 + 1)),
            taumax=float(rng.uniform(t0, t1)),
        )
    )


def crossover(rng: np.random.Generator, a: Genes, b: Genes, bounds: Bounds) -> Genes:
    g = Genes(
        K=a.K if rng.random() < 0.5 else b.K,
        K_mode=a.K_mode if rng.random() < 0.5 else b.K_mode,
        mean_nbin=a.mean_nbin if rng.random() < 0.5 else b.mean_nbin,
        deriv1st_nbin=a.deriv1st_nbin if rng.random() < 0.5 else b.deriv1st_nbin,
        taumax=a.taumax if rng.random() < 0.5 else b.taumax,
    )
    return bounds.clip(g)


def mutate(rng: np.random.Generator, g: Genes, bounds: Bounds, p: float = 0.25) -> Genes:
    K0, K1 = bounds.K
    m0, m1 = bounds.mean_nbin
    d0, d1 = bounds.deriv1st_nbin
    t0, t1 = bounds.taumax

    K, km, m, d, t = g.K, g.K_mode, g.mean_nbin, g.deriv1st_nbin, g.taumax
    if rng.random() < p:
        step = int(rng.integers(-2, 3))
        K = int(np.clip(K + step, K0, K1))
    if rng.random() < p and len(bounds.K_mode) > 1:
        km = rng.choice(bounds.K_mode)
    if rng.random() < p:
        step = int(rng.integers(-3, 4))
        m = int(np.clip(m + step, m0, m1))
    if rng.random() < p:
        step = int(rng.integers(-3, 4))
        d = int(np.clip(d + step, d0, d1))
    if rng.random() < p:
        span = t1 - t0
        t = float(np.clip(t + rng.normal(0, 0.15 * max(span, 1e-9)), t0, t1))

    return bounds.clip(Genes(K=K, K_mode=km, mean_nbin=m, deriv1st_nbin=d, taumax=t))


def tournament(rng: np.random.Generator, pop: List[Genes], k: int = 3) -> Genes:
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    contestants = [pop[i] for i in idx]
    return min(contestants, key=lambda ind: ind.fitness if ind.fitness is not None else np.inf)


def apply_genes(base_params: Dict[str, Any], g: Genes) -> Dict[str, Any]:
    """Return a copy of base_params with the 5 EA-controlled keys overridden.
    train_files / val_files / test_files and all other base keys are preserved
    verbatim so Mains.multi sees the same dataset split on every evaluation."""
    p = dict(base_params)
    p["K"] = g.K
    p["K_mode"] = g.K_mode
    p["mean_nbin"] = g.mean_nbin
    p["deriv1st_nbin"] = g.deriv1st_nbin
    p["taumax"] = g.taumax
    return p


def _coerce_finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _nanmean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    m = np.isfinite(arr)
    return float(np.mean(arr[m])) if m.any() else float("nan")


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
        g.per_segment_test_rmse = dict(entry["per_segment_test_rmse"])
        return g.fitness

    Utils.write_params(work_par, apply_genes(base_params, g), preserve_comments=False)
    # Mains.multi has its own internal stochasticity; repeat n_repeats times
    # and average per-metric RMSEs to smooth noise (matches v1 convention).
    tr_samples: List[float] = []
    val_samples: List[float] = []
    u_samples: List[float] = []
    pers_samples: List[float] = []
    per_seg_accum: Dict[str, List[float]] = {}
    for _ in range(max(1, int(n_repeats))):
        try:
            result = Mains.multi(paramfile=work_par, exefile=exefile, verbose=verbose_eval)
        except Exception as e:
            if verbose_eval:
                print(f"  eval failed: {e}", file=sys.stderr)
            tr_samples.append(float("nan"))
            val_samples.append(float("nan"))
            u_samples.append(float("nan"))
            pers_samples.append(float("nan"))
            continue
        # Mains.multi returns 6-tuple:
        #   (rmse_train, rmse_validation, rmse_unseen,
        #    rmse_generation, rmse_persistence, per_segment_test_rmse)
        # rmse_generation is NaN in multi-mode; we drop it.
        rmse_tr, rmse_val, rmse_u, _rmse_gen, rmse_pers, per_seg = result
        tr_samples.append(_coerce_finite(rmse_tr))
        val_samples.append(_coerce_finite(rmse_val))
        u_samples.append(_coerce_finite(rmse_u))
        pers_samples.append(_coerce_finite(rmse_pers))
        for label, val in (per_seg or {}).items():
            per_seg_accum.setdefault(label, []).append(_coerce_finite(val))

    g.rmse_train = _nanmean(tr_samples)
    g.rmse_validation = _nanmean(val_samples)
    g.rmse_unseen = _nanmean(u_samples)
    g.rmse_persistence = _nanmean(pers_samples)
    g.per_segment_test_rmse = {lab: _nanmean(vals) for lab, vals in per_seg_accum.items()}

    fit = g.rmse_validation if np.isfinite(g.rmse_validation) else float("inf")
    g.fitness = fit
    cache[key] = {
        "fitness": fit,
        "rmse_train": g.rmse_train,
        "rmse_validation": g.rmse_validation,
        "rmse_unseen": g.rmse_unseen,
        "rmse_persistence": g.rmse_persistence,
        "per_segment_test_rmse": dict(g.per_segment_test_rmse),
    }
    return fit


def _clone(g: Genes) -> Genes:
    return Genes(
        K=g.K,
        K_mode=g.K_mode,
        mean_nbin=g.mean_nbin,
        deriv1st_nbin=g.deriv1st_nbin,
        taumax=g.taumax,
        fitness=g.fitness,
        rmse_train=g.rmse_train,
        rmse_validation=g.rmse_validation,
        rmse_unseen=g.rmse_unseen,
        rmse_persistence=g.rmse_persistence,
        per_segment_test_rmse=dict(g.per_segment_test_rmse),
    )


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
    for required in ("train_files", "val_files", "test_files"):
        if required not in base_params:
            raise KeyError(
                f"Base param file '{base_param_path}' is missing required key "
                f"'{required}'. evolve_hyperparams_v2.py drives Mains.multi(), "
                f"which requires train_files/val_files/test_files. Use a "
                f"Parameters_sp81-style file (or fall back to v1 / Mains.general).")
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
                row = {
                    "t": time.perf_counter() - t_start,
                    "gen": generation,
                    "objective": OBJECTIVE,
                    "K": ind.K,
                    "K_mode": ind.K_mode,
                    "mean_nbin": ind.mean_nbin,
                    "deriv1st_nbin": ind.deriv1st_nbin,
                    "taumax": ind.taumax,
                    "rmse_train": ind.rmse_train,
                    "rmse_validation": ind.rmse_validation,
                    "rmse_unseen": ind.rmse_unseen,
                    "rmse_persistence": ind.rmse_persistence,
                    "per_segment_test_rmse": dict(ind.per_segment_test_rmse),
                    "fitness": ind.fitness,
                    "best_fitness": best_ever.fitness,
                }
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
        description="EA minimizing rmse_validation from Mains.multi() over "
                    "K, K_mode, mean_nbin, deriv1st_nbin, taumax. "
                    "train_files/val_files/test_files are read from --base and held fixed.")
    parser.add_argument("--chdir", default=here, help="Working directory (default: script dir)")
    parser.add_argument("--base", default="Parameters_sp100.par",
                        help="Base .par template (must contain train_files/val_files/test_files); "
                             "default Parameters_sp100.par")
    parser.add_argument("--work-par", default="Parameters_sp100_ea_work.par",
                        help="Scratch .par path written for every evaluation")
    parser.add_argument("--time", type=float, default=3600.0, help="Time budget in seconds (default 3600)")
    parser.add_argument("--pop", type=int, default=12, help="Population size")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exefile", default="./ltslmain", help='Executable (use "" to skip sim)')
    parser.add_argument("--verbose-eval", action="store_true", help="Forward verbose Mains.multi")
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
    parser.add_argument("--K", nargs=2, type=int, metavar=("MIN", "MAX"), default=[5, 25])
    parser.add_argument(
        "--K_mode", nargs="+", choices=VALID_K_MODES,
        default=list(VALID_K_MODES),
        help="K_mode values to explore (default: both linear and logarithmic)",
    )
    parser.add_argument("--mean_nbin", nargs=2, type=int, metavar=("MIN", "MAX"), default=[15, 50])
    parser.add_argument("--deriv1st_nbin", nargs=2, type=int, metavar=("MIN", "MAX"), default=[15, 50])
    parser.add_argument(
        "--taumax", nargs=2, type=float, metavar=("MIN", "MAX"), default=[10.0, 1000.0],
        help="Search bounds for taumax (slowest LTSL time constant; daily-stock "
             "default range 10-1000 days, much larger than v1's FOREX-scale 0.001-0.01).",
    )

    args = parser.parse_args()
    os.chdir(args.chdir)

    b = Bounds(
        K=(min(args.K), max(args.K)),
        K_mode=list(dict.fromkeys(args.K_mode)),
        mean_nbin=(min(args.mean_nbin), max(args.mean_nbin)),
        deriv1st_nbin=(min(args.deriv1st_nbin), max(args.deriv1st_nbin)),
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
    print(f"  objective       = {OBJECTIVE} (fixed in v2)")
    print(f"  K               = {best.K}")
    print(f"  K_mode          = {best.K_mode}")
    print(f"  mean_nbin       = {best.mean_nbin}")
    print(f"  deriv1st_nbin   = {best.deriv1st_nbin}")
    print(f"  taumax          = {best.taumax}")
    print(f"  {OBJECTIVE} (minimized) = {best.fitness}")
    if best.rmse_train is not None:
        print(f"  rmse_train       = {best.rmse_train}")
    if best.rmse_validation is not None:
        print(f"  rmse_validation  = {best.rmse_validation}")
    if best.rmse_unseen is not None:
        print(f"  rmse_unseen      = {best.rmse_unseen}")
    if best.rmse_persistence is not None:
        print(f"  rmse_persistence = {best.rmse_persistence}  "
              f"(naive y_t -> y_{{t+offs}}=y_t baseline; "
              f"any model worth its salt should beat this)")
    if best.per_segment_test_rmse:
        print("  per-segment test RMSE:")
        for label, val in best.per_segment_test_rmse.items():
            print(f"    {label:<32s} = {val:.6g}")

    if args.out_json:
        out = {
            "objective": OBJECTIVE,
            "fitness": best.fitness,
            "K": best.K,
            "K_mode": best.K_mode,
            "mean_nbin": best.mean_nbin,
            "deriv1st_nbin": best.deriv1st_nbin,
            "taumax": best.taumax,
            "rmse_train": best.rmse_train,
            "rmse_validation": best.rmse_validation,
            "rmse_unseen": best.rmse_unseen,
            "rmse_persistence": best.rmse_persistence,
            "per_segment_test_rmse": best.per_segment_test_rmse,
            "base_paramfile": args.base,
            "time_budget_s": args.time,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
