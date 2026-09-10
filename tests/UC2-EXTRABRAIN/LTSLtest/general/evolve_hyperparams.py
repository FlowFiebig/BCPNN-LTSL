#!/usr/bin/env python3
"""
Evolutionary search for K, K_mode, mean_nbin, deriv1st_nbin, taumax, trnpat,
and outagain_gen to minimize rmse_train, rmse_validation, rmse_unseen, or
rmse_generation from Mains.general(). Runs until a wall-clock time budget
(default 1 hour).

outagain_gen is the softmax temperature applied to outpop_md1 only during the
free-generation phase (see ltslmain.cpp); it does not affect train/val/unseen
RMSE, so use --objective rmse_generation when sweeping it.

Run from the ``general`` directory (or pass --chdir). Example:

    python evolve_hyperparams.py --time 3600 \\
        --K 5 25 --K_mode linear logarithmic \\
        --mean_nbin 15 50 --deriv1st_nbin 15 50 \\
        --taumax 0.001 0.01 --trnpat 500 4000 \\
        --outagain_gen 0.01 100

    python evolve_hyperparams.py --objective rmse_generation --time 3600 ...
    python evolve_hyperparams.py --objective rmse_train --time 3600 ...
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

import Mains
import Utils


VALID_K_MODES = ("linear", "logarithmic")


@dataclass
class Bounds:
    K: Tuple[int, int]
    K_mode: Sequence[str]
    mean_nbin: Tuple[int, int]
    deriv1st_nbin: Tuple[int, int]
    taumax: Tuple[float, float]
    trnpat: Tuple[int, int]
    # outagain_gen: softmax gain on outpop_md1 during free generation.
    # Sampled and mutated in log-space because it's a multiplicative gain
    # that can span several orders of magnitude.
    outagain_gen: Tuple[float, float]

    def clip(self, genes: "Genes") -> "Genes":
        K0, K1 = self.K
        m0, m1 = self.mean_nbin
        d0, d1 = self.deriv1st_nbin
        t0, t1 = self.taumax
        r0, r1 = self.trnpat
        og0, og1 = self.outagain_gen
        km = genes.K_mode if genes.K_mode in self.K_mode else self.K_mode[0]
        return Genes(
            K=int(np.clip(genes.K, K0, K1)),
            K_mode=km,
            mean_nbin=int(np.clip(genes.mean_nbin, m0, m1)),
            deriv1st_nbin=int(np.clip(genes.deriv1st_nbin, d0, d1)),
            taumax=float(np.clip(genes.taumax, t0, t1)),
            trnpat=int(np.clip(genes.trnpat, r0, r1)),
            outagain_gen=float(np.clip(genes.outagain_gen, og0, og1)),
        )


@dataclass
class Genes:
    K: int
    K_mode: str
    mean_nbin: int
    deriv1st_nbin: int
    taumax: float
    trnpat: int
    outagain_gen: float = 1.0
    fitness: Optional[float] = None  # objective being minimized (lower is better)
    rmse_train: Optional[float] = None
    rmse_validation: Optional[float] = None
    rmse_unseen: Optional[float] = None
    rmse_generation: Optional[float] = None
    rmse_persistence: Optional[float] = None

    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.K,
            self.K_mode,
            self.mean_nbin,
            self.deriv1st_nbin,
            round(float(self.taumax), 8),
            self.trnpat,
            round(float(self.outagain_gen), 6),
        )


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Sample uniformly in log-space between strictly positive bounds."""
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_genes(rng: np.random.Generator, b: Bounds) -> Genes:
    K0, K1 = b.K
    m0, m1 = b.mean_nbin
    d0, d1 = b.deriv1st_nbin
    t0, t1 = b.taumax
    r0, r1 = b.trnpat
    og0, og1 = b.outagain_gen
    return b.clip(
        Genes(
            K=int(rng.integers(K0, K1 + 1)),
            K_mode=rng.choice(b.K_mode),
            mean_nbin=int(rng.integers(m0, m1 + 1)),
            deriv1st_nbin=int(rng.integers(d0, d1 + 1)),
            taumax=float(rng.uniform(t0, t1)),
            trnpat=int(rng.integers(r0, r1 + 1)),
            outagain_gen=_log_uniform(rng, og0, og1),
        )
    )


def crossover(rng: np.random.Generator, a: Genes, b: Genes, bounds: Bounds) -> Genes:
    g = Genes(
        K=a.K if rng.random() < 0.5 else b.K,
        K_mode=a.K_mode if rng.random() < 0.5 else b.K_mode,
        mean_nbin=a.mean_nbin if rng.random() < 0.5 else b.mean_nbin,
        deriv1st_nbin=a.deriv1st_nbin if rng.random() < 0.5 else b.deriv1st_nbin,
        taumax=a.taumax if rng.random() < 0.5 else b.taumax,
        trnpat=a.trnpat if rng.random() < 0.5 else b.trnpat,
        outagain_gen=a.outagain_gen if rng.random() < 0.5 else b.outagain_gen,
    )
    return bounds.clip(g)


def mutate(rng: np.random.Generator, g: Genes, bounds: Bounds, p: float = 0.25) -> Genes:
    K0, K1 = bounds.K
    m0, m1 = bounds.mean_nbin
    d0, d1 = bounds.deriv1st_nbin
    t0, t1 = bounds.taumax
    r0, r1 = bounds.trnpat
    og0, og1 = bounds.outagain_gen

    K, km, m, d, t, r, og = (
        g.K, g.K_mode, g.mean_nbin, g.deriv1st_nbin, g.taumax, g.trnpat, g.outagain_gen,
    )
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
    if rng.random() < p:
        step = int(rng.integers(-200, 201))
        r = int(np.clip(r + step, r0, r1))
    if rng.random() < p:
        # Multiplicative perturbation in log-space; sigma covers ~25% of the
        # search range per mutation step, similar to the integer/float perturbations above.
        log_span = np.log(max(og1, 1e-12)) - np.log(max(og0, 1e-12))
        sigma = 0.25 * max(log_span, 1e-9)
        og = float(np.clip(np.exp(np.log(max(og, 1e-12)) + rng.normal(0, sigma)), og0, og1))

    return bounds.clip(Genes(
        K=K, K_mode=km, mean_nbin=m, deriv1st_nbin=d, taumax=t, trnpat=r, outagain_gen=og,
    ))


def tournament(rng: np.random.Generator, pop: List[Genes], k: int = 3) -> Genes:
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    contestants = [pop[i] for i in idx]
    return min(contestants, key=lambda ind: ind.fitness if ind.fitness is not None else np.inf)


def apply_genes(base_params: Dict[str, Any], g: Genes) -> Dict[str, Any]:
    p = dict(base_params)
    p["K"] = g.K
    p["K_mode"] = g.K_mode
    p["mean_nbin"] = g.mean_nbin
    p["deriv1st_nbin"] = g.deriv1st_nbin
    p["taumax"] = g.taumax
    p["trnpat"] = g.trnpat
    p["outagain_gen"] = g.outagain_gen
    return p


def _fitness_from_objective(
    objective: str,
    rmse_train: float,
    rmse_validation: float,
    rmse_unseen: float,
    rmse_generation: float,
) -> float:
    if objective == "rmse_unseen":
        v = rmse_unseen
    elif objective == "rmse_generation":
        v = rmse_generation
    elif objective == "rmse_train":
        v = rmse_train
    elif objective == "rmse_validation":
        v = rmse_validation
    else:
        raise ValueError(f"unknown objective: {objective}")
    if v is None or not np.isfinite(v):
        return float("inf")
    return float(v)


def _coerce_finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def evaluate(
    g: Genes,
    base_params: Dict[str, Any],
    work_par: str,
    exefile: Optional[str],
    cache: Dict[Tuple[Any, ...], Tuple[float, float, float, float, float, float]],
    verbose_eval: bool,
    objective: str,
    n_repeats: int = 3,
) -> float:
    key = (objective, g.cache_key())
    if key in cache:
        (fit, g.rmse_train, g.rmse_validation, g.rmse_unseen,
         g.rmse_generation, g.rmse_persistence) = cache[key]
        g.fitness = fit
        return fit

    Utils.write_params(work_par, apply_genes(base_params, g), preserve_comments=False)
    # The simulator carries its own internal stochasticity, so repeating
    # Mains.general for the same .par yields independent samples. Average
    # the per-metric RMSEs across n_repeats runs to smooth out noise.
    tr_samples: List[float] = []
    val_samples: List[float] = []
    u_samples: List[float] = []
    gen_samples: List[float] = []
    pers_samples: List[float] = []
    for _ in range(max(1, int(n_repeats))):
        try:
            result = Mains.general(paramfile=work_par, exefile=exefile, verbose=verbose_eval)
        except Exception as e:
            if verbose_eval:
                print(f"  eval failed: {e}", file=sys.stderr)
            tr_samples.append(float("nan"))
            val_samples.append(float("nan"))
            u_samples.append(float("nan"))
            gen_samples.append(float("nan"))
            pers_samples.append(float("nan"))
            continue
        # Mains.general now returns five RMSEs (rmse_persistence appended);
        # accept four-tuples as well for backwards compatibility.
        if len(result) == 5:
            rmse_tr, rmse_val, rmse_u, rmse_gen, rmse_pers = result
        else:
            rmse_tr, rmse_val, rmse_u, rmse_gen = result
            rmse_pers = float("nan")
        tr_samples.append(_coerce_finite(rmse_tr))
        val_samples.append(_coerce_finite(rmse_val))
        u_samples.append(_coerce_finite(rmse_u))
        gen_samples.append(_coerce_finite(rmse_gen))
        pers_samples.append(_coerce_finite(rmse_pers))

    def _nanmean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        m = np.isfinite(arr)
        return float(np.mean(arr[m])) if m.any() else float("nan")

    g.rmse_train = _nanmean(tr_samples)
    g.rmse_validation = _nanmean(val_samples)
    g.rmse_unseen = _nanmean(u_samples)
    g.rmse_generation = _nanmean(gen_samples)
    g.rmse_persistence = _nanmean(pers_samples)
    fit = _fitness_from_objective(objective, g.rmse_train, g.rmse_validation, g.rmse_unseen, g.rmse_generation)
    cache[key] = (fit, g.rmse_train, g.rmse_validation, g.rmse_unseen,
                  g.rmse_generation, g.rmse_persistence)
    g.fitness = fit
    return fit


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
    objective: str,
    n_repeats: int,
    history_path: Optional[str] = None,
) -> Tuple[Genes, List[Dict[str, Any]]]:
    rng = np.random.default_rng(rng_seed)
    base_params = Utils.load_params(base_param_path)
    cache: Dict[Tuple[Any, ...], Tuple[float, float, float, float, float, float]] = {}
    history: List[Dict[str, Any]] = []

    history_file = open(history_path, "w", encoding="utf-8") if history_path else None
    try:
        t_start = time.perf_counter()
        n_eval = 0

        population: List[Genes] = [random_genes(rng, bounds) for _ in range(pop_size)]
        best_ever: Optional[Genes] = None
    
        def tick_log(force: bool = False) -> None:
            nonlocal n_eval
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
                evaluate(ind, base_params, work_par, exefile, cache, verbose_eval, objective, n_repeats=n_repeats)
                n_eval += 1
                if best_ever is None or (ind.fitness is not None and ind.fitness < best_ever.fitness):
                    best_ever = Genes(
                        K=ind.K,
                        K_mode=ind.K_mode,
                        mean_nbin=ind.mean_nbin,
                        deriv1st_nbin=ind.deriv1st_nbin,
                        taumax=ind.taumax,
                        trnpat=ind.trnpat,
                        outagain_gen=ind.outagain_gen,
                        fitness=ind.fitness,
                        rmse_train=ind.rmse_train,
                        rmse_validation=ind.rmse_validation,
                        rmse_unseen=ind.rmse_unseen,
                        rmse_generation=ind.rmse_generation,
                        rmse_persistence=ind.rmse_persistence,
                    )
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                row = {
                    "t": time.perf_counter() - t_start,
                    "gen": generation,
                    "objective": objective,
                    "K": ind.K,
                    "K_mode": ind.K_mode,
                    "mean_nbin": ind.mean_nbin,
                    "deriv1st_nbin": ind.deriv1st_nbin,
                    "taumax": ind.taumax,
                    "trnpat": ind.trnpat,
                    "outagain_gen": ind.outagain_gen,
                    "rmse_train": ind.rmse_train,
                    "rmse_validation": ind.rmse_validation,
                    "rmse_unseen": ind.rmse_unseen,
                    "rmse_generation": ind.rmse_generation,
                    "rmse_persistence": ind.rmse_persistence,
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
            elites = [
                Genes(
                    K=p.K,
                    K_mode=p.K_mode,
                    mean_nbin=p.mean_nbin,
                    deriv1st_nbin=p.deriv1st_nbin,
                    taumax=p.taumax,
                    trnpat=p.trnpat,
                    outagain_gen=p.outagain_gen,
                    fitness=p.fitness,
                    rmse_train=p.rmse_train,
                    rmse_validation=p.rmse_validation,
                    rmse_unseen=p.rmse_unseen,
                    rmse_generation=p.rmse_generation,
                    rmse_persistence=p.rmse_persistence,
                )
                for p in population[:2]
            ]
    
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
    parser = argparse.ArgumentParser(description="EA minimize rmse_train, rmse_validation, rmse_unseen or rmse_generation (1h default budget)")
    parser.add_argument(
        "--objective",
        choices=("rmse_train", "rmse_validation", "rmse_unseen", "rmse_generation"),
        default="rmse_unseen",
        help="Metric to minimize (default: rmse_unseen). Use rmse_generation only if ngenstep > offs in the base .par, "
             "or rmse_validation only if vanpat > offs. "
             "rmse_train evaluates fit on the training split itself (useful for capacity/diagnostic studies; prone to overfitting).",
    )
    parser.add_argument("--chdir", default=here, help="Working directory (default: script dir)")
    parser.add_argument("--base", default="Parameters.par", help="Base .par template")
    parser.add_argument("--work-par", default="Parameters_ea_work.par", help="Scratch .par path")
    parser.add_argument("--time", type=float, default=3600.0, help="Time budget in seconds (default 3600)")
    parser.add_argument("--pop", type=int, default=12, help="Population size")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exefile", default="./ltslmain", help="Executable (use \"\" to skip sim)")
    parser.add_argument("--verbose-eval", action="store_true", help="Forward verbose Mains.general")
    parser.add_argument("--log-every", type=int, default=1, help="Print progress every N evals (0=only final)")
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="Number of independent simulator runs per genotype; the per-metric "
             "RMSEs are averaged across runs to reduce noise (default 3).",
    )
    parser.add_argument(
        "--history",
        default=None,
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
    parser.add_argument("--taumax", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.001, 0.01])
    parser.add_argument("--trnpat", nargs=2, type=int, metavar=("MIN", "MAX"), default=[500, 4000])
    parser.add_argument(
        "--outagain_gen", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.01, 100.0],
        help="Search bounds for outagain_gen (softmax temperature applied to outpop_md1 "
             "during free generation only). Sampled and mutated in log-space. "
             "Only meaningful with --objective rmse_generation; has no effect on "
             "rmse_train/rmse_validation/rmse_unseen.",
    )

    args = parser.parse_args()
    os.chdir(args.chdir)

    b = Bounds(
        K=(min(args.K), max(args.K)),
        K_mode=list(dict.fromkeys(args.K_mode)),
        mean_nbin=(min(args.mean_nbin), max(args.mean_nbin)),
        deriv1st_nbin=(min(args.deriv1st_nbin), max(args.deriv1st_nbin)),
        taumax=(min(args.taumax), max(args.taumax)),
        trnpat=(min(args.trnpat), max(args.trnpat)),
        outagain_gen=(min(args.outagain_gen), max(args.outagain_gen)),
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
        objective=args.objective,
        n_repeats=args.n_repeats,
        history_path=args.history,
    )

    print("\n=== Best after time budget ===")
    print(f"  objective       = {args.objective}")
    print(f"  K               = {best.K}")
    print(f"  K_mode          = {best.K_mode}")
    print(f"  mean_nbin       = {best.mean_nbin}")
    print(f"  deriv1st_nbin   = {best.deriv1st_nbin}")
    print(f"  taumax          = {best.taumax}")
    print(f"  trnpat          = {best.trnpat}")
    print(f"  outagain_gen    = {best.outagain_gen}")
    print(f"  {args.objective} (minimized) = {best.fitness}")
    if best.rmse_train is not None:
        print(f"  rmse_train      = {best.rmse_train}")
    if best.rmse_validation is not None:
        print(f"  rmse_validation = {best.rmse_validation}")
    if best.rmse_unseen is not None:
        print(f"  rmse_unseen     = {best.rmse_unseen}")
    if best.rmse_generation is not None:
        print(f"  rmse_generation = {best.rmse_generation}")
    if best.rmse_persistence is not None:
        print(f"  rmse_persistence = {best.rmse_persistence}  "
              f"(naive y_t -> y_{{t+offs}}=y_t baseline; "
              f"any model worth its salt should beat this)")

    if args.out_json:
        out = {
            "objective": args.objective,
            "fitness": best.fitness,
            "K": best.K,
            "K_mode": best.K_mode,
            "mean_nbin": best.mean_nbin,
            "deriv1st_nbin": best.deriv1st_nbin,
            "taumax": best.taumax,
            "trnpat": best.trnpat,
            "outagain_gen": best.outagain_gen,
            "rmse_train": best.rmse_train,
            "rmse_validation": best.rmse_validation,
            "rmse_unseen": best.rmse_unseen,
            "rmse_generation": best.rmse_generation,
            "rmse_persistence": best.rmse_persistence,
            "time_budget_s": args.time,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
