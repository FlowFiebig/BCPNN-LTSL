# BCPNN-LTSL

Private research code for Florian Fiebig (FlowFiebig), 2026-08-01. All rights reserved.

This project tests a Bayesian Confidence Propagation Neural Network (BCPNN) combined with an explicit logarithmic temporal scale ladder (LTSL) of time-causal, time-recursive smoothing kernels, in the sense of Lindeberg’s temporal scale-space theory. The LTSL maintains K geometrically spaced leaky integrators over a radial-basis-function encoding of the input. That gives the network a compact multi-scale memory of the past that requires no buffering, never accesses the future, and thus enables continual learning. The BCPNN associates that multi-scale code with the next observation and, at every step, emits a full discrete predictive distribution rather than a point estimate.

Included notebooks first test the model on Mackey–Glass chaotic time series and S&P 100 equities. The main endpoint is a benchmark against DeepVaR (Fatouros et al., 2023) on a FOREX dataset. Direct comparison shows that BCPNN-LTSL can produce a very fast, compute-efficient Value-at-Risk prediction without sacrificing performance against that state-of-the-art baseline.

A technical report is in [`documentation/`](documentation/).

The original BCPNN implementation is by Anders Lansner and is based on [BCPNNSim2.2](https://github.com/anderslan/BCPNNSim2.2/tree/master).

## Repository layout

- [`libsrc_cu/`](libsrc_cu/) — CUDA/C++ BCPNN library (LTSL extensions live here)
- [`tests/UC2-EXTRABRAIN/LTSLtest/financial/`](tests/UC2-EXTRABRAIN/LTSLtest/financial/) — FOREX / S&P 100 notebooks and parameter files
- [`tests/UC2-EXTRABRAIN/LTSLtest/general/`](tests/UC2-EXTRABRAIN/LTSLtest/general/) — earlier Mackey–Glass and sum-of-sinusoids experiments
- [`documentation/`](documentation/) — project technical report

## Notebooks

Paths are under `tests/UC2-EXTRABRAIN/LTSLtest/`.

**Financial (main study)**

- `financial/nb-FOREX.ipynb` — main notebook: same FOREX dataset as DeepVaR, used to evaluate BCPNN as a risk-estimation model
- `financial/nb-multislice_return_encoding.ipynb` — encoding: distribution, code-space usage, and quantile coding
- `financial/nb-SYNTH.ipynb` — calibrate the long tail at 1% for a calibrated 99% Value-at-Risk model
- `financial/nb-multislice_mean_1stderiv.ipynb`, `financial/nb-multislice_return_variance.ipynb`, `financial/nb-1ticker.ipynb` — earlier S&P 100 notebooks with different feature sets

**General (earlier development)**

- `general/notebook-MG_offs1.ipynb` — Mackey–Glass chaotic time series (point estimation / free generation)
- `general/notebook-SOS_offs1.ipynb` — multiscale sum of sinusoids

## What is not in git

Generated simulation outputs stay on disk locally but are excluded from GitHub (several files exceed GitHub’s 100 MB limit; the working tree is about 5.4 GB). That includes:

- `features.dat` and `netw1_*.bin` activation / distribution logs
- compiled objects (`*.o`, `libbcpnn-2.2.a`) and the `ltslmain` binary

Rebuild them by compiling the library and re-running the notebooks.

## Build (CUDA)

Requires NVIDIA CUDA (the bundled BCPNNSim notes assume CUDA ≥ 12.3 and GPU compute capability ≥ 7.0).

```bash
cd libsrc_cu
make
cd ../tests/UC2-EXTRABRAIN/LTSLtest
# or: ./recompile.sh
make -f Makefile_lib.workstation
```

`recompile.sh` rebuilds `libsrc_cu` and then `ltslmain`.

## Python notebooks

```bash
pip install -r requirements.txt
```

Then open the notebooks from `tests/UC2-EXTRABRAIN/LTSLtest/financial/` or `.../general/`. Datasets used by the notebooks are under each folder’s `datasets/` directory.

## Reference

Fatouros, G., Makridis, G., Kotios, D., Soldatos, J., Filippakis, M., & Kyriazis, D. (2023). DeepVaR: a framework for portfolio risk assessment leveraging probabilistic deep neural networks. *Digital Finance*, 5, 29–56. https://doi.org/10.1007/s42521-022-00071-9
