FlowFiebig 20260801

This project aimed to test a novel combination of a Bayesian Confidence Propagation Neural Network (BCPNN) with an explicit
logarithmic temporal scale ladder (LTSL) of time-causal, time-recursive smoothing kernels in the sense of Lindeberg's 
temporal scale-space theory. The LTSL maintains K geometrically spaced leaky integrators
over a radial-basis-function encoding of the input, giving the network a compact multi-scale memory of the past that
requires no buffering, never accesses the future, and thus enables continual learning. The BCPNN associates that
multi-scale code with the next observation and importantly, at every step, emits a full discrete predictive distribution
rather than a point estimate. 

Included notebooks test out this model on Mackey–Glass chaotic time series and S&P 100 equities, but the eventual 
goal was to benchmark this models performance against a State-Of-The-Art risk estimation model called 
DeepVaR (Fatouros et al., 2023). Our direct comparison with DeepVaR demonstrates that BCPNN-LTSL enables a 
veryfast and compute efficient prediction of Value-At-Risk without sacrificing performance against the state-of-the-art.

A comprehensive technical report on this projects current achievements can be found in the documentation subfolder, along with the preceeding DeepVAR model that we benchmark against.

The original BCPNN implementation was by Anders Lansner(ALa), and is based on BCPNNSim2.2, found at
https://github.com/anderslan/BCPNNSim2.2/tree/master

nb-FOREX.ipynb 
This is the main notebook that uses the same FOREX dataset used by the Authors of DeepVaR, to evaluate BCPNN as a risk estimation model.

nb-multislice_return_encoding.ipynb
This notebooks evaluates the encoding: distribution, code-space usage, and quantile coding

nb-SYNTH.ipynb
This notebook was used to calibrate the probability estimations long tail at 1% to achive a calibrated 99%-Value-At-Risk model.

nb-multislice_mean_1stderiv.ipynb
nb-multislice_return_variance.ipynb
nb-1ticker.ipynb
These earlier notebooks leading up to this benchmarked study also use S&P100 Stock data and test different feature sets
