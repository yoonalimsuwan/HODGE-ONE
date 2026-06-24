`
# HODGE ONE – Differentiable Hodge Conjecture Platform (SSC Edition)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19869633-blue)](https://doi.org/10.5281/zenodo.19869633)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20821864-blue)](https://doi.org/10.5281/zenodo.20821864)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20502106-blue)](https://doi.org/10.5281/zenodo.20502106)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)


A fully differentiable computational framework that combines **Semantic‑State Contraction (SSC) dynamics** with a **learnable Self‑Organised Criticality (SOC) kernel** to explore the **Hodge Conjecture**.

The central idea:
- **Particles** (positions) represent sampling points on an algebraic cycle.
- **SSC dynamics** (with learnable SOC kernel) govern particle evolution.
- A **differentiable period map** converts particle positions into a period vector (simulating integration of a holomorphic form).
- Training minimises the discrepancy between the computed period vector and a target Hodge class, thereby tuning the SOC kernel and SSC parameters to "grow" an appropriate algebraic cycle.

This is a **research prototype** – it does **not** prove the Hodge Conjecture but offers a gradient‑based search for algebraic cycles using ideas from self‑organised criticality.

---

## Features
- Differentiable SSC simulator with trainable parameters
- Learnable SOC kernel (`LearnableSOCKernel`) with log‑space parameterisation
- Differentiable period computer as a surrogate for Hodge‑period integration
- Gradient‑based optimisation of the whole system via PyTorch
- Optional differentiable Renormalisation Group (RG) filter
- Soft histogram density estimation for fully differentiable density analysis
- Logging of SOC parameters during training

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (>=1.10)
- NumPy
- SciPy
- Matplotlib (optional, for plotting)

### Setup
```bash
git clone https://github.com/yoonalimsuwan/HODGE-ONE.git
cd hodge-one
pip install -r requirements.txt
```

requirements.txt (create if not present):

```
torch>=1.10
numpy
scipy
matplotlib
```

---

Quick Start

1. Demo run (no training, just a simulation)

```bash
python hodge_one_ssc.py --mode demo --N 200 --sim-steps 80
```

2. Train to match a random Hodge class

```bash
python hodge_one_ssc.py --mode train --epochs 500 --lr 0.02 --device cpu
```

3. Show information about the platform

```bash
python hodge_one_ssc.py --mode info
```

---

Command‑Line Arguments

Argument Default Description

--mode train train, demo, or info
--device cpu cpu, cuda, or mps
--N 200 Number of particles
--XMIN -5.0 Lower bound of the 1D domain
--XMAX 5.0 Upper bound of the 1D domain
--NGRID 256 Number of grid points for density estimation
--period-dim 5 Dimension of the target Hodge class vector
--epochs 500 Training epochs
--sim-steps 80 SSC simulation steps per training epoch
--lr 0.02 Learning rate for the Adam optimiser
--seed 42 Random seed for reproducibility

---

Project Structure

· LearnableSOCKernel – trainable SOC kernel with parameters Cs, λ, α, τ.
· DiffRGRefiner – (optional) differentiable low‑pass Fourier filter.
· soft_histogram – differentiable density estimator using a Gaussian kernel.
· SSCSimulator – particle simulator with drift and noise, using the SOC kernel and density.
· DifferentiablePeriodComputer – maps particle positions to a period vector via a fixed projection.
· HodgeClass – simple wrapper for a target period vector.
· HodgeSSCTrainer – handles training loop, loss computation, and logging.

The training loop:

1. Initialise particles uniformly.
2. Evolve them using the SSC simulator (gradients tracked through all steps).
3. Compute the period vector from final positions.
4. Minimise squared distance to the target Hodge class vector.
5. Backpropagate to update the SOC kernel and SSC parameters.

---

How It Works

We model a 1D "cycle" as a set of N particles whose positions evolve under a stochastic differential equation that depends on:

· a density‑dependent drift derived from a Hilbert transform (mimicking self‑organised criticality),
· a learnable SOC kernel that weights interactions by distance,
· density gradient (entropy‑like pressure),
· optional linear restoring force.

The final particle positions are passed to a differentiable “period computer” – a simple surrogate that mimics the integration of a holomorphic differential form over an algebraic cycle. By minimising the difference between the computed period vector and a randomly chosen target Hodge class, the system searches for particle configurations that might represent a valid algebraic cycle realising that Hodge class.

In theory, if such a configuration exists, the optimised SOC parameters and particle distribution could hint at the structure of the algebraic cycle. In practice, this is a toy model and not a rigorous mathematical tool.

---

Limitations & Disclaimer

· Toy model only. The domain is 1‑dimensional and the period map is a fixed, hand‑designed surrogate. It does not correspond to any actual algebraic variety or Hodge decomposition.
· No algebraic geometry guarantee. Particle configurations do not necessarily form a valid algebraic subvariety, and no geometric constraints are enforced.
· The Hodge Conjecture is a deep, unsolved problem in complex geometry. This code provides a numerical playground to experiment with gradient‑based search for algebraic cycles; it does not prove or disprove the conjecture.
· Numerical instability. The SSC dynamics may produce divergent behaviour; hyperparameters must be tuned carefully.
· Hardware requirements. Training is CPU‑friendly but benefits from GPU acceleration if available.

Use this code for experimentation, education, and inspiration only. For a rigorous study of the Hodge Conjecture, please refer to classical mathematical literature.

---

License

MIT License – see the LICENSE file for details.

Author

Yoon A Limsuwan – 2026

Open‑source components: PyTorch (BSD), NumPy (BSD), SciPy (BSD), Matplotlib (PSF).

```
