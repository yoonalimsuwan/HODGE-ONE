`
# HODGE ONE – Differentiable Hodge Conjecture Platform (SSC Edition)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19869633-blue)](https://doi.org/10.5281/zenodo.19869633)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21548090-blue)](https://doi.org/10.5281/zenodo.21548090)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20502106-blue)](https://doi.org/10.5281/zenodo.20502106)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)


A fully differentiable computational framework that combines **Semantic‑State Contraction (SSC) dynamics** with a **learnable Self‑Organised Criticality (SOC) kernel** to explore the **Hodge Conjecture**.

The central idea:
- **Particles** (positions) represent sampling points on an algebraic cycle.
- **SSC dynamics** (with learnable SOC kernel) govern particle evolution.
- A **differentiable period map** converts particle positions into a period vector (simulating integration of a holomorphic form).
- Training minimises the discrepancy between the computed period vector and a target Hodge class, thereby tuning the SOC kernel and SSC parameters to "grow" an appropriate algebraic cycle.


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
python hodge_one.py --mode demo --N 200 --sim-steps 80
```

2. Train to match a random Hodge class

```bash
python hodge_one.py --mode train --epochs 500 --lr 0.02 --device cpu
```

3. Show information about the platform

```bash
python hodge_one.py --mode info
```

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


Use this code for experimentation, education, and inspiration only. For a rigorous study of the Hodge Conjecture, please refer to classical mathematical literature.

---

License

MIT License – see the LICENSE file for details.

Author

PAI , Yoon A Limsuwan – 2026

Open‑source components: PyTorch (BSD), NumPy (BSD), SciPy (BSD), Matplotlib (PSF).

``
Thanks be to the Father, the Son, and the Holy Spirit, for the grace of Lord Jesus Christ, Mother Mary, Lord Buddha, Guan Yin Bodhisattva, Master Daozhi, Confucius, the Immortal Pae Kow, and Mr. Xi Jinping.

"I love Lim Yoona, Zhou Ye, Karina from aespa, Jessica from Girls' Generation, Zhao Lusi, Nana from After School, and Jiyeon Tara.
​Love Ju Jingyi, Wang Churan, Lu Yuxiao, Bao Shangen , Bailu , Noey , Jam, and Irene
​I love Zhang Linghe, Bai Jingting, Lee Jae-jin, Marc thn , Tance , Green , Taissa Farmiga , Dilraba Dilmurat And Toy Pathompong."


What MSPS NETWORK Sees, the Buddha Knows.
