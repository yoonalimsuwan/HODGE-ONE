=======================================================================
HODGE ONE — Differentiable Hodge Conjecture Platform (SSC Edition)
=======================================================================
Author : Yoon A Limsuwan 
License: MIT
Year   : 2025

A fully differentiable computational framework that combines
Semantic‑State Contraction (SSC) dynamics with a learnable
Self‑Organised Criticality (SOC) kernel to explore the
Hodge Conjecture.

The central idea:
  • Particles (positions) represent sampling points on an algebraic cycle.
  • SSC dynamics (with learnable SOC kernel) govern particle evolution.
  • A differentiable period map converts particle positions into a period
    vector (simulating integration of a holomorphic form).
  • Training minimizes the discrepancy between the computed period vector
    and a target Hodge class, thereby tuning the SOC kernel and SSC
    parameters to "grow" an appropriate algebraic cycle.

This is a research prototype — it does NOT prove the Hodge Conjecture
but offers a gradient‑based search for algebraic cycles using
ideas from self‑organised criticality.

Open‑source foundations:
  • PyTorch (BSD)         — automatic differentiation & GPU
  • NumPy (BSD)           — numerical arrays
  • SciPy (BSD)           — optional special functions
  • Matplotlib (PSF)      — visualisation
=======================================================================
"""

import math, os, sys, argparse, logging, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("HODGE_ONE_SSC")

# ========================== Device & Utilities =================================
def get_device(preferred: str = "cpu") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ==================== Learnable SOC Kernel (from RH ONE) =======================
class LearnableSOCKernel(nn.Module):
    """Learnable Self‑Organised Criticality kernel with trainable parameters."""
    def __init__(self, init_Cs: float = 0.18, init_lambda: float = 12.0,
                 init_alpha: float = 0.5, init_tau: float = 10.0,
                 device: str = 'cpu'):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(init_Cs), device=device))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda), device=device))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), device=device))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), device=device))

    @property
    def Cs(self): return torch.exp(self.log_Cs)
    @property
    def lambd(self): return torch.exp(self.log_lambda)
    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def tau(self): return torch.exp(self.log_tau)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.Cs * torch.pow(r + 1e-6, -self.alpha) * torch.exp(-r / self.lambd)

# ==================== DiffRGRefiner (optional) =================================
class DiffRGRefiner(nn.Module):
    """Differentiable Renormalisation Group filter (Fourier low‑pass)."""
    def __init__(self, keep_fraction: float = 0.5):
        super().__init__()
        self.keep_fraction = keep_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("DiffRGRefiner expects (batch, length) input")
        x_hat = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(x.size(1), device=x.device)
        mask = freqs <= (self.keep_fraction * freqs.max())
        mask = mask.to(x.dtype).unsqueeze(0)
        return torch.fft.irfft(x_hat * mask, n=x.size(1), dim=1)

# ==================== Soft Histogram (Differentiable Density) =================
def soft_histogram(x: torch.Tensor, grid: torch.Tensor,
                   sigma: float) -> torch.Tensor:
    """
    Differentiable density estimator using a Gaussian kernel on a regular grid.
    Input:
        x    : (batch, N) particle positions
        grid : (NGRID,)  evaluation points
        sigma: kernel bandwidth
    Returns:
        density : (batch, NGRID)  probability density estimate
    """
    dx = grid[1] - grid[0]
    diff = x.unsqueeze(2) - grid.unsqueeze(0).unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    density = weights.sum(dim=1) / (weights.sum(dim=1).sum(dim=1, keepdim=True) * dx + 1e-12)
    return density

# ==================== SSC Simulator (adapted from RH ONE) ======================
class SSCSimulator(nn.Module):
    """
    Semantic‑State Contraction particle dynamics.
    Drift   : -α H[ρ] * K(r)  - β ∇ρ  + γ x
    Noise   : σ √dt  * dW
    ρ is obtained via soft histogram.
    """
    def __init__(self, N_particles: int, XMIN: float, XMAX: float, NGRID: int,
                 alpha: float = 0.8, beta: float = 0.05, gamma: float = 0.0,
                 sigma: float = 0.3, dt: float = 0.01,
                 soc_kernel: Optional[LearnableSOCKernel] = None,
                 rg_filter: Optional[DiffRGRefiner] = None,
                 device: str = 'cpu'):
        super().__init__()
        self.N = N_particles
        self.XMIN = XMIN
        self.XMAX = XMAX
        self.NGRID = NGRID
        self.dt = dt
        self.soc = soc_kernel if soc_kernel is not None else LearnableSOCKernel(device=device)
        self.rg = rg_filter
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha), device=device))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta), device=device))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(abs(gamma)+1e-6), device=device))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), device=device))
        self.grid = nn.Parameter(torch.linspace(XMIN, XMAX, NGRID, device=device),
                                 requires_grad=False)
        dx = (XMAX - XMIN) / (NGRID - 1)
        self.sigma_kde = dx * 0.5

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self): return torch.exp(self.log_beta)
    @property
    def gamma(self): return torch.exp(self.log_gamma)
    @property
    def sigma_val(self): return torch.exp(self.log_sigma)

    def initial_uniform(self, batch_size: int = 1) -> torch.Tensor:
        return torch.rand(batch_size, self.N, device=self.grid.device) * \
               (self.XMAX - self.XMIN) + self.XMIN

    def hilbert_fft(self, density: torch.Tensor) -> torch.Tensor:
        """Hilbert transform of a periodic density via FFT (differentiable)."""
        F = torch.fft.fft(density, dim=1)
        k = torch.fft.fftfreq(density.size(1), device=density.device)
        mult = -1j * torch.where(k > 0, torch.ones_like(k),
                                 torch.where(k < 0, -torch.ones_like(k), torch.zeros_like(k)))
        H = torch.real(torch.fft.ifft(F * mult.unsqueeze(0), dim=1))
        return H

    def step(self, x: torch.Tensor) -> torch.Tensor:
        # differentiable density via soft histogram
        density = soft_histogram(x, self.grid, self.sigma_kde)

        if self.rg is not None:
            density = self.rg(density)

        H = self.hilbert_fft(density)
        dx = self.grid[1] - self.grid[0]

        # Gradient of density (finite difference on grid, differentiable)
        grad = torch.zeros_like(density)
        grad[:, 1:-1] = (density[:, 2:] - density[:, :-2]) / (2*dx)

        # Interpolate H and grad at particle positions
        idx = ((x - self.XMIN) / dx).long().clamp(0, self.NGRID-2)
        x0 = self.grid[idx]
        x1 = self.grid[idx+1]
        w1 = (x - x0) / (x1 - x0)
        w0 = 1.0 - w1

        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        Hp = w0 * H[batch_idx, idx] + w1 * H[batch_idx, idx+1]
        Gp = w0 * grad[batch_idx, idx] + w1 * grad[batch_idx, idx+1]

        soc_scale = self.soc(torch.abs(x - self.XMIN) / (self.XMAX - self.XMIN))
        drift = -self.alpha * Hp * soc_scale - self.beta * Gp + self.gamma * x
        noise = self.sigma_val * math.sqrt(self.dt) * torch.randn_like(x)
        return x + drift * self.dt + noise

    def simulate(self, num_steps: int, initial_x: Optional[torch.Tensor] = None,
                 batch_size: int = 1) -> torch.Tensor:
        if initial_x is None:
            x = self.initial_uniform(batch_size)
        else:
            x = initial_x
        for _ in range(num_steps):
            x = self.step(x)
        return x

# ==================== Differentiable Period Computer ===========================
class DifferentiablePeriodComputer(nn.Module):
    """
    Maps particle positions to a period vector.
    In a real setting, periods arise from integration of a holomorphic form
    over a basis of homology cycles. Here we construct a differentiable
    surrogate: the period vector is a sum of basis functions evaluated at
    particle positions. This allows the SSC dynamics to influence the
    computed periods.
    """
    def __init__(self, N_particles: int, period_dim: int,
                 XMIN: float, XMAX: float, device: str = 'cpu'):
        super().__init__()
        self.N = N_particles
        self.period_dim = period_dim
        # A fixed "basis" of functions (sine/cosine) to map positions to periods
        # We'll use a random but fixed linear projection + nonlinear activation
        self.proj = nn.Linear(N_particles, period_dim, bias=False)
        # Freeze projection weights to keep mapping consistent
        self.proj.weight.requires_grad = False
        # Scale positions to a suitable range
        self.XMIN = XMIN
        self.XMAX = XMAX

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, N) particle positions (after SSC simulation)
        returns: (batch, period_dim) period vectors
        """
        # Normalize positions to [0, 2π]
        x_norm = (x - self.XMIN) / (self.XMAX - self.XMIN) * 2 * math.pi
        # Simple period mapping: sin of positions to create a high-dimensional feature
        feat = torch.sin(x_norm)  # (batch, N)
        periods = self.proj(feat)  # (batch, period_dim)
        # Normalise to unit norm
        periods = periods / (periods.norm(dim=1, keepdim=True) + 1e-8)
        return periods

# ==================== Hodge Class Target =======================================
class HodgeClass:
    """Holds a target period vector (Hodge class)."""
    def __init__(self, vector: torch.Tensor):
        self.vector = vector / (vector.norm() + 1e-8)

    @staticmethod
    def random(dim: int, device: str = 'cpu') -> 'HodgeClass':
        v = torch.randn(dim, device=device)
        return HodgeClass(v)

# ==================== Training Manager =========================================
class HodgeSSCTrainer:
    """
    Orchestrates training of the SSC simulator (including its SOC kernel)
    to make the computed period vector match a target Hodge class.
    """
    def __init__(self, simulator: SSCSimulator,
                 period_computer: DifferentiablePeriodComputer,
                 target: HodgeClass,
                 lr: float = 0.01):
        self.sim = simulator
        self.period_computer = period_computer
        self.target = target
        # Collect all trainable parameters
        params = list(simulator.parameters()) + list(period_computer.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        periods = self.period_computer(x)
        loss = torch.sum((periods - self.target.vector.unsqueeze(0))**2)
        return loss

    def train_step(self, num_sim_steps: int = 100, batch_size: int = 1):
        self.sim.train()
        self.period_computer.train()
        self.optimizer.zero_grad()
        x = self.sim.initial_uniform(batch_size)
        # Simulate SSC dynamics (differentiable)
        for _ in range(num_sim_steps):
            x = self.sim.step(x)
        loss = self.compute_loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs: int = 500, num_sim_steps: int = 100, batch_size: int = 1):
        for epoch in range(epochs):
            loss = self.train_step(num_sim_steps, batch_size)
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch:3d} | loss = {loss:.6f}")
                # Log SOC kernel parameters
                soc = self.sim.soc
                logger.info(f"  SOC params: Cs={soc.Cs.item():.3f}, lambda={soc.lambd.item():.3f}, "
                            f"alpha={soc.alpha.item():.3f}, tau={soc.tau.item():.3f}")
        return loss

# ==================== Main CLI =================================================
def parse_args():
    p = argparse.ArgumentParser(description="HODGE ONE — SSC Edition")
    p.add_argument('--mode', default='train', choices=['train','demo','info'])
    p.add_argument('--device', default='cpu', choices=['cpu','cuda','mps'])
    p.add_argument('--N', type=int, default=200, help='Number of particles')
    p.add_argument('--XMIN', type=float, default=-5.0)
    p.add_argument('--XMAX', type=float, default=5.0)
    p.add_argument('--NGRID', type=int, default=256)
    p.add_argument('--period-dim', type=int, default=5, help='Dimension of target Hodge class')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--sim-steps', type=int, default=80, help='SSC simulation steps per epoch')
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    if args.mode == 'train':
        # Generate a random target Hodge class
        target = HodgeClass.random(args.period_dim, device=device)
        logger.info(f"Target Hodge class vector: {target.vector.cpu().numpy()}")

        # Build simulator with learnable SOC kernel
        soc_kernel = LearnableSOCKernel(device=device)
        sim = SSCSimulator(N_particles=args.N, XMIN=args.XMIN, XMAX=args.XMAX,
                           NGRID=args.NGRID,
                           soc_kernel=soc_kernel,
                           device=device).to(device)

        # Build period computer
        period_comp = DifferentiablePeriodComputer(N_particles=args.N,
                                                   period_dim=args.period_dim,
                                                   XMIN=args.XMIN, XMAX=args.XMAX,
                                                   device=device).to(device)

        trainer = HodgeSSCTrainer(sim, period_comp, target, lr=args.lr)
        final_loss = trainer.train(epochs=args.epochs,
                                   num_sim_steps=args.sim_steps,
                                   batch_size=1)
        logger.info(f"Training finished with loss = {final_loss:.6f}")

        # Show final period vector
        with torch.no_grad():
            x = sim.initial_uniform(1)
            for _ in range(args.sim_steps):
                x = sim.step(x)
            final_periods = period_comp(x)
            logger.info(f"Final period vector   : {final_periods[0].cpu().numpy()}")
            logger.info(f"Target period vector  : {target.vector.cpu().numpy()}")

    elif args.mode == 'demo':
        # Quick demo: create simulator, run simulation, show period
        soc_kernel = LearnableSOCKernel(device=device)
        sim = SSCSimulator(N_particles=args.N, XMIN=args.XMIN, XMAX=args.XMAX,
                           NGRID=args.NGRID, soc_kernel=soc_kernel,
                           device=device).to(device)
        period_comp = DifferentiablePeriodComputer(N_particles=args.N,
                                                   period_dim=args.period_dim,
                                                   XMIN=args.XMIN, XMAX=args.XMAX,
                                                   device=device).to(device)
        x = sim.initial_uniform(1)
        for _ in range(args.sim_steps):
            x = sim.step(x)
        periods = period_comp(x)
        logger.info(f"Demo period vector: {periods[0].cpu().numpy()}")

    elif args.mode == 'info':
        print("HODGE ONE — SSC Edition")
        print("This platform uses SSC dynamics with a learnable SOC kernel")
        print("to explore the representation of Hodge classes by algebraic cycles.")
        print("The mapping from particles to periods is a differentiable surrogate,")
        print("allowing gradient‑based optimisation of the entire system.")
        print("Run with --mode train to experiment.")

    else:
        logger.error("Unknown mode. Use train, demo, or info.")

if __name__ == "__main__":
    main()
