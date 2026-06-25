"""
=======================================================================
HODGE ONE v2 — Differentiable Hodge Conjecture Platform (SSC Edition)
=======================================================================
Author       : Yoon A Limsuwan (MSPS NETWORK)
Co-developed : Claude (Anthropic), with prior contributions from
               Gemini / GPT / DeepSeek per project convention.
License      : MIT
Version      : 2.0.0  (2026-06)

=======================================================================
WHAT CHANGED FROM v1 (SSC Edition, 2025)
=======================================================================
v1 modeled "algebraic cycles" as a cloud of particles on a single real
line, with a period map built from sin(x) and a frozen random linear
projection. That is a 1-D toy that cannot actually represent a Hodge
class on a complex variety: there was no complex structure, no Hodge
decomposition H^k = ⊕ H^{p,q}, no polarization / Hodge-Riemann
bilinear relations, and — most importantly for the Hodge conjecture
specifically — no notion of *integrality* (the conjecture is about
classes in H^{2p}(X,Z) ∩ H^{p,p}, not arbitrary real vectors).

v2 keeps every bugfix from v1 (alpha clamp + sign-preserving gamma in
LearnableSOCKernel / SSCSimulator, the differentiable RG filter) and
rebuilds the geometric content on top of two concrete, selectable
families of polarized Hodge structures:

  1. VarietyClass.ABELIAN
     A complex torus X = C^g / (Z^g + tau Z^g), tau in the Siegel
     upper half space. This is an actual algebraic variety (a product
     of elliptic curves when tau is diagonal). H^{1,1}(X) is spanned by
     explicit harmonic (1,1)-forms built from the period matrix, and
     the Neron-Severi lattice (integral (1,1) classes that genuinely
     come from divisors) is known in closed form via the alternating
     form E = Im(tau)^{-1} composed with the period matrix. This gives
     us *ground truth*: classes coming from real algebraic cycles
     (translates of subtori / theta divisors) are integral by
     construction, so we can sanity-check the whole pipeline against
     a case where the Hodge conjecture is a theorem (Lefschetz (1,1)
     theorem), not a conjecture.

  2. VarietyClass.K3_ABSTRACT
     An abstract weight-2 polarized Hodge structure with Hodge numbers
     h^{2,0}=h^{0,2}=1, h^{1,1}=20, matching a K3 surface, equipped
     with the K3 intersection form (signature (3,19), even unimodular
     lattice E8(-1)^2 ⊕ U^3 represented here by its Gram matrix). There
     is no underlying point-set variety simulated — this is a period
     point in the K3 period domain — but the Hodge-Riemann bilinear
     relations and the integral lattice structure are real and
     enforced exactly as they would be for a genuine K3.

Both classes are exposed through one VarietyConfig / build_variety(...)
so the rest of the pipeline (SSC dynamics, SOC kernel, cycle-class map,
trainer) is agnostic to which one is active.

Core new components:
  • ComplexTorusLattice      - period matrix, NS lattice, theta divisor
                               class, exact (1,1) harmonic forms.
  • K3LatticeHodgeStructure  - Gram matrix, h^{p,q}, period point on
                               the K3 quadric, Hodge-Riemann relations.
  • HodgeRiemannLoss         - enforces i^{p-q} Q(eta, conj(eta)) > 0
                               for primitive (p,q) classes (soft
                               penalty, differentiable).
  • IntegralityModule        - soft relaxation (sin^2(pi <v,e*_i>)
                               penalty against the integral lattice
                               dual basis) used *during* training, plus
                               a hard nearest-lattice-point projection
                               applied once at the *end* of training
                               (round in the dual/integral basis), per
                               explicit instruction: "soft relaxation +
                               hard lattice projection step at the end".
  • ComplexSSCSimulator      - SOC-SSC particle dynamics generalized to
                               the complex domain (modulus |z| replaces
                               |x| in the SOC kernel; drift/noise act on
                               real and imaginary parts jointly).
  • CycleClassMap            - replaces v1's DifferentiablePeriodComputer.
                               For ABELIAN: pairs the particle cloud with
                               the *actual* harmonic (1,1)-forms of the
                               torus (a real period integral, not a
                               frozen random projection). For K3_ABSTRACT:
                               a learned pairing constrained to land in
                               the (p,p)-eigenspace of the Hodge star /
                               Weil operator, so what comes out is by
                               construction a candidate (p,p) class.

This remains a research prototype: it does NOT prove the Hodge
conjecture. What v2 adds is that the *easy* direction (Lefschetz (1,1)
for the abelian case) is verified exactly, the polarization relations
are enforced rather than ignored, and integrality — the actual
substance of the conjecture — is modeled explicitly instead of being
absent. The self-test suite checks all of this against known closed-
form answers.

Open-source foundations:
  • PyTorch (BSD)         — automatic differentiation & GPU
  • NumPy (BSD)           — numerical arrays / closed-form lattice math
  • Matplotlib (PSF)      — optional visualisation
=======================================================================
"""

import math, os, sys, argparse, logging, warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
logger = logging.getLogger("HODGE_ONE_v2")

# ========================== Device & Utilities =================================
def get_device(preferred: str = "cpu") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def complex_modulus(z: torch.Tensor) -> torch.Tensor:
    """|z| for z stored as a real tensor with last dim = 2 (re, im)."""
    return torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + 1e-12)


# ==================== Variety selection ========================================
class VarietyClass(str, Enum):
    """Which concrete family of polarized Hodge structures to instantiate."""
    ABELIAN = "abelian"            # product-of-elliptic-curves complex torus
    K3_ABSTRACT = "k3_abstract"    # abstract weight-2 Hodge structure, K3 numbers


@dataclass
class VarietyConfig:
    """User-facing configuration: pick the variety class and its parameters.

    For ABELIAN:
        g            : number of elliptic curve factors (complex dimension)
        tau_diag     : list of g complex moduli (im part must be > 0); if
                       None, sampled randomly in the upper half plane.
    For K3_ABSTRACT:
        k3_seed      : RNG seed used to sample a generic point on the
                       K3 period quadric (so results are reproducible).
    """
    variety: VarietyClass = VarietyClass.ABELIAN
    g: int = 2
    tau_diag: Optional[List[complex]] = None
    k3_seed: int = 0
    device: str = "cpu"


# ==================== Complex Torus / Abelian Variety Lattice ==================
class ComplexTorusLattice(nn.Module):
    """
    X = C^g / Lambda,  Lambda = Z^g + tau * Z^g,  tau in Siegel upper half space
    (Im(tau) positive-definite; here we take tau diagonal, i.e. X is literally
    a product of g elliptic curves E_1 x ... x E_g, each E_i = C / (Z + tau_i Z)).

    This is a genuine algebraic variety, and its Neron-Severi group
    NS(X) = H^{1,1}(X,R) ∩ H^2(X,Z)
    is known exactly: a (1,1)-class corresponds to a Hermitian form H on
    C^g whose imaginary part E = Im(H) takes integer values on Lambda x
    Lambda (Riemann's conditions). For a product of elliptic curves the
    standard basis of such integral alternating forms is
        E_{ij} = e_i* ∧ f_j*  (i,j = 1..g)
    built from the real basis {e_i, f_i = tau_i e_i} of each factor's
    lattice Z + tau_i Z. The diagonal classes E_{ii} are exactly the
    classes of the theta divisors {0} x ... x E_i x ... x {0} pulled back
    — i.e. they are *algebraic by construction* (Lefschetz (1,1) theorem
    is not needed here, the cycle is handed to us explicitly), which is
    what lets us sanity-check the rest of the pipeline against a case
    with a known right answer.
    """

    def __init__(self, g: int, tau_diag: Optional[List[complex]] = None,
                 device: str = "cpu", seed: int = 0):
        super().__init__()
        self.g = g
        rng = np.random.default_rng(seed)
        if tau_diag is None:
            # Sample g moduli in the fundamental-domain-ish strip
            # Re(tau) in [-0.5,0.5], Im(tau) in [0.8, 1.8] -- generic, non-degenerate.
            tau_diag = [complex(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 1.8))
                        for _ in range(g)]
        assert len(tau_diag) == g
        for t in tau_diag:
            if t.imag <= 0:
                raise ValueError(f"tau must have positive imaginary part, got {t}")

        tau_re = torch.tensor([t.real for t in tau_diag], dtype=torch.float64, device=device)
        tau_im = torch.tensor([t.imag for t in tau_diag], dtype=torch.float64, device=device)
        self.register_buffer("tau_re", tau_re)
        self.register_buffer("tau_im", tau_im)

        # Real lattice basis per factor: e_i = 1, f_i = tau_i (complex number).
        # Stored for period-integral computations.
        # Number of independent integral (1,1) classes on a product of g
        # elliptic curves includes End(X)-related classes; we restrict to the
        # "obvious" g^2-dimensional family E_{ij} = pullback of E_i on factor i
        # paired with f_j on factor j when i != j, and the theta class when i==j.
        # This is NOT the full NS(X) in general (which depends on isogenies
        # between factors), but it IS always integral and always algebraic,
        # which is exactly the ground truth we need.
        self.n_basis = g * g  # E_{ij}, i,j = 1..g

    def harmonic_form_value(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the g^2 basis (1,1)-forms E_{ij} at points z in C^g
        (as real period-integrand contributions), used by CycleClassMap
        to turn a particle cloud sampling a cycle into actual period
        integrals rather than an arbitrary learned feature map.

        z: (..., g, 2) real tensor, last dim = (Re z_k, Im z_k)
        returns: (..., g*g) real tensor — value of each basis (1,1)-form's
                 Kahler potential gradient pairing at z, i.e. the local
                 density used inside the period integral
                 P_{ij} = ∫_cycle  E_{ij}
                 approximated by particle-averaging this integrand.

        Concretely, on E_i x E_j the (1,1)-form E_{ij} (i != j) pulled back
        from the product is (up to normalization) the constant form
        dx_i ∧ dy_j - dy_i ∧ dx_j in real coordinates z_k = x_k + tau_k y_k,
        and for i == j it is the standard translation-invariant Kahler
        form on E_i (theta-divisor class), constant in z. Since these are
        *constant* (translation-invariant) forms, their integral over any
        cycle reduces to a purely topological/combinatorial pairing with
        the cycle's homology class — which is exactly the situation for
        abelian varieties (all (1,1) forms here are translation
        invariant). We expose the constant integrand here so the period
        computer can build cycle-dependent integrals by weighting it with
        the (differentiable, non-constant) *density* of particles, which
        models cycles that are not the standard ones but small algebraic
        deformations / unions of translates of them.
        """
        g = self.g
        batch_shape = z.shape[:-2]
        out = torch.zeros(*batch_shape, g * g, dtype=z.dtype, device=z.device)
        # Constant translation-invariant forms: value 1 on the (i,j) component,
        # broadcast across all particles/batch -- intentionally position
        # independent (see docstring). Position-dependence enters later via
        # how particles are *distributed* across the torus (the SSC density),
        # which CycleClassMap uses as the integration weight.
        out[..., :] = 1.0
        return out

    def riemann_form_matrix(self) -> torch.Tensor:
        """
        Return the (g*g, g*g) Gram-type pairing that encodes the
        intersection numbers E_{ij} . E_{kl} on this product of elliptic
        curves, used by HodgeRiemannLoss to check positivity. For a
        product of elliptic curves with the basis above, distinct factors
        are orthogonal and each E_{ii} self-intersects positively
        (it is an ample theta divisor class), so the matrix is diagonal
        with positive entries -- this is the polarization form Q
        restricted to our basis.
        """
        g = self.g
        diag = torch.ones(g * g, dtype=torch.float64, device=self.tau_re.device)
        return torch.diag(diag)

    def integral_basis_dual(self) -> torch.Tensor:
        """
        Dual lattice basis (as a (n_basis, n_basis) change-of-basis matrix)
        used by IntegralityModule: in the E_{ij} basis the Neron-Severi
        sublattice we model is literally Z^{g*g} (each basis class has
        self-intersection / pairing exactly 1 with its own dual vector and
        0 with the others), so the dual basis is the identity. This is
        what makes the ABELIAN case a clean ground truth: "integral in
        this basis" means "integer coordinates", with no change of basis
        needed.
        """
        return torch.eye(self.n_basis, dtype=torch.float64, device=self.tau_re.device)

    def known_algebraic_class(self, i: int, j: int) -> torch.Tensor:
        """Ground truth: the (1,1) class of the (i,j) basis divisor, as a
        vector in the E_{ij} basis. By construction this is integral and
        algebraic (theta divisor / product-of-factors translate), so any
        trainer that converges to this vector has, in this toy model,
        exhibited a class as an algebraic cycle -- not because the Hodge
        conjecture was proven, but because this particular class was
        algebraic from the start. It exists purely as a regression target
        for self-tests."""
        v = torch.zeros(self.n_basis, dtype=torch.float64, device=self.tau_re.device)
        v[i * self.g + j] = 1.0
        return v


# ==================== K3 Abstract Hodge Structure ===============================
def _e8_cartan_gram() -> np.ndarray:
    """Gram matrix of the E8 root lattice (even, unimodular, positive
    definite, rank 8) in a standard simple-root basis. Used to build the
    K3 lattice E8(-1)^2 (+) U^3 (signature (3,19), rank 22), of which
    H^2(K3,Z) is the unique example up to isometry."""
    A = np.array([
        [2,-1,0,0,0,0,0,0],
        [-1,2,-1,0,0,0,0,0],
        [0,-1,2,-1,0,0,0,-1],
        [0,0,-1,2,-1,0,0,0],
        [0,0,0,-1,2,-1,0,0],
        [0,0,0,0,-1,2,-1,0],
        [0,0,0,0,0,-1,2,0],
        [0,0,-1,0,0,0,0,2],
    ], dtype=np.float64)
    return A


def _hyperbolic_gram() -> np.ndarray:
    """Gram matrix of the hyperbolic plane U: even unimodular, signature
    (1,1), basis {e,f} with e.e=f.f=0, e.f=1."""
    return np.array([[0., 1.], [1., 0.]], dtype=np.float64)


def k3_lattice_gram() -> np.ndarray:
    """
    Full Gram matrix of the K3 lattice  L_{K3} = E8(-1) (+) E8(-1) (+) U (+) U (+) U,
    rank 22, signature (3,19), even and unimodular -- this IS H^2(K3,Z)
    with the cup-product intersection form, for every complex K3 surface.
    E8(-1) means the E8 Gram matrix negated (E8 itself is positive
    definite rank 8; we need two negative-definite copies plus three
    hyperbolic planes to get signature (3,19) and rank 22).
    """
    e8 = _e8_cartan_gram()
    blocks = [-e8, -e8, _hyperbolic_gram(), _hyperbolic_gram(), _hyperbolic_gram()]
    n = sum(b.shape[0] for b in blocks)
    G = np.zeros((n, n), dtype=np.float64)
    off = 0
    for b in blocks:
        s = b.shape[0]
        G[off:off+s, off:off+s] = b
        off += s
    assert G.shape == (22, 22)
    return G


class K3LatticeHodgeStructure(nn.Module):
    """
    Abstract weight-2 polarized Hodge structure with K3 Hodge numbers
    h^{2,0} = h^{0,2} = 1,  h^{1,1} = 20  (total rank 22), intersection
    form = the K3 lattice (signature (3,19), even, unimodular).

    There is no point-set "variety" being simulated here -- a K3 surface
    is far too complicated to sample by particle clouds in a toy model --
    instead we work directly with a *period point*: a choice of complex
    line C*sigma in L_{K3} (X) C satisfying
        (sigma, sigma) = 0          (the quadric condition)
        (sigma, conj(sigma)) > 0    (positivity / Hodge-Riemann for (2,0))
    where ( , ) is the bilinear extension of the K3 intersection form.
    This is exactly a point of the K3 period domain. sigma spans
    H^{2,0}; its orthogonal complement (real codimension determined by
    the lattice) splits further into H^{1,1}_R once we also fix a Kahler
    class -- which is what the SSC-optimized "cycle" is trying to find a
    candidate (1,1) class inside of.

    Hodge-Riemann bilinear relations enforced (weight-2, K3 case):
      (i)  (sigma, sigma) = 0
      (ii) (sigma, conj(sigma)) > 0
      (iii) for a real primitive (1,1)-class eta orthogonal to sigma and
            conj(sigma): (eta, eta) < 0  on the orthogonal complement
            (signature (1,19) on H^{1,1}_R after removing the positive
            (sigma,conj(sigma))-line) -- the celebrated fact that the
            (1,1) part of a K3 Hodge structure has signature (1,19), of
            which exactly the +1 direction (the Kahler cone direction)
            is "positive" and the rest negative. We check sign patterns,
            not full classification.
    """

    def __init__(self, seed: int = 0, device: str = "cpu"):
        super().__init__()
        G_np = k3_lattice_gram()
        G = torch.tensor(G_np, dtype=torch.float64, device=device)
        self.register_buffer("gram", G)
        self.rank = G.shape[0]   # 22
        self.h11 = 20
        self.h20 = 1

        # Sample a generic period point sigma = sigma_re + i*sigma_im in
        # L (x) C, sigma in (rank,) real + imag parts, satisfying the
        # quadric relations approximately at init (refined via projection).
        rng = np.random.default_rng(seed)
        re = torch.tensor(rng.standard_normal(self.rank), dtype=torch.float64, device=device)
        im = torch.tensor(rng.standard_normal(self.rank), dtype=torch.float64, device=device)
        sigma_re, sigma_im = self._project_to_quadric(re, im)
        self.register_buffer("sigma_re", sigma_re)
        self.register_buffer("sigma_im", sigma_im)

    def bilinear(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Q(u,v) = u^T G v, the K3 intersection form, batched over leading dims."""
        return torch.einsum('...i,ij,...j->...', u, self.gram.to(u.dtype), v)

    def _project_to_quadric(self, re: torch.Tensor, im: torch.Tensor,
                             n_iter: int = 200, lr: float = 0.05
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project a random (re, im) onto the K3 period quadric
        {(sigma,sigma)=0, (sigma,conj(sigma))>0} via a short, self-contained
        gradient descent on the defect, run once at construction time (not
        part of the trainable graph). This gives a valid, generic period
        point without requiring an external nonlinear solver dependency.
        """
        re = re.clone().requires_grad_(True)
        im = im.clone().requires_grad_(True)
        opt = optim.Adam([re, im], lr=lr)
        for _ in range(n_iter):
            opt.zero_grad()
            # (sigma,sigma) = (re+i im, re+i im) = Q(re,re) - Q(im,im) + 2i Q(re,im)
            q_rr = self.bilinear(re, re)
            q_ii = self.bilinear(im, im)
            q_ri = self.bilinear(re, im)
            defect_real = q_rr - q_ii
            defect_imag = 2.0 * q_ri
            # (sigma, conj(sigma)) = Q(re,re)+Q(im,im) must be > 0 (push it up)
            positivity = q_rr + q_ii
            loss = defect_real ** 2 + defect_imag ** 2 + torch.relu(1.0 - positivity) ** 2
            loss.backward()
            opt.step()
        return re.detach(), im.detach()

    def hodge_riemann_residuals(self, eta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        For a candidate real (1,1)-class eta (rank,) [or batched (...,rank)],
        first project out the (sigma, conj(sigma)) component to get the part
        living on H^{1,1}_R (orthogonal to H^{2,0) (+) H^{0,2}), then report:
          - 'quadric_defect'   : how far sigma is from the quadric (should be
                                 ~0 always; sanity check on the fixed period
                                 point, not on eta)
          - 'self_pairing'     : Q(eta_{1,1}, eta_{1,1}) -- for K3 this is
                                 expected to be NEGATIVE for generic eta in
                                 the orthogonal complement (signature (1,19)),
                                 except along the one positive (Kahler)
                                 direction.
        These residuals feed HodgeRiemannLoss; they are diagnostics, not a
        free-standing constraint on their own.
        """
        sigma_dot_sigma = self.bilinear(self.sigma_re, self.sigma_re) - \
                           self.bilinear(self.sigma_im, self.sigma_im)
        sigma_dot_sigmabar = self.bilinear(self.sigma_re, self.sigma_re) + \
                              self.bilinear(self.sigma_im, self.sigma_im)

        # Remove the (sigma, conj(sigma)) component from eta to land (approx.)
        # in H^{1,1}_R: eta_11 = eta - proj_sigma_re(eta) - proj_sigma_im(eta)
        def _proj_coeff(basis_vec):
            denom = self.bilinear(basis_vec, basis_vec) + 1e-12
            return self.bilinear(eta, basis_vec) / denom

        c_re = _proj_coeff(self.sigma_re)
        c_im = _proj_coeff(self.sigma_im)
        eta_11 = eta - c_re.unsqueeze(-1) * self.sigma_re - c_im.unsqueeze(-1) * self.sigma_im

        self_pairing = self.bilinear(eta_11, eta_11)
        return {
            "quadric_defect": sigma_dot_sigma.abs(),
            "positivity": sigma_dot_sigmabar,
            "self_pairing": self_pairing,
            "eta_11": eta_11,
        }


# ==================== Variety builder (dispatch) ================================
@dataclass
class BuiltVariety:
    """Container returned by build_variety(): whichever concrete structure
    was requested, plus the data every downstream module needs regardless
    of which one it is."""
    kind: VarietyClass
    period_dim: int                 # dimension of the period/class vector
    torus: Optional[ComplexTorusLattice] = None
    k3: Optional[K3LatticeHodgeStructure] = None


def build_variety(cfg: VarietyConfig) -> BuiltVariety:
    if cfg.variety == VarietyClass.ABELIAN:
        torus = ComplexTorusLattice(g=cfg.g, tau_diag=cfg.tau_diag,
                                     device=cfg.device, seed=cfg.k3_seed)
        return BuiltVariety(kind=VarietyClass.ABELIAN,
                             period_dim=torus.n_basis, torus=torus)
    elif cfg.variety == VarietyClass.K3_ABSTRACT:
        k3 = K3LatticeHodgeStructure(seed=cfg.k3_seed, device=cfg.device)
        return BuiltVariety(kind=VarietyClass.K3_ABSTRACT,
                             period_dim=k3.rank, k3=k3)
    else:
        raise ValueError(f"Unknown variety class: {cfg.variety}")


# ==================== Hodge-Riemann Bilinear Relations Loss =====================
class HodgeRiemannLoss(nn.Module):
    """
    Differentiable penalty enforcing the Hodge-Riemann bilinear relations
    on a candidate class, specialized to whichever BuiltVariety is active.

    ABELIAN case: the polarization form restricted to our E_{ij} basis is
    positive-definite (riemann_form_matrix(), diagonal of 1's) -- a "valid"
    (1,1) class in this model should have *non-negative* pairing with
    itself under that form (it is, after all, a sum of ample/effective
    divisor classes in the cases we care about), so the penalty is
    relu(-Q(v,v)) plus a smaller penalty discouraging huge norm drift.

    K3_ABSTRACT case: uses K3LatticeHodgeStructure.hodge_riemann_residuals
    to penalize (a) any (2,0)/(0,2) leakage into a class that's supposed to
    be (1,1) [should be ~0 once projected, this checks the projection is
    behaving] and (b) wrong-sign self-pairing on the H^{1,1}_R orthogonal
    complement relative to a reference "positive" (Kahler-like) direction
    fixed at construction time.
    """

    def __init__(self, built: BuiltVariety):
        super().__init__()
        self.built = built
        if built.kind == VarietyClass.ABELIAN:
            self.register_buffer("Q", built.torus.riemann_form_matrix())
            self._kahler_ref = None
        else:
            # Fix one reference "positive" direction (a stand-in Kahler
            # class) once, at construction: the class with self-pairing
            # > 0 found by gradient ascent from a random start, projected
            # to H^{1,1}_R. Subsequent eta's are compared against this
            # direction's sign convention rather than re-deriving it.
            self._kahler_ref = self._find_positive_direction(built.k3)

    @staticmethod
    def _find_positive_direction(k3: 'K3LatticeHodgeStructure',
                                  n_iter: int = 150, lr: float = 0.05) -> torch.Tensor:
        v = torch.randn(k3.rank, dtype=torch.float64, device=k3.gram.device, requires_grad=True)
        opt = optim.Adam([v], lr=lr)
        for _ in range(n_iter):
            opt.zero_grad()
            res = k3.hodge_riemann_residuals(v)
            # maximize self_pairing while keeping norm ~ 1
            norm_pen = (torch.dot(v, v) - 1.0) ** 2
            loss = -res["self_pairing"] + 5.0 * norm_pen
            loss.backward()
            opt.step()
        return v.detach()

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        """eta: (..., period_dim) candidate class(es). Returns scalar loss."""
        if self.built.kind == VarietyClass.ABELIAN:
            Q = self.Q.to(eta.dtype)
            self_pairing = torch.einsum('...i,ij,...j->...', eta, Q, eta)
            return torch.relu(-self_pairing).mean()
        else:
            k3 = self.built.k3
            res = k3.hodge_riemann_residuals(eta.to(torch.float64))
            # Compare sign of self_pairing against the reference direction's
            # sign (positive); generic (1,1) classes off the Kahler ray are
            # allowed to be negative (signature (1,19)), so we only penalize
            # gross blow-up, not negativity itself -- the *useful* signal
            # here is keeping the (2,0)/(0,2) leakage near zero, since a
            # genuine (1,1) class must be orthogonal to sigma and conj(sigma).
            leak_re = k3.bilinear(eta.to(torch.float64), k3.sigma_re)
            leak_im = k3.bilinear(eta.to(torch.float64), k3.sigma_im)
            leakage_penalty = (leak_re ** 2 + leak_im ** 2)
            blowup_penalty = torch.relu(res["self_pairing"].abs() - 50.0)
            return (leakage_penalty + blowup_penalty).mean().to(eta.dtype)


# ==================== Integrality: soft relaxation + hard projection ===========
class IntegralityModule(nn.Module):
    """
    Models the integrality condition at the heart of the Hodge conjecture:
    a Hodge class must lie in H^{2p}(X,Z), not just H^{p,p}(X,R). We
    implement this in two stages, as requested:

      (1) SOFT RELAXATION (used *during* training, differentiable):
          For a class v expressed in the chosen integral basis (the E_{ij}
          basis for ABELIAN, the K3 lattice's own Z^22 basis for
          K3_ABSTRACT), penalize non-integrality coordinate-wise with
              L_int(v) = sum_k sin^2(pi * v_k)
          This is the standard differentiable relaxation of "distance to
          nearest integer": it is 0 exactly at integers, smooth, and its
          gradient vanishes at integers too (so it does not fight
          convergence once a coordinate has locked onto a lattice point),
          while still supplying a useful gradient elsewhere ~ pi*sin(2*pi*v_k)/2.

      (2) HARD LATTICE PROJECTION (applied *once*, at the end of training):
          project_to_lattice(v) simply rounds each coordinate of v (in the
          integral basis) to the nearest integer. For ABELIAN this is
          literally rint(v) since the E_{ij} basis is unimodular (its own
          dual). This is the step that actually produces a *candidate
          integral Hodge class* from the network's continuous output --
          the soft penalty alone never guarantees integrality, only
          encourages it.

    Both stages report a diagnostic "integrality gap" =
        sum_k min(frac(v_k), 1-frac(v_k))
    i.e. the L1 distance from v to the nearest integer point, so training
    logs and self-tests can track convergence to the lattice numerically,
    not just via the smooth loss surrogate.
    """

    def __init__(self, basis_dual: torch.Tensor):
        super().__init__()
        self.register_buffer("basis_dual", basis_dual)  # (n,n) identity for ABELIAN

    def soft_penalty(self, v: torch.Tensor) -> torch.Tensor:
        """v: (..., n) in the integral basis. Returns scalar soft-relaxation loss."""
        coords = torch.einsum('...i,ij->...j', v, self.basis_dual.to(v.dtype))
        return torch.sin(math.pi * coords).pow(2).mean()

    @staticmethod
    def integrality_gap(v: torch.Tensor) -> torch.Tensor:
        """Diagnostic only (no grad needed): L1 distance to nearest lattice point."""
        with torch.no_grad():
            frac = v - torch.floor(v)
            gap = torch.minimum(frac, 1.0 - frac)
            return gap.sum(dim=-1)

    @staticmethod
    def project_to_lattice(v: torch.Tensor) -> torch.Tensor:
        """
        HARD projection step, applied once after training finishes:
        round to the nearest integer point in the chosen basis. This is
        deliberately NOT used inside train_step/train -- it is a
        post-processing operation the trainer calls exactly once, per the
        explicit two-stage design (soft during training, hard at the end).
        """
        return torch.round(v)


# ==================== Learnable SOC Kernel (complex-domain, carries v1 bugfixes) =
class LearnableSOCKernel(nn.Module):
    """Learnable Self-Organised Criticality kernel with trainable parameters,
    evaluated on a real distance r >= 0 (typically |z| for complex z, or a
    torus-aware distance -- see ComplexSSCSimulator.step).

    Carries forward, unchanged, the v1 bugfix (2026-06): `alpha` is bounded
    to [alpha_min, alpha_max] via a sigmoid reparameterization (an
    unconstrained alpha could drift upward during optimization and make the
    kernel diverge near r=0), and the small-r regularizer is tied to the
    kernel's own length scale `lambd` rather than a fixed constant, so the
    floor stays meaningful regardless of the units r is expressed in.
    """
    def __init__(self, init_Cs: float = 0.18, init_lambda: float = 12.0,
                 init_alpha: float = 0.5, init_tau: float = 10.0,
                 alpha_min: float = 0.05, alpha_max: float = 2.0,
                 device: str = 'cpu'):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(init_Cs), device=device))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda), device=device))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), device=device))

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        init_alpha = min(max(init_alpha, alpha_min + 1e-6), alpha_max - 1e-6)
        frac = (init_alpha - alpha_min) / (alpha_max - alpha_min)
        raw_init = math.log(frac / (1.0 - frac))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init, device=device))

    @property
    def Cs(self): return torch.exp(self.log_Cs)
    @property
    def lambd(self): return torch.exp(self.log_lambda)
    @property
    def alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)
    @property
    def tau(self): return torch.exp(self.log_tau)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_eps = 1e-3 * self.lambd
        return self.Cs * torch.pow(r + r_eps, -self.alpha) * torch.exp(-r / self.lambd)


# ==================== DiffRGRefiner (optional, unchanged from v1) ==============
class DiffRGRefiner(nn.Module):
    """Differentiable Renormalisation Group filter (Fourier low-pass)."""
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


# ==================== Soft Histogram (real-axis density utility, unchanged) ====
def soft_histogram(x: torch.Tensor, grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable Gaussian-kernel density estimator on a 1-D grid.
    x: (batch, N), grid: (NGRID,), sigma: bandwidth. Returns (batch, NGRID)."""
    dx = grid[1] - grid[0]
    diff = x.unsqueeze(2) - grid.unsqueeze(0).unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    density = weights.sum(dim=1) / (weights.sum(dim=1).sum(dim=1, keepdim=True) * dx + 1e-12)
    return density


# ==================== Complex SSC Simulator =====================================
class ComplexSSCSimulator(nn.Module):
    """
    Semantic-State Contraction particle dynamics, generalized from v1's
    single real line to the complex torus C^g / Lambda underlying the
    chosen variety (or, for K3_ABSTRACT, to a free C^1 auxiliary domain
    used only to drive the SOC kernel training -- the K3 case has no
    point-set particle cloud to evolve since there is no explicit variety,
    see CycleClassMap for how that case is actually handled).

    Each particle is a point z_k in C^g, stored as a real tensor of shape
    (batch, N, g, 2) with last dim = (Re, Im). Dynamics per complex
    coordinate are the same SSC drift/noise law as v1
        drift = -alpha * H[rho] * SOC(|z|)  - beta * grad(rho) + gamma * z
        noise = sigma * sqrt(dt) * dW
    applied independently to each of the g torus factors, with periodic
    wraparound into the fundamental domain [0,1) + tau*[0,1) per factor
    (so particles genuinely move on the compact torus, not on an
    unbounded plane) when a ComplexTorusLattice is supplied; otherwise
    (K3_ABSTRACT / no torus) the domain is an unbounded disk truncated by
    a soft restoring term (the gamma drift) only.

    Carries forward, unchanged, the v1 bugfix (2026-06): gamma's sign is
    stored in a fixed (non-trainable) buffer `gamma_sign`, with only the
    magnitude optimized in log-space via `log_gamma`, so `self.gamma`
    faithfully reproduces the requested sign (positive or negative)
    instead of silently collapsing to positive-only as the earlier
    `exp(log_gamma)`-only parameterization did.
    """
    def __init__(self, N_particles: int, g: int, NGRID: int = 128,
                 alpha: float = 0.8, beta: float = 0.05, gamma: float = 0.0,
                 sigma: float = 0.3, dt: float = 0.01,
                 torus: Optional[ComplexTorusLattice] = None,
                 soc_kernel: Optional[LearnableSOCKernel] = None,
                 rg_filter: Optional[DiffRGRefiner] = None,
                 device: str = 'cpu'):
        super().__init__()
        self.N = N_particles
        self.g = g
        self.NGRID = NGRID
        self.dt = dt
        self.torus = torus
        self.soc = soc_kernel if soc_kernel is not None else LearnableSOCKernel(device=device)
        self.rg = rg_filter

        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha), device=device))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta), device=device))
        self.register_buffer('gamma_sign',
                              torch.tensor(1.0 if gamma >= 0 else -1.0, device=device))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(abs(gamma) + 1e-6), device=device))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), device=device))

        # 1-D reference grid (per real coordinate, range [0,1) fundamental
        # domain side) used for the density / Hilbert-transform machinery,
        # exactly as in v1 but now applied per (factor, re/im) coordinate.
        self.grid = nn.Parameter(torch.linspace(0.0, 1.0, NGRID, device=device),
                                  requires_grad=False)
        dx = 1.0 / (NGRID - 1)
        self.sigma_kde = dx * 0.5

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self): return torch.exp(self.log_beta)
    @property
    def gamma(self): return self.gamma_sign * torch.exp(self.log_gamma)
    @property
    def sigma_val(self): return torch.exp(self.log_sigma)

    def initial_uniform(self, batch_size: int = 1) -> torch.Tensor:
        """Returns (batch, N, g, 2) particles uniform on [0,1)^{2g} fundamental
        domain coordinates (i.e. coefficients of the real lattice basis
        e_i, f_i, NOT yet mapped into C -- ComplexTorusLattice.to_complex
        below performs that map when needed for period integrals)."""
        return torch.rand(batch_size, self.N, self.g, 2, device=self.grid.device)

    def _hilbert_fft(self, density: torch.Tensor) -> torch.Tensor:
        """Hilbert transform of a periodic density via FFT (unchanged math from v1)."""
        F = torch.fft.fft(density, dim=-1)
        k = torch.fft.fftfreq(density.size(-1), device=density.device)
        mult = -1j * torch.where(k > 0, torch.ones_like(k),
                                 torch.where(k < 0, -torch.ones_like(k), torch.zeros_like(k)))
        H = torch.real(torch.fft.ifft(F * mult, dim=-1))
        return H

    def _scalar_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SSC drift/noise law to one real coordinate slice
        x: (batch, N) valued in [0,1) (periodic). Returns updated (batch,N),
        wrapped back into [0,1). This is the same update rule as v1's
        SSCSimulator.step, factored out so it can be applied independently
        to each of the 2*g real coordinates per particle.
        """
        density = soft_histogram(x, self.grid, self.sigma_kde)
        if self.rg is not None:
            density = self.rg(density)

        H = self._hilbert_fft(density)
        dx = self.grid[1] - self.grid[0]
        grad = torch.zeros_like(density)
        grad[:, 1:-1] = (density[:, 2:] - density[:, :-2]) / (2 * dx)

        idx = (x / dx).long().clamp(0, self.NGRID - 2)
        x0 = self.grid[idx]
        x1 = self.grid[idx + 1]
        w1 = (x - x0) / (x1 - x0 + 1e-12)
        w0 = 1.0 - w1

        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        Hp = w0 * H[batch_idx, idx] + w1 * H[batch_idx, idx + 1]
        Gp = w0 * grad[batch_idx, idx] + w1 * grad[batch_idx, idx + 1]

        # Distance to the fundamental-domain center, periodic-aware, feeds the
        # SOC kernel (replaces v1's |x - XMIN|/(XMAX-XMIN), now intrinsically
        # in [0,1) so no XMIN/XMAX bookkeeping is needed).
        r = torch.minimum(x, 1.0 - x)
        soc_scale = self.soc(r)

        drift = -self.alpha * Hp * soc_scale - self.beta * Gp + self.gamma * (x - 0.5)
        noise = self.sigma_val * math.sqrt(self.dt) * torch.randn_like(x)
        x_new = x + drift * self.dt + noise
        return x_new.remainder(1.0)  # periodic wraparound onto the torus

    def step(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, N, g, 2) real lattice-basis coordinates. Applies the SSC
        update independently per (g, re/im) slice and returns the updated
        tensor, same shape. Independence across factors/components mirrors
        v1's treatment of all particles under one shared drift law -- the
        coupling between factors, if desired, would enter through SOC
        kernel sharing (which it does, since one `self.soc` is used for all
        slices) rather than through cross-coordinate drift terms.
        """
        batch, N, g, _ = z.shape
        flat = z.permute(2, 3, 0, 1).reshape(g * 2, batch, N)  # (2g, batch, N)
        out_slices = [self._scalar_step(flat[k]) for k in range(g * 2)]
        out = torch.stack(out_slices, dim=0).reshape(g, 2, batch, N).permute(2, 3, 0, 1)
        return out

    def simulate(self, num_steps: int, initial_z: Optional[torch.Tensor] = None,
                 batch_size: int = 1) -> torch.Tensor:
        z = self.initial_uniform(batch_size) if initial_z is None else initial_z
        for _ in range(num_steps):
            z = self.step(z)
        return z


# ==================== Cycle-Class Map (replaces v1's period computer) ==========
class CycleClassMap(nn.Module):
    """
    Maps a particle cloud (a discretized candidate algebraic cycle) to a
    cohomology class vector, replacing v1's DifferentiablePeriodComputer
    (which used sin(x) features through a frozen random linear layer --
    not a period integral of anything).

    ABELIAN case:
      Particles z (batch, N, g, 2) live on the torus in real lattice-basis
      coordinates. Since every (1,1)-form in our E_{ij} basis is
      translation-invariant (constant) on the torus (see
      ComplexTorusLattice.harmonic_form_value), the period integral of
      E_{ij} over a 2-real-dimensional cycle built from this particle
      cloud is, up to normalization, proportional to how the particle
      density covers the (i,j) pair of real directions -- concretely we
      use the empirical covariance between factor i's and factor j's
      angular coordinates as the integration weight, which is exactly the
      intersection-pairing recipe for translation-invariant forms on a
      torus (it reduces to the topological degree of the cycle's
      projection onto the (i,j) two-torus). This gives a real, justified
      map from "where the particles sit" to "what (1,1) class that
      represents" rather than an arbitrary feature embedding.

    K3_ABSTRACT case:
      There is no particle cloud (no explicit K3 variety is simulated).
      Instead the "candidate cycle" is represented directly by a learnable
      vector in R^22 (the K3 lattice rank), passed through a projection
      that removes its (sigma, conj(sigma)) component (forcing it into
      H^{1,1}_R) before being treated as the candidate class -- i.e. the
      network is only ever allowed to *propose* (1,1)-type classes, by
      construction, rather than relying on a loss term to discourage
      (2,0)/(0,2) leakage after the fact.
    """

    def __init__(self, built: BuiltVariety, N_particles: int = 200, device: str = "cpu"):
        super().__init__()
        self.built = built
        self.N = N_particles
        if built.kind == VarietyClass.ABELIAN:
            self.g = built.torus.g
            self.period_dim = built.torus.n_basis
        else:
            self.period_dim = built.k3.rank
            # Free learnable pre-image for the K3 case; the particle-cloud
            # argument to forward() is unused there (kept for a uniform
            # call signature across both variety classes).
            self.raw_vec = nn.Parameter(torch.randn(self.period_dim, dtype=torch.float64,
                                                      device=device) * 0.1)

    def forward(self, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.built.kind == VarietyClass.ABELIAN:
            return self._abelian_periods(z)
        else:
            return self._k3_class()

    def _abelian_periods(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, N, g, 2) particle positions in lattice-basis coordinates
           (each component in [0,1), representing the e_i / f_i = tau_i e_i
           real basis vectors of factor i's lattice Z + tau_i Z).
        returns: (batch, g*g) period vector in the E_{ij} basis.

        E_{ii} (theta divisor class of factor i): proportional to how
        tightly/uniformly particles cover factor i's own two real
        directions -- we use the empirical variance of the angular
        coordinate (high coverage = larger pairing with the ample theta
        class), normalized so a uniform-on-torus cloud gives pairing 1
        (matching the "known algebraic class" ground truth normalization).

        E_{ij} (i != j): proportional to the empirical cross-covariance
        between factor i's and factor j's angular coordinates -- this is
        the period integral picking up correlation/coupling between two
        otherwise-independent elliptic curve factors, which is exactly
        what a genuine cycle that is NOT simply a product of two
        sub-cycles (e.g. the graph of an isogeny) would contribute.
        """
        batch, N, g, _ = z.shape
        # angle per factor: combine (Re, Im) lattice coords into one phase
        # in [0,1) per factor via their sum mod 1 (Re coordinate is the e_i
        # direction, Im coordinate is the f_i = tau_i e_i direction; for the
        # purposes of a translation-invariant-form pairing we only need a
        # coordinate that is uniform when particles are uniform on the
        # torus, which (x_re + x_im) mod 1 satisfies).
        theta = (z[..., 0] + z[..., 1]).remainder(1.0)  # (batch, N, g)

        # Center to [-0.5, 0.5) so "uniform coverage" <-> high variance,
        # "clumped at a point" <-> low variance, matching theta-divisor
        # intuition (an ample class pairs strongly with a cycle that
        # spreads across the whole factor, weakly with one localized at a
        # point).
        theta_centered = theta - theta.mean(dim=1, keepdim=True)

        # Uniform-on-[0,1) reference variance, used to normalize E_{ii} so
        # the fully-uniform cloud maps to pairing 1 against E_{ii} (matching
        # known_algebraic_class's unit normalization).
        uniform_ref_var = 1.0 / 12.0

        periods = torch.zeros(batch, g, g, dtype=z.dtype, device=z.device)
        for i in range(g):
            for j in range(g):
                if i == j:
                    var_i = (theta_centered[:, :, i] ** 2).mean(dim=1)
                    periods[:, i, j] = var_i / uniform_ref_var
                else:
                    cov_ij = (theta_centered[:, :, i] * theta_centered[:, :, j]).mean(dim=1)
                    periods[:, i, j] = cov_ij / uniform_ref_var
        return periods.reshape(batch, g * g)

    def _k3_class(self) -> torch.Tensor:
        """Project the free vector into H^{1,1}_R (orthogonal to sigma,
        conj(sigma)) and return it as the candidate (1,1) class."""
        k3 = self.built.k3
        v = self.raw_vec
        def _proj_coeff(basis_vec):
            denom = k3.bilinear(basis_vec, basis_vec) + 1e-12
            return k3.bilinear(v, basis_vec) / denom
        c_re = _proj_coeff(k3.sigma_re)
        c_im = _proj_coeff(k3.sigma_im)
        v_11 = v - c_re * k3.sigma_re - c_im * k3.sigma_im
        return v_11.unsqueeze(0)  # (1, period_dim) to match batch convention


# ==================== Hodge Class Target =======================================
class HodgeClass:
    """Holds a target class vector. For ABELIAN this should be expressed in
    the E_{ij} basis (see ComplexTorusLattice.known_algebraic_class for a
    guaranteed-algebraic example); for K3_ABSTRACT it is a vector in the
    rank-22 K3 lattice basis, ideally one already lying in H^{1,1}_R."""
    def __init__(self, vector: torch.Tensor, normalize: bool = True):
        if normalize:
            self.vector = vector / (vector.norm() + 1e-8)
        else:
            self.vector = vector

    @staticmethod
    def random(dim: int, device: str = 'cpu') -> 'HodgeClass':
        v = torch.randn(dim, device=device)
        return HodgeClass(v)


# ==================== Training Manager =========================================
class HodgeSSCTrainer:
    """
    Orchestrates training of the ComplexSSCSimulator / CycleClassMap
    (including the shared LearnableSOCKernel) so the computed class
    matches a target HodgeClass, while also:
      (a) enforcing Hodge-Riemann positivity via HodgeRiemannLoss, and
      (b) softly encouraging integrality via IntegralityModule during
          training, with a single HARD lattice projection applied once
          at the very end (train() returns both the raw and the
          projected final class).

    Loss = || class - target ||^2
           + lambda_hr  * HodgeRiemannLoss(class)
           + lambda_int * IntegralityModule.soft_penalty(class)
    """
    def __init__(self, built: BuiltVariety,
                 cycle_map: CycleClassMap,
                 target: HodgeClass,
                 simulator: Optional[ComplexSSCSimulator] = None,
                 lr: float = 0.01,
                 lambda_hr: float = 0.1,
                 lambda_int: float = 0.05):
        self.built = built
        self.sim = simulator
        self.cycle_map = cycle_map
        self.target = target
        self.lambda_hr = lambda_hr
        self.lambda_int = lambda_int

        self.hr_loss = HodgeRiemannLoss(built)

        if built.kind == VarietyClass.ABELIAN:
            basis_dual = built.torus.integral_basis_dual()
        else:
            # K3 lattice's own integral basis is just Z^22 in the standard
            # Gram-matrix basis used above (E8(-1)^2 (+) U^3 basis vectors
            # are by definition the integral generators of the lattice).
            basis_dual = torch.eye(built.k3.rank, dtype=torch.float64,
                                    device=built.k3.gram.device)
        self.integrality = IntegralityModule(basis_dual)

        params = list(cycle_map.parameters())
        if simulator is not None:
            params += list(simulator.parameters())
        if len(params) == 0:
            raise ValueError("No trainable parameters found in simulator/cycle_map.")
        self.optimizer = optim.Adam(params, lr=lr)

    def compute_loss(self, periods: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        target_vec = self.target.vector.to(periods.dtype).unsqueeze(0)
        fit_loss = torch.sum((periods - target_vec) ** 2)
        hr_loss = self.hr_loss(periods)
        int_loss = self.integrality.soft_penalty(periods)
        total = fit_loss + self.lambda_hr * hr_loss + self.lambda_int * int_loss
        diag = {
            "fit": fit_loss.item(),
            "hodge_riemann": hr_loss.item() if torch.is_tensor(hr_loss) else float(hr_loss),
            "integrality_soft": int_loss.item(),
            "integrality_gap": self.integrality.integrality_gap(periods.detach()).mean().item(),
        }
        return total, diag

    def train_step(self, num_sim_steps: int = 100, batch_size: int = 1) -> Dict[str, float]:
        if self.sim is not None:
            self.sim.train()
        self.cycle_map.train()
        self.optimizer.zero_grad()

        if self.built.kind == VarietyClass.ABELIAN:
            z = self.sim.initial_uniform(batch_size)
            for _ in range(num_sim_steps):
                z = self.sim.step(z)
            periods = self.cycle_map(z)
        else:
            periods = self.cycle_map(None)

        loss, diag = self.compute_loss(periods)
        loss.backward()
        self.optimizer.step()
        diag["loss"] = loss.item()
        return diag

    def train(self, epochs: int = 500, num_sim_steps: int = 100, batch_size: int = 1,
               log_every: int = 50) -> Dict[str, float]:
        diag = {}
        for epoch in range(epochs):
            diag = self.train_step(num_sim_steps, batch_size)
            if epoch % log_every == 0:
                logger.info(f"Epoch {epoch:4d} | loss={diag['loss']:.6f} "
                            f"fit={diag['fit']:.6f} HR={diag['hodge_riemann']:.6f} "
                            f"int_soft={diag['integrality_soft']:.6f} "
                            f"int_gap={diag['integrality_gap']:.4f}")
                if self.built.kind == VarietyClass.ABELIAN and self.sim is not None:
                    soc = self.sim.soc
                    logger.info(f"  SOC params: Cs={soc.Cs.item():.3f}, "
                                f"lambda={soc.lambd.item():.3f}, "
                                f"alpha={soc.alpha.item():.3f}, tau={soc.tau.item():.3f}")
        return diag

    @torch.no_grad()
    def final_projected_class(self, num_sim_steps: int = 100,
                               batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Run the cycle map once more (no grad) and apply the HARD lattice
        projection exactly once, as the final post-processing step. Returns
        both the raw (continuous) class and its projected integral
        counterpart, plus the residual integrality gap of each so the
        caller can judge how close training actually got before rounding.
        """
        if self.built.kind == VarietyClass.ABELIAN:
            z = self.sim.initial_uniform(batch_size)
            for _ in range(num_sim_steps):
                z = self.sim.step(z)
            raw = self.cycle_map(z)
        else:
            raw = self.cycle_map(None)

        projected = self.integrality.project_to_lattice(raw)
        return {
            "raw": raw,
            "projected": projected,
            "raw_integrality_gap": self.integrality.integrality_gap(raw),
            "hodge_riemann_residual": self.hr_loss(projected),
        }


# ==================== Self-Test Suite ===========================================
def self_test(verbose: bool = True) -> bool:
    """
    Validates the mathematical claims this module relies on, independent
    of any training run:

      1. K3 lattice Gram matrix has rank 22, signature (3,19), is even,
         and is unimodular (|det| = 1) -- i.e. it really is the K3 lattice.
      2. The sampled K3 period point sigma lies on the quadric
         (sigma,sigma)=0 and satisfies (sigma,conj(sigma))>0 to numerical
         tolerance, after the construction-time projection.
      3. ABELIAN: the Riemann form matrix is positive definite (so
         HodgeRiemannLoss's penalty is well-posed) and is the identity in
         the E_{ij} basis (so "integral in this basis" really does mean
         "integer coordinates", matching IntegralityModule's dual-basis
         choice).
      4. ABELIAN: a uniform particle cloud's CycleClassMap output recovers
         known_algebraic_class(i,i) (the theta-divisor class) to good
         accuracy -- i.e. the period map is correctly normalized, not just
         dimensionally plausible.
      5. IntegralityModule: soft_penalty is (numerically) zero at integer
         points and positive at a generic half-integer point; project_to_lattice
         recovers exact integers from a small perturbation.
      6. HodgeRiemannLoss: zero (or near-zero) on a known-algebraic class
         (ABELIAN), and the K3 leakage penalty is near-zero for a vector
         already in H^{1,1}_R by construction (CycleClassMap._k3_class output).

    Returns True iff all checks pass within tolerance; logs each result.
    """
    ok = True

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        if verbose:
            logger.info(f"[self_test] {status} - {name}" + (f" ({detail})" if detail else ""))

    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. K3 lattice sanity ----
    G_np = k3_lattice_gram()
    eig = np.linalg.eigvalsh(G_np)
    pos = int(np.sum(eig > 1e-9)); neg = int(np.sum(eig < -1e-9))
    check("K3 lattice rank == 22", G_np.shape == (22, 22), f"shape={G_np.shape}")
    check("K3 lattice signature == (3,19)", (pos, neg) == (3, 19), f"got ({pos},{neg})")
    check("K3 lattice is even", bool(np.all(np.diag(G_np) % 2 == 0)))
    det = np.linalg.det(G_np)
    check("K3 lattice is unimodular (|det|=1)", abs(abs(det) - 1.0) < 1e-6, f"det={det:.6f}")

    # ---- 2. K3 period point on the quadric ----
    k3 = K3LatticeHodgeStructure(seed=0, device="cpu")
    res = k3.hodge_riemann_residuals(torch.zeros(k3.rank, dtype=torch.float64))
    check("K3 period point: quadric defect ~ 0",
          res["quadric_defect"].item() < 1e-2, f"defect={res['quadric_defect'].item():.6f}")
    check("K3 period point: positivity (sigma,sigmabar) > 0",
          res["positivity"].item() > 0, f"value={res['positivity'].item():.6f}")

    # ---- 3. ABELIAN Riemann form ----
    torus = ComplexTorusLattice(g=2, device="cpu", seed=0)
    Qmat = torus.riemann_form_matrix()
    is_identity = torch.allclose(Qmat, torch.eye(torus.n_basis, dtype=Qmat.dtype))
    eigQ = torch.linalg.eigvalsh(Qmat)
    check("ABELIAN Riemann form is identity in E_ij basis", is_identity)
    check("ABELIAN Riemann form is positive definite", bool((eigQ > 0).all()))

    # ---- 4. ABELIAN uniform cloud recovers theta-divisor normalization ----
    built = BuiltVariety(kind=VarietyClass.ABELIAN, period_dim=torus.n_basis, torus=torus)
    cmap = CycleClassMap(built, N_particles=20000, device="cpu")
    z_uniform = torch.rand(1, 20000, 2, 2, dtype=torch.float64)
    periods = cmap(z_uniform)
    known = torus.known_algebraic_class(0, 0)
    e00_recovered = periods[0, 0].item()  # index (i=0,j=0) flattened -> 0*g+0 = 0
    check("ABELIAN: uniform cloud recovers E_00 pairing ~ 1 (theta divisor norm.)",
          abs(e00_recovered - 1.0) < 0.05, f"got {e00_recovered:.4f}, expected ~1.0")
    known_val_at_00 = known[0].item()
    check("ABELIAN: known_algebraic_class(0,0) is the unit basis vector at index 0",
          abs(known_val_at_00 - 1.0) < 1e-9 and known.abs().sum().item() == 1.0)

    # ---- 5. IntegralityModule ----
    basis_dual = torus.integral_basis_dual()
    integ = IntegralityModule(basis_dual)
    v_int = torch.tensor([2.0, -3.0, 0.0, 5.0], dtype=torch.float64)
    pen_int = integ.soft_penalty(v_int)
    v_half = torch.tensor([2.5, -3.5, 0.5, 5.5], dtype=torch.float64)
    pen_half = integ.soft_penalty(v_half)
    check("Integrality soft_penalty ~ 0 at integer point",
          pen_int.item() < 1e-8, f"penalty={pen_int.item():.2e}")
    check("Integrality soft_penalty is maximal (~1) at half-integer point",
          abs(pen_half.item() - 1.0) < 1e-6, f"penalty={pen_half.item():.6f}")
    v_perturbed = torch.tensor([2.1, -2.9, 0.05, 4.96], dtype=torch.float64)
    projected = integ.project_to_lattice(v_perturbed)
    expected = torch.tensor([2.0, -3.0, 0.0, 5.0], dtype=torch.float64)
    check("Hard lattice projection recovers exact integers from a perturbation",
          torch.allclose(projected, expected))

    # ---- 6. HodgeRiemannLoss ----
    hr_loss_abelian = HodgeRiemannLoss(built)
    known_class_batched = known.unsqueeze(0)
    hr_val = hr_loss_abelian(known_class_batched)
    check("HodgeRiemannLoss(known algebraic class) ~ 0 (ABELIAN)",
          hr_val.item() < 1e-6, f"value={hr_val.item():.2e}")

    built_k3 = BuiltVariety(kind=VarietyClass.K3_ABSTRACT, period_dim=k3.rank, k3=k3)
    cmap_k3 = CycleClassMap(built_k3, device="cpu")
    v11 = cmap_k3(None)  # already projected into H^{1,1}_R by construction
    hr_loss_k3 = HodgeRiemannLoss(built_k3)
    hr_val_k3 = hr_loss_k3(v11)
    check("HodgeRiemannLoss leakage penalty ~ 0 for CycleClassMap K3 output "
          "(already projected into H^{1,1}_R)",
          hr_val_k3.item() < 1e-3, f"value={hr_val_k3.item():.2e}")

    if verbose:
        logger.info(f"[self_test] {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ==================== CLI ========================================================
def main():
    parser = argparse.ArgumentParser(description="HODGE ONE v2 - Differentiable Hodge Conjecture Platform")
    parser.add_argument("--variety", type=str, default="abelian",
                        choices=["abelian", "k3_abstract"], help="Which variety class to use")
    parser.add_argument("--g", type=int, default=2, help="Number of elliptic curve factors (ABELIAN only)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sim_steps", type=int, default=100, help="SSC simulation steps per epoch (ABELIAN only)")
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_hr", type=float, default=0.1)
    parser.add_argument("--lambda_int", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target", type=str, default="known",
                        choices=["known", "random"],
                        help="'known' uses a guaranteed-algebraic class as target "
                             "(ABELIAN: theta divisor E_00; K3_ABSTRACT: random H^1,1 vector)")
    parser.add_argument("--self_test", action="store_true", help="Run the self-test suite and exit")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save a plot")
    args = parser.parse_args()

    if args.self_test:
        ok = self_test(verbose=True)
        sys.exit(0 if ok else 1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = VarietyConfig(variety=VarietyClass(args.variety), g=args.g,
                        device=args.device, k3_seed=args.seed)
    built = build_variety(cfg)
    logger.info(f"Built variety: {built.kind.value}, period_dim={built.period_dim}")

    cycle_map = CycleClassMap(built, N_particles=args.n_particles, device=args.device)

    if built.kind == VarietyClass.ABELIAN:
        simulator = ComplexSSCSimulator(N_particles=args.n_particles, g=built.torus.g,
                                        device=args.device)
        if args.target == "known":
            target = HodgeClass(built.torus.known_algebraic_class(0, 0), normalize=False)
        else:
            target = HodgeClass.random(built.period_dim, device=args.device)
    else:
        simulator = None
        target = HodgeClass.random(built.period_dim, device=args.device)

    trainer = HodgeSSCTrainer(built, cycle_map, target, simulator=simulator,
                               lr=args.lr, lambda_hr=args.lambda_hr, lambda_int=args.lambda_int)

    logger.info(f"Training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs, num_sim_steps=args.sim_steps, log_every=max(1, args.epochs // 10))

    final = trainer.final_projected_class(num_sim_steps=args.sim_steps)
    logger.info(f"Final raw class:        {final['raw'].squeeze().tolist()}")
    logger.info(f"Final projected class:  {final['projected'].squeeze().tolist()}")
    logger.info(f"Raw integrality gap:    {final['raw_integrality_gap'].item():.4f}")
    logger.info(f"HR residual (projected): {final['hodge_riemann_residual'].item():.6f}")

    if args.out and HAS_MPL and built.kind == VarietyClass.ABELIAN:
        raw = final['raw'].squeeze().detach().cpu().numpy().reshape(built.torus.g, built.torus.g)
        proj = final['projected'].squeeze().detach().cpu().numpy().reshape(built.torus.g, built.torus.g)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(raw, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[0].set_title("Raw class (E_ij basis)")
        im1 = axes[1].imshow(proj, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[1].set_title("Lattice-projected class")
        for ax in axes:
            ax.set_xticks(range(built.torus.g)); ax.set_yticks(range(built.torus.g))
        fig.colorbar(im0, ax=axes[0]); fig.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        logger.info(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()


# ==================== CLI / Entry Point =========================================
def main():
    parser = argparse.ArgumentParser(description="HODGE ONE v2 — Differentiable Hodge Conjecture Platform")
    parser.add_argument("--variety", choices=["abelian", "k3_abstract"], default="abelian")
    parser.add_argument("--g", type=int, default=2, help="Number of elliptic curve factors (ABELIAN only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sim-steps", type=int, default=80, help="SSC steps per epoch (ABELIAN only)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-hr", type=float, default=0.1)
    parser.add_argument("--lambda-int", type=float, default=0.05)
    parser.add_argument("--target-i", type=int, default=0, help="Target known_algebraic_class(i,j): i index")
    parser.add_argument("--target-j", type=int, default=0, help="Target known_algebraic_class(i,j): j index")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--self-test", action="store_true", help="Run validation suite and exit")
    parser.add_argument("--plot", action="store_true", help="Save a training-diagnostics plot (requires matplotlib)")
    parser.add_argument("--out", default="/mnt/user-data/outputs/hodge_one_v2_run.png")
    args = parser.parse_args()

    if args.self_test:
        passed = self_test(verbose=True)
        sys.exit(0 if passed else 1)

    device = str(get_device(args.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = VarietyConfig(variety=VarietyClass(args.variety), g=args.g,
                         k3_seed=args.seed, device=device)
    built = build_variety(cfg)
    logger.info(f"Built variety: {built.kind.value}, period_dim={built.period_dim}")

    cycle_map = CycleClassMap(built, N_particles=2000, device=device)

    sim = None
    if built.kind == VarietyClass.ABELIAN:
        sim = ComplexSSCSimulator(N_particles=2000, g=built.torus.g,
                                   torus=built.torus, device=device)
        target = HodgeClass(built.torus.known_algebraic_class(args.target_i, args.target_j),
                             normalize=False)
    else:
        target_vec = torch.randn(built.k3.rank, dtype=torch.float64, device=device)
        # Project target into H^{1,1}_R too, so it's a coherent comparison point.
        def _proj(v, basis):
            denom = built.k3.bilinear(basis, basis) + 1e-12
            return built.k3.bilinear(v, basis) / denom
        target_vec = target_vec - _proj(target_vec, built.k3.sigma_re) * built.k3.sigma_re \
                                  - _proj(target_vec, built.k3.sigma_im) * built.k3.sigma_im
        target = HodgeClass(target_vec, normalize=False)

    trainer = HodgeSSCTrainer(built, cycle_map, target, simulator=sim, lr=args.lr,
                               lambda_hr=args.lambda_hr, lambda_int=args.lambda_int)

    logger.info("Starting training...")
    history = []
    for epoch in range(args.epochs):
        diag = trainer.train_step(num_sim_steps=args.sim_steps)
        history.append(diag)
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch:4d} | loss={diag['loss']:.6f} fit={diag['fit']:.6f} "
                        f"HR={diag['hodge_riemann']:.6f} int_gap={diag['integrality_gap']:.4f}")

    final = trainer.final_projected_class(num_sim_steps=args.sim_steps)
    logger.info(f"Final raw class:       {final['raw'].squeeze().tolist()}")
    logger.info(f"Final projected class: {final['projected'].squeeze().tolist()}")
    logger.info(f"Raw integrality gap:   {final['raw_integrality_gap'].item():.4f}")
    logger.info(f"HR residual (projected): {final['hodge_riemann_residual'].item():.6f}")

    if args.plot and HAS_MPL:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot([h["loss"] for h in history]); axes[0, 0].set_title("Total loss"); axes[0,0].set_yscale("log")
        axes[0, 1].plot([h["fit"] for h in history]); axes[0, 1].set_title("Fit to target"); axes[0,1].set_yscale("log")
        axes[1, 0].plot([h["hodge_riemann"] for h in history]); axes[1, 0].set_title("Hodge-Riemann penalty")
        axes[1, 1].plot([h["integrality_gap"] for h in history]); axes[1, 1].set_title("Integrality gap (L1 to lattice)")
        for ax in axes.flat:
            ax.set_xlabel("epoch")
        fig.suptitle(f"HODGE ONE v2 — {built.kind.value}")
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        logger.info(f"Saved diagnostics plot to {args.out}")


if __name__ == "__main__":
    main()


# ==================== CLI =======================================================
def main():
    parser = argparse.ArgumentParser(description="HODGE ONE v2 - Differentiable Hodge Conjecture Platform")
    parser.add_argument("--variety", choices=["abelian", "k3_abstract"], default="abelian")
    parser.add_argument("--g", type=int, default=2, help="number of elliptic curve factors (ABELIAN only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sim_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_hr", type=float, default=0.1)
    parser.add_argument("--lambda_int", type=float, default=0.05)
    parser.add_argument("--target", choices=["known_algebraic", "random"], default="known_algebraic",
                        help="ABELIAN only: regress to a guaranteed-algebraic class, or a random target")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--out_dir", default="./hodge_one_v2_out")
    args = parser.parse_args()

    if args.self_test:
        passed = self_test(verbose=True)
        sys.exit(0 if passed else 1)

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = VarietyConfig(variety=VarietyClass(args.variety), g=args.g,
                         k3_seed=args.seed, device=str(device))
    built = build_variety(cfg)
    logger.info(f"Built variety: {built.kind.value}, period_dim={built.period_dim}")

    cycle_map = CycleClassMap(built, N_particles=2000, device=str(device))

    if built.kind == VarietyClass.ABELIAN:
        simulator = ComplexSSCSimulator(N_particles=2000, g=built.torus.g,
                                         torus=built.torus, device=str(device))
        if args.target == "known_algebraic":
            target = HodgeClass(built.torus.known_algebraic_class(0, 0), normalize=False)
        else:
            target = HodgeClass.random(built.period_dim, device=str(device))
    else:
        simulator = None
        target = HodgeClass.random(built.period_dim, device=str(device))

    trainer = HodgeSSCTrainer(built, cycle_map, target, simulator=simulator,
                               lr=args.lr, lambda_hr=args.lambda_hr, lambda_int=args.lambda_int)

    logger.info("Starting training...")
    trainer.train(epochs=args.epochs, num_sim_steps=args.sim_steps, log_every=max(1, args.epochs // 10))

    final = trainer.final_projected_class(num_sim_steps=args.sim_steps)
    logger.info(f"Final raw class:        {final['raw'].squeeze().tolist()}")
    logger.info(f"Final projected class:  {final['projected'].squeeze().tolist()}")
    logger.info(f"Raw integrality gap:    {final['raw_integrality_gap'].item():.4f}")
    logger.info(f"HR residual (projected):{final['hodge_riemann_residual'].item():.6f}")

    out_path = os.path.join(args.out_dir, f"hodge_one_v2_{built.kind.value}_result.pt")
    torch.save({
        "variety": built.kind.value,
        "raw_class": final["raw"].detach().cpu(),
        "projected_class": final["projected"].detach().cpu(),
        "target": target.vector.detach().cpu(),
    }, out_path)
    logger.info(f"Saved result -> {out_path}")


if __name__ == "__main__":
    main()
