import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Import the foundational components from the user's module
import hodge_one

# =============================================================================
# 1. Configuration Dataclass
# =============================================================================
@dataclass
class SGNOHodgeConfig:
    node_in_dim: int = 2         # Particle features: [Position, Local Spacing]
    period_dim: int = 5          # Dimension of the target Hodge class vector
    hidden_dim: int = 128
    num_layers: int = 5
    dropout: float = 0.1

# =============================================================================
# 2. Structural FiLM-Modulated Convolution Block
# =============================================================================
class HodgeFiLMBlock(nn.Module):
    """
    1D Convolutional Neural Operator block that modulates particle evolution
    based on the SOC structural field (sigma) and the target Hodge class.
    """
    def __init__(self, dim: int, context_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size=3, padding=1, padding_mode='circular')
        )
        # FiLM Modulators: Mapping (sigma + Hodge Target) -> gamma, beta
        self.film_gamma = nn.Linear(context_dim, dim)
        self.film_beta  = nn.Linear(context_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Dim, N_particles)
        gamma = self.film_gamma(context).unsqueeze(-1)  # (Batch, Dim, 1)
        beta  = self.film_beta(context).unsqueeze(-1)   # (Batch, Dim, 1)
        
        # Apply Structural Modulation
        modulated_x = (gamma * x) + beta
        out = self.conv(modulated_x)
        
        # Residual connection + LayerNorm
        out = out.permute(0, 2, 1)
        x_res = x.permute(0, 2, 1)
        return self.norm(x_res + out).permute(0, 2, 1)

# =============================================================================
# 3. Main AI Surrogate: StructuralGNOHodge
# =============================================================================
class StructuralGNOHodge(nn.Module):
    """
    The AI Surrogate that learns the topological flow of particles to form
    an algebraic cycle corresponding to a specific Hodge class.
    """
    def __init__(self, cfg: SGNOHodgeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        
        # Context consists of 1 scalar (sigma) + the target Hodge period vector
        context_dim = 1 + cfg.period_dim
        
        self.node_embed = nn.Sequential(
            nn.Linear(cfg.node_in_dim, d),
            nn.LayerNorm(d)
        )
        
        self.layers = nn.ModuleList([
            HodgeFiLMBlock(d, context_dim, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        
        # Output head predicts the topological drift (displacement) of particles
        self.drift_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, x_pos: torch.Tensor, target_hodge: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x_pos: (Batch, N) Current particle positions on the manifold
        target_hodge: (Batch, period_dim) The target Hodge class vector
        sigma: (Batch, 1) Current SOC stress/criticality metric
        """
        # Sort positions to compute topological spacings (1D approximation of cycle geometry)
        sorted_x, _ = torch.sort(x_pos, dim=1)
        spacings = torch.cat([
            torch.zeros(x_pos.size(0), 1, device=x_pos.device),
            sorted_x[:, 1:] - sorted_x[:, :-1]
        ], dim=1)
        
        # Node features: (Batch, N, 2)
        nodes = torch.stack([sorted_x, spacings], dim=-1)
        h = self.node_embed(nodes).permute(0, 2, 1)  # (Batch, d, N)
        
        # Create Structural Context for FiLM
        context = torch.cat([sigma, target_hodge], dim=-1)  # (Batch, 1 + period_dim)
        
        for layer in self.layers:
            h = layer(h, context)
            
        h = h.permute(0, 2, 1)  # (Batch, N, d)
        topological_drift = self.drift_head(h).squeeze(-1)  # (Batch, N)
        
        return sorted_x + topological_drift

# =============================================================================
# 4. Integrated Trainer for HODGE ONE
# =============================================================================
class UnifiedHodgeOperatorTrainer:
    """
    Orchestrates the training of the SGNO model, forcing it to learn how to
    arrange particles such that their differentiable period map evaluates
    exactly to the target Hodge class.
    """
    def __init__(self, 
                 operator_model: StructuralGNOHodge, 
                 period_computer: hodge_one.DifferentiablePeriodComputer,
                 lr: float = 1e-3):
        self.model = operator_model
        self.period_computer = period_computer
        
        # We train both the operator model and optionally fine-tune the period mapping
        params = list(self.model.parameters()) + list(self.period_computer.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def train_step(self, x_init: torch.Tensor, target_class: hodge_one.HodgeClass, soc_kernel: hodge_one.LearnableSOCKernel):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = x_init.size(0)
        
        # Calculate dynamic sigma from the SOC kernel based on initial dispersion
        # Normalizing dispersion to act as an input to the SOC kernel
        r_dispersion = torch.std(x_init, dim=1, keepdim=True) 
        sigma = soc_kernel(r_dispersion) # (Batch, 1)
        
        target_vector = target_class.vector.unsqueeze(0).repeat(batch_size, 1)
        
        # The AI surrogate predicts the final optimal algebraic cycle arrangement in ONE shot
        x_optimal = self.model(x_init, target_vector, sigma)
        
        # Evaluate the periods using the differentiable surrogate
        computed_periods = self.period_computer(x_optimal)
        
        # Loss: Mean Squared Error between generated periods and the Target Hodge Class
        loss = F.mse_loss(computed_periods, target_vector)
        
        # Add a topological regularization term to prevent particles from collapsing into a single point
        spacing_regularization = 0.1 * torch.mean(1.0 / (torch.diff(torch.sort(x_optimal, dim=1)[0], dim=1) + 1e-3))
        total_loss = loss + spacing_regularization

        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
