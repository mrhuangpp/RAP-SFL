#!/usr/bin/env python3
"""
RAP-SFL components: SAC controller and Split-Point Adaptive Perturbation Head (SPAH)
Reinforcement Learning based Adaptive Differential Privacy Forward Pass Framework
"""

import math
import random
import copy
from collections import deque
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ---------------------------------------------------------------------------- #
# 1. RL Core Components
# ---------------------------------------------------------------------------- #

class SharedStateEncoder(nn.Module):
    """Encodes flat state vector into compact embedding for SAC"""
    def __init__(self, input_dim: int, hidden: int = 128, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer for off-policy RL (SAC)"""
    def __init__(self, max_size, state_dim, action_dim, device):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward     = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.done       = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, s, a, r, s2, done):
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.reward[self.ptr, 0]  = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr, 0]    = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.state[idx],      device=self.device),
            torch.tensor(self.action[idx],     device=self.device),
            torch.tensor(self.reward[idx],     device=self.device),
            torch.tensor(self.next_state[idx], device=self.device),
            torch.tensor(self.done[idx],       device=self.device),
        )


class SACActor(nn.Module):
    """Gaussian Policy Network for SAC"""
    def __init__(self, state_dim, action_dim, hidden=128,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, s):
        h = self.trunk(s)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min,
                                             self.log_std_max)
        std = log_std.exp()
        return mu, std

    def sample(self, s):
        mu, std = self(s)
        dist    = Normal(mu, std)
        x_t     = dist.rsample()               # reparameterized
        logp    = dist.log_prob(x_t).sum(-1, keepdim=True)
        return x_t, logp


class SACCritic(nn.Module):
    """Double Q-Network for SAC"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class RunningMeanStd:
    """Online estimation of mean and variance using Welford's algorithm for SAC state normalization"""
    def __init__(self, size, eps=1e-4, device="cpu"):
        self.device = device
        self.mean   = torch.zeros(size, device=device)
        self.var    = torch.ones(size, device=device)
        self.count  = eps

    def update(self, x: torch.Tensor):
        x = x.view(-1, self.mean.numel()).to(self.device)
        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


# ---------------------------------------------------------------------------- #
# 2. State Space Construction
# ---------------------------------------------------------------------------- #

def compute_embedding_norm_stats(embeddings: torch.Tensor) -> dict:
    """
    Compute norm statistics of the embedding matrix
    
    Args:
        embeddings: Embedding tensor of shape (batch_size, seq_len, hidden_dim)
        
    Returns:
        Dictionary containing norm statistics
    """
    # Compute Frobenius norm for each sample (before normalization)
    # embeddings: [B, L, D] -> flatten to [B, L*D]
    batch_size = embeddings.shape[0]
    flat_embeds = embeddings.reshape(batch_size, -1)  # Use reshape instead of view
    norms = torch.norm(flat_embeds, p='fro', dim=1)  # [B]
    
    # Compute statistics
    norms_cpu = norms.detach().cpu()
    q25 = float(torch.quantile(norms_cpu, 0.25))
    q50 = float(torch.quantile(norms_cpu, 0.50))
    q75 = float(torch.quantile(norms_cpu, 0.75))
    mean = float(norms_cpu.mean())
    var = float(norms_cpu.var(unbiased=False))
    
    # Compute skewness and kurtosis
    centered = norms_cpu - mean
    std = math.sqrt(var + 1e-8)
    skew = float((centered ** 3).mean() / (std ** 3 + 1e-8))
    kurt = float((centered ** 4).mean() / (std ** 4 + 1e-8))
    
    return {
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'mean': mean,
        'var': var,
        'skew': skew,
        'kurt': kurt,
        'norms': norms  # Keep for subsequent calculations
    }


def compute_attention_sensitivity(
    attention_outputs: Tuple[torch.Tensor],
    layer_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Attention-based sensitivity mask.
    Args:
        attention_outputs: Tuple of [Batch, Num_Heads, Seq_Len, Seq_Len] (softmaxed probabilities)
        layer_weights: [Num_Layers], RL determines which layers' attention to focus on
    Returns:
        sensitivity_mask: [Batch, Seq_Len, 1] - Larger value means less important (can add more noise)
    """
    if not attention_outputs:
        return None
        
    # attention_outputs[0] shape: [B, H, N, N]
    batch_size, num_heads, seq_len, _ = attention_outputs[0].shape
    device = attention_outputs[0].device
    
    # 1. Weighted fusion of Attention Maps from layers
    # attention_outputs is a tuple, length equals number of layers
    # We only take [CLS] token (index 0) attention to other tokens
    # Or take average attention. Here we take [CLS] attention as suggested.
    fused_attention = torch.zeros(batch_size, seq_len, device=device)
    
    # Ensure layer_weights matches number of layers
    num_layers = len(attention_outputs)
    if len(layer_weights) != num_layers:
        # If not matching, use uniform weights or truncate/pad (Assuming match or using first few layers for simplicity)
        # layer_weights dimensions should be handled before calling
        pass

    for i, layer_attn in enumerate(attention_outputs):
        if i >= len(layer_weights): break
        # layer_attn: [B, H, N, N] -> Take [:, :, 0, :] -> [B, H, N] (CLS token attention to all tokens)
        cls_attention = layer_attn[:, :, 0, :].mean(dim=1)  # Average over Head dimension -> [B, N]
        fused_attention += layer_weights[i] * cls_attention

    # 2. Normalization (Min-Max Normalize per sample)
    # Make the most attended token 1, least attended 0 per sample
    min_val = fused_attention.min(dim=1, keepdim=True)[0]
    max_val = fused_attention.max(dim=1, keepdim=True)[0]
    normalized_attn = (fused_attention - min_val) / (max_val - min_val + 1e-6)

    # 3. Convert to noise mask
    # High Attention (1.0) -> Small Noise Mask (0.0) -> Low Noise (Sensitive, Preserve)
    # Low Attention (0.0) -> Large Noise Mask (1.0) -> High Noise (Insensitive, Perturb)
    # [B, N, 1] for broadcasting
    noise_mask = (1.0 - normalized_attn).unsqueeze(-1)
    
    return noise_mask


def compute_attention_entropy(attention_outputs: tuple) -> float:
    """
    Compute Attention Entropy
    Reflects model focus: Lower entropy means more focused attention (clear semantics); Higher entropy means scattered attention.
    """
    if not attention_outputs:
        return 0.0
    
    # Take last layer attention [Batch, NumHeads, SeqLen, SeqLen]
    last_layer_attn = attention_outputs[-1]
    
    # HF returns attentions already softmaxed
    # We focus on CLS token (index 0) attention distribution to other tokens
    # shape: [Batch, NumHeads, SeqLen]
    cls_attn_probs = last_layer_attn[:, :, 0, :] 
    
    # Avoid log(0)
    probs = cls_attn_probs + 1e-9
    
    # Compute Entropy: H(p) = -sum(p * log(p))
    # Sum over last dimension (SeqLen) to get entropy for each Query (here CLS)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1) # [Batch, NumHeads]
    
    # Return average entropy across all samples, heads, tokens
    return float(entropy.mean().item())


def build_state_vector(
    embedding_norm_stats: dict,
    utility: float,  # Current batch utility (e.g., negative perplexity)
    spent_eps: float,  # Consumed privacy budget
    batch_loss: float,  # Batch loss
    num_blocks: int = 12,  # Number of embedding blocks
    target_epsilon: float = None,  # Target privacy budget (optional)
    prev_loss: float = None,  # Previous loss (for change rate)
    gradient_norm: float = None,  # Gradient norm (sensitivity metric)
    attention_entropy: float = None,  # Attention entropy (semantic focus metric)
    device: torch.device = None,  # Device
) -> torch.Tensor:
    """
    Construct state vector s_t (Enhanced: includes remaining budget ratio, loss change rate, gradient norm, attention entropy)
    
    State vector includes:
    1. Embedding norm stats (q25, q50, q75, mean, var, skew, kurt)
    2. Utility signal
    3. Accumulated privacy budget
    4. Remaining Budget Ratio
    5. Epsilon Ratio
    6. Loss
    7. Loss Change Rate - Reflects if model is stabilizing
    8. Gradient Norm - Semantics-aware sensitivity metric
    9. Attention Entropy - Semantics-aware focus metric
    10. Reserved positions for each block (for future extension)
    """
    state_parts = [
        embedding_norm_stats['q25'],
        embedding_norm_stats['q50'],
        embedding_norm_stats['q75'],
        embedding_norm_stats['mean'],
        embedding_norm_stats['var'],
        embedding_norm_stats['skew'],
        embedding_norm_stats['kurt'],
        utility,
        spent_eps,
    ]
    
    # 1. Compute Remaining Budget Ratio - Let Agent perceive remaining "money"
    if target_epsilon is not None and target_epsilon > 0:
        remaining_eps = max(0.0, target_epsilon - spent_eps)
        remaining_budget_ratio = remaining_eps / target_epsilon  # Remaining budget ratio
        eps_ratio = spent_eps / target_epsilon  # Consumed budget ratio
        state_parts.append(remaining_budget_ratio)
        state_parts.append(eps_ratio)
    else:
        state_parts.append(0.0)  # Placeholder
        state_parts.append(0.0)  # Placeholder
    
    state_parts.append(batch_loss)
    
    # 2. Compute Loss Change Rate - Let Agent perceive if model is stabilizing
    if prev_loss is not None:
        loss_delta = batch_loss - prev_loss  # Loss delta
        state_parts.append(loss_delta)
    else:
        state_parts.append(0.0)  # Initial step has no previous step, set to 0
    
    # 3. Gradient Norm - Semantics-aware sensitivity metric
    # When grad_norm is large (sensitive/hard sample), Agent should reduce Noise Multiplier; otherwise increase
    if gradient_norm is not None:
        state_parts.append(gradient_norm)
    else:
        state_parts.append(0.0)  # Placeholder

    # 4. Attention Entropy - Semantics-aware focus metric (New)
    # Low entropy -> Focused attention -> Clear semantics -> Should reduce noise
    if attention_entropy is not None:
        state_parts.append(attention_entropy)
    else:
        state_parts.append(0.0)
    
    # Reserve positions for each block (init to 0, can add block-level stats later)
    state_parts.extend([0.0] * num_blocks)
    
    return torch.tensor(state_parts, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------- #
# 3. Action Parsing and Execution
# ---------------------------------------------------------------------------- #

def parse_action(
    action_vector: torch.Tensor,
    num_layers: int,
    num_blocks: int,
    progress: float = 0.0,
    eps_ratio: float = 0.0,
    min_noise_bound: float = 0.3,
    ablation_mode: str = "none",  # New: Ablation mode
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Parse action vector a_t, map to: layer weights, normalization constants, noise multiplier, Attention influence factor
    
    Args:
        action_vector: Raw action output from RL Agent [num_layers + num_blocks + 2]
        num_layers: Transformer layers (including Embedding)
        num_blocks: Number of embedding blocks
        progress: Training progress [0, 1]
        eps_ratio: Consumed privacy budget ratio [0, 1]
        min_noise_bound: Dynamic noise lower bound, usually calculated based on global budget
        ablation_mode: Ablation mode, options: "none", "no_alf", "no_sana"
        
    Returns:
        layer_weights: [num_layers] Softmax normalized layer weights
        norm_constants: [num_blocks] Clipping constant for each block
        noise_multiplier: float Noise multiplier
        attn_influence: float Attention influence factor
    """
    # 1. Adaptive Layer Fusion (ALF)
    a_layer = action_vector[:num_layers]

    if ablation_mode == "no_alf":
        # === Ablation ALF ===
        # Force Last-Layer Strategy (Static)
        # Simulates traditional DP-Forward method, only perturbing last layer output
        layer_weights = torch.zeros(num_layers, device=action_vector.device)
        layer_weights[-1] = 1.0
    elif ablation_mode == "first_only":
        # ALF analysis variant: only the embedding layer (index 0)
        layer_weights = torch.zeros(num_layers, device=action_vector.device)
        layer_weights[0] = 1.0
    elif ablation_mode == "middle_only":
        # ALF analysis variant: only a middle Transformer block.
        # For BERT-base num_layers = 13 (embed + 12 blocks), pick index 6 (block 6 / 12).
        layer_weights = torch.zeros(num_layers, device=action_vector.device)
        layer_weights[num_layers // 2] = 1.0
    elif ablation_mode == "uniform":
        # ALF analysis variant: uniform mixture w_l = 1/L
        layer_weights = torch.full(
            (num_layers,), 1.0 / num_layers, device=action_vector.device)
    else:
        # === Normal ALF ===
        layer_weights = F.softmax(a_layer, dim=-1).detach().clone()
    
    # 2. Semantics-Aware Noise Allocation (SANA)
    # Models like GPT-2 have large Norms (usually 20-40). Too aggressive clipping causes signal loss.
    # Expand dynamic range to [10.0, 50.0] to give RL more exploration space
    a_norm_blocks = action_vector[num_layers:num_layers + num_blocks]
    norm_constants = (torch.tanh(a_norm_blocks) + 1.0) * 20.0 + 10.0 # [10.0, 50.0]
    
    if ablation_mode == "no_sana":
        # === Ablation SANA ===
        # Force uniform clipping/noise allocation, simulating a single threshold strategy.
        avg_norm = norm_constants.mean().item()  # Must convert to python scalar
        norm_constants = torch.full_like(norm_constants, avg_norm)
    
    norm_constants = norm_constants.detach().clone()
    
    # 3. Attention Influence Factor (beta)
    # Limit max influence to 1.5 to avoid excessive local noise
    a_attn = action_vector[-2]
    if ablation_mode == "no_sana":
        # Also remove attention-guided semantic noise modulation.
        attn_influence = 0.0
    else:
        attn_influence = float((torch.sigmoid(a_attn) * 1.5).item())

    # 4. Dynamic Noise Constraint
    a_noise = action_vector[-1]
    
    # Strategy: Dynamically tighten or relax based on budget progress
    # GPT-2 is extremely sensitive to noise, must limit RL exploration range
    lower_bound_factor = 0.90 # Default safety factor
    if progress > 0.1:
        lag = progress - eps_ratio
        if lag > 0.05: # Budget not exhausted
            lower_bound_factor = max(0.80, 0.90 - lag * 0.5)  # Weaken dynamic adjustment magnitude
    
    current_min = min_noise_bound * lower_bound_factor
    max_noise = min_noise_bound * 1.3  # Lower upper bound: from 2.5x to 1.3x, limiting aggressive exploration
    
    noise_range = max_noise - current_min
    noise_multiplier = torch.tanh(a_noise) * (noise_range / 2) + (current_min + noise_range / 2)
    
    return layer_weights, norm_constants, float(noise_multiplier.item()), attn_influence


def apply_spah(
    layer_outputs: List[torch.Tensor],  # Outputs of each layer [f_layer_1, ..., f_layer_L]
    layer_weights: torch.Tensor,  # Layer weights [L]
    norm_constants: torch.Tensor,  # Block normalization constants [k]
    noise_multiplier: float,  # Global noise multiplier
    device: torch.device,
    attention_outputs: Tuple[torch.Tensor] = None, # New input: Attention maps
    attn_influence: float = 0.0, # New action: Attention influence factor
) -> torch.Tensor:
    """
    Execute SPAH: Adaptive Layer Fusion + block clipping + adaptive noise + (optional) Semantics-Aware Noise Allocation
    
    Args:
        ...
        attention_outputs: Model attention outputs
        attn_influence: Attention influence level (beta)
    """
    batch_size, seq_len, hidden_dim = layer_outputs[0].shape
    
    # Get device from input tensor
    actual_device = layer_outputs[0].device
    
    # Ensure layer_weights and norm_constants are on correct device
    layer_weights = layer_weights.to(actual_device)
    norm_constants = norm_constants.to(actual_device)
    
    # 1. Adaptive Layering: Weighted Combination
    weighted_embedding = torch.zeros_like(layer_outputs[0])
    for i, layer_out in enumerate(layer_outputs):
        weighted_embedding += layer_weights[i] * layer_out
    
    # 2. [New] Compute Attention Noise Adjustment Mask
    # mask: [B, N, 1], range [0, 1] (0=Important/Low Noise, 1=Unimportant/High Noise)
    attn_noise_mask = None
    if attention_outputs is not None and attn_influence > 0.01:
        attn_noise_mask = compute_attention_sensitivity(attention_outputs, layer_weights)
        if attn_noise_mask is not None:
             attn_noise_mask = attn_noise_mask.to(actual_device)

    # 3. Block Normalization and Noise Injection
    # Split embedding into k blocks
    num_blocks = len(norm_constants)
    block_size = hidden_dim // num_blocks
    remainder = hidden_dim % num_blocks
    
    processed_blocks = [] # Renamed to avoid ambiguity
    start_idx = 0
    
    for i in range(num_blocks):
        # Handle potential size difference of last block
        end_idx = start_idx + block_size + (1 if i < remainder else 0)
        
        # Extract block [B, L, block_dim]
        block = weighted_embedding[:, :, start_idx:end_idx].contiguous()
        
        # Compute Frobenius norm for each sample
        flat_block = block.reshape(batch_size, -1)
        block_norms = torch.norm(flat_block, p='fro', dim=1)  # [B]
        
        # Normalization (Clipping)
        C_i = norm_constants[i]
        clip_factor = (C_i / (block_norms + 1e-6)).clamp(max=1.0)  # [B]
        
        # Apply clipping
        normalized_block = block * clip_factor.reshape(batch_size, 1, 1)
        
        # --- Modified Logic: Semantics-Aware Noise Injection ---
        # Base noise standard deviation
        base_std = noise_multiplier * C_i
        
        # Generate base noise
        # Use randn to generate standard normal noise, scale later
        raw_noise = torch.randn(
            normalized_block.shape,
            device=normalized_block.device,
            dtype=normalized_block.dtype
        )

        # Compute final standard deviation
        if attn_noise_mask is not None:
             # Adjust noise distribution with Attention
             # Final Std = Base Std * (1 + beta * AttentionMask)
             # Explanation: If beta=0, revert to original; If beta>0, amplify noise for unimportant words (mask=1)
             # mask shape [B, N, 1], block shape [B, N, D_block]. Broadcast ok.
             adaptive_std = base_std * (1.0 + attn_influence * attn_noise_mask)
             noise = raw_noise * adaptive_std
        else:
             noise = raw_noise * base_std

        processed_blocks.append(normalized_block + noise)
        
        start_idx = end_idx
    
    # Reassemble blocks
    return torch.cat(processed_blocks, dim=-1)


# ---------------------------------------------------------------------------- #
# 4. Reward Function
# ---------------------------------------------------------------------------- #

def compute_reward(
    delta_utility: float,       # Utility gain (Current - Previous)
    delta_epsilon: float,       # Privacy cost
    current_step: int = 0,      # Current step (passed from train loop)
    total_steps: int = 1,       # Total steps
    target_epsilon: float = 8.0,  # Target total budget
    current_epsilon: float = 0.0, # Current cumulative epsilon (for budget pacing)
    reward_mode: str = "log_ratio_asym",  # Reward mode
    max_steps: int = None,  # Max steps (for expected cost calculation)
    steps_in_interval: int = 1,  # New: Steps included in current interval
) -> float:
    """
    Refactored reward function...
    """
    # 1. Hard Constraint for Privacy Violation and Performance Collapse
    if current_epsilon >= target_epsilon:
        return -100.0
    
    # Add penalty for Loss explosion (Assume delta_utility = -delta_loss)
    # If Loss increases by more than 2.0, model is collapsing
    if delta_utility < -2.0:
        return -20.0 * abs(delta_utility)
    
    # Modify budget pacing: Use non-linear curve (Square root function)
    progress = min(max(current_step / max(total_steps, 1), 0.0), 1.0)
    paced_eps = target_epsilon * (math.sqrt(progress))
    
    # Compute pacing penalty (Only penalize overspeed, not underspeed)
    pace_penalty = max(0.0, current_epsilon - paced_eps) / max(target_epsilon, 1e-6)
    
    # 2. Dynamic Privacy Penalty Weight lambda_t
    # Correction: Compare "Current Interval Consumption" vs "Expected Interval Consumption"
    lambda_t = 1.0
    if max_steps is not None and max_steps > 0:
        # Compute expected cost for current interval
        expected_cost_interval = (target_epsilon / max_steps) * steps_in_interval
        
        if delta_epsilon > 0 and expected_cost_interval > 0:
            if delta_epsilon > expected_cost_interval:
                # Compute ratio
                ratio = delta_epsilon / expected_cost_interval
                # Limit exponent max to 10, prevent overflow (2^10 = 1024 is large enough)
                ratio = min(ratio, 10.0)
                lambda_t = 2.0 ** ratio
            else:
                lambda_t = 1.0

    def _log_ratio() -> float:
        if abs(delta_epsilon) < 1e-8:
            return 0.0  # Avoid division by zero
        ratio = delta_utility / (delta_epsilon + 1e-6)
        return math.log1p(max(ratio, -0.99))

    # Performance protection: Default strong penalty, changed to smooth penalty in soft_tanh mode
    if reward_mode != "soft_tanh" and delta_utility < 0:
        return delta_utility * 7.0  # Weaken penalty, allow slight exploration
    
    # 1) Asymmetric Log Ratio (Default)
    if reward_mode == "log_ratio_asym":
        return _log_ratio()

    # 2) Budget Pacing Log Ratio: Overspeed epsilon consumption penalized
    if reward_mode == "budget_pace_log":
        # Weaken pacing penalty, encourage spending budget, but use dynamic weights
        base = _log_ratio()
        # Adjust privacy cost using dynamic weights
        privacy_penalty = lambda_t * delta_epsilon if delta_epsilon > 0 else 0.0
        return base - 3.0 * pace_penalty - 0.1 * privacy_penalty

    # 3) Late Stage Privacy Saving: Linear Reward - Progress weighted epsilon penalty, stabilized by tanh
    if reward_mode == "epsilon_guard":
        # More sensitive in late stage: Amplify progress factor, but moderately weaken penalty strength
        # Combine dynamic weights for smarter penalty
        late_factor = 1.0 + 4.0 * progress  # progress=1 -> ~5x penalty
        weighted_eps_cost = lambda_t * late_factor * delta_epsilon
        score = delta_utility - weighted_eps_cost
        return math.tanh(score * 6.0) - 1.0 * pace_penalty

    # 4) Smooth Type: Tolerate small fluctuations, emphasize trend
    if reward_mode == "soft_tanh":
        # Add "Spending Bonus": If budget lags significantly, reward spending
        spending_bonus = 0.0
        eps_ratio = current_epsilon / target_epsilon
        if progress > 0.5 and eps_ratio < progress * 0.9:
            # More lag, larger reward
            spending_bonus = (progress - eps_ratio) * 0.5

        if delta_utility < 0:
            return -math.tanh(abs(delta_utility) * 5.0) - 0.5 * pace_penalty + spending_bonus
        
        ratio = delta_utility / (delta_epsilon + 1e-6)
        return math.tanh(0.5 * ratio) - 0.5 * pace_penalty + spending_bonus

    # Fallback: Use default log ratio and consider pacing penalty
    return _log_ratio() - pace_penalty
