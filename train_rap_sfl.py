#!/usr/bin/env python3
"""
RAP-SFL: Reinforcement-driven Adaptive Perturbation at the Split Point of Split Federated Learning
=======================================================================================
Architecture:
  - N clients, each holding a local data partition (non-IID Dirichlet split)
  - Client: BERT encoder + local SAC RL agent (RAP-SFL perturbation policy)
  - Server: classifier head + global model aggregation (FedAvg)
  - Split point: BERT encoder output (embeddings after RAP-SFL perturbation) → server classifier

Training loop (per federated round):
  1. Server broadcasts the global encoder weights to all clients
  2. Randomly sample K clients to take part in this round
  3. Each participating client:
     a. Forward pass → Adaptive Layer Fusion → RAP-SFL noise injection → perturbed embedding
     b. Send the embedding to the server → server computes the loss → backward pass yields gradients
     c. Client continues backpropagation with those gradients → updates local encoder weights
     d. RL agent observes the state → updates its policy
  4. FedAvg aggregation over the encoder weights
  5. Evaluate the global model every few rounds and log the convergence curves
"""

import argparse
import copy
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, suitable for server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

from rap_sfl_components import (
    SharedStateEncoder,
    ReplayBuffer,
    SACActor,
    SACCritic,
    RunningMeanStd,
    compute_embedding_norm_stats,
    build_state_vector,
    parse_action,
    apply_spah,
    compute_reward,
    compute_attention_entropy,
)
from prv_accountant import PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism


# ============================================================================ #
# Privacy Accountant (reuses the original logic)
# ============================================================================ #

class NumericalPrivacyAccountant:
    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.history = []
        self.total_steps = 0

    def step(self, noise_multiplier: float, sample_rate: float):
        self.total_steps += 1
        if self.history:
            last_nm, last_sr, last_n = self.history[-1]
            if abs(last_nm - noise_multiplier) < 1e-4 and abs(last_sr - sample_rate) < 1e-6:
                self.history[-1] = (last_nm, last_sr, last_n + 1)
                return
        self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(self, delta: float = None) -> float:
        if delta is None:
            delta = self.delta
        if not self.history:
            return 0.0
        try:
            prvs, nums = [], []
            for nm, sr, n in self.history:
                nm = max(nm, 1e-8)
                prvs.append(PoissonSubsampledGaussianMechanism(
                    noise_multiplier=nm, sampling_probability=sr))
                nums.append(n)
            accountant = PRVAccountant(
                prvs=prvs, max_self_compositions=nums,
                eps_error=0.1, delta_error=1e-10)
            _, eps, _ = accountant.compute_epsilon(delta=delta, num_self_compositions=nums)
            return eps if eps > 0 else 0.0
        except Exception as e:
            print(f"[PrivacyWarning] PRVAccountant failed: {e}")
            return 0.0


def estimate_epsilon_from_params(
    noise_multiplier: float,
    sample_rate: float,
    total_steps: int,
    delta: float,
) -> float:
    """Estimate epsilon for repeated Gaussian mechanism with fixed params."""
    if noise_multiplier is None or noise_multiplier <= 0 or sample_rate <= 0 or total_steps <= 0:
        return 0.0
    accountant = NumericalPrivacyAccountant(delta=delta)
    accountant.history = [(float(noise_multiplier), float(sample_rate), int(total_steps))]
    accountant.total_steps = int(total_steps)
    return accountant.get_epsilon(delta=delta)


def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    total_steps: int,
    tol: float = 0.15,
) -> float:
    """
    Calibrate a fixed noise multiplier so that the estimated epsilon is close to target.
    Higher noise_multiplier => lower epsilon.
    """
    if target_epsilon <= 0 or sample_rate <= 0 or total_steps <= 0:
        return 1.0

    low, high = 0.01, 1.0
    high_eps = estimate_epsilon_from_params(high, sample_rate, total_steps, delta)
    while high_eps > target_epsilon and high < 50.0:
        high *= 2.0
        high_eps = estimate_epsilon_from_params(high, sample_rate, total_steps, delta)

    best_nm = high
    best_gap = abs(high_eps - target_epsilon)

    for _ in range(30):
        mid = (low + high) / 2.0
        eps = estimate_epsilon_from_params(mid, sample_rate, total_steps, delta)
        gap = abs(eps - target_epsilon)
        if gap < best_gap:
            best_gap = gap
            best_nm = mid
        if gap <= tol:
            return mid
        if eps > target_epsilon:
            low = mid
        else:
            high = mid

    return best_nm


# ============================================================================ #
# Non-IID data partitioning (Dirichlet)
# ============================================================================ #

def partition_dirichlet(dataset, num_clients: int, alpha: float, seed: int = 42):
    """
    Partition the dataset into num_clients non-IID subsets using a Dirichlet distribution.
    Smaller alpha gives a more uneven split (extreme non-IID); larger alpha approaches IID.

    Returns:
        List[List[int]]: list of sample indices for each client
    """
    rng = np.random.default_rng(seed)

    # Get the labels
    if "label" in dataset.column_names:
        labels = np.array(dataset["label"])
    else:
        raise ValueError("Dataset must have a 'label' column for Non-IID partitioning.")

    num_classes = len(np.unique(labels))
    # Group indices by class
    class_indices = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    for ci in class_indices:
        rng.shuffle(ci)

    # Assign each class to the clients according to a Dirichlet distribution
    client_indices = [[] for _ in range(num_clients)]
    for c_idx, indices in enumerate(class_indices):
        # Dirichlet sampling: share of this class that each client receives
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        # Split according to the proportions
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)
        cuts = np.clip(cuts, 0, len(indices))
        splits = np.split(indices, cuts[:-1])
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    # Shuffle the data order of each client
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
        print(f"  Client {cid}: {len(client_indices[cid])} samples")

    return client_indices


def print_partition_stats(client_indices, dataset, num_classes: int):
    """Print the class-distribution statistics of each client."""
    labels = np.array(dataset["label"])
    print("\n[Data Partition Statistics]")
    print(f"{'Client':>8} | {'Total':>6} | " + " | ".join(f"C{c:>3}" for c in range(num_classes)))
    print("-" * (20 + num_classes * 8))
    for cid, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        client_labels = labels[indices]
        counts = [int((client_labels == c).sum()) for c in range(num_classes)]
        row = f"  {cid:>6} | {len(indices):>6} | " + " | ".join(f"{c:>5}" for c in counts)
        print(row)
    print()


# ============================================================================ #
# Server classifier head (server-side classifier)
# ============================================================================ #

class ServerClassifier(nn.Module):
    """
    Server-side classifier head.
    Receives the perturbed embedding sent by the client ([CLS] position vector) and outputs classification logits.
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


# ============================================================================ #
# Client encoder (client-side encoder, with the RAP-SFL perturbation logic)
# ============================================================================ #

class ClientEncoder(nn.Module):
    """
    Client encoder: the BERT/RoBERTa/DistilBERT encoder part only (no classification head).
    Applies RAP-SFL perturbation during the forward pass and outputs the perturbed embedding to be sent to the server.
    """
    def __init__(self, base_model, model_type: str):
        super().__init__()
        self.model_type = model_type  # 'bert', 'roberta', 'distilbert'

        # Extract the encoder according to the model type
        if model_type == "bert":
            self.encoder = base_model.bert
        elif model_type == "roberta":
            self.encoder = base_model.roberta
        elif model_type == "distilbert":
            self.encoder = base_model.distilbert
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.config = base_model.config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        # RAP-SFL parameters
        rl_layer_weights=None,
        rl_norm_constants=None,
        rl_noise_multiplier=None,
        rl_attn_influence=0.0,
    ):
        """
        Returns:
            perturbed_cls: perturbed [CLS] vector, shape [B, hidden_size], requires_grad=True
            hidden_states: original per-layer hidden states (used to build the RL state)
            attentions: attention weights (used for the semantic mask)
        """
        fwd_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        # BERT supports token_type_ids (separates the two segments in sentence-pair tasks); RoBERTa/DistilBERT do not use it
        if token_type_ids is not None and self.model_type == "bert":
            fwd_kwargs["token_type_ids"] = token_type_ids

        encoder_out = self.encoder(**fwd_kwargs)
        hidden_states = encoder_out.hidden_states  # tuple: (embed, l0, l1, ..., lN)
        attentions = encoder_out.attentions

        # Build the list of layer outputs (including the embedding layer)
        layer_outputs = list(hidden_states)

        # Apply RAP-SFL perturbation
        if (rl_layer_weights is not None and
                rl_norm_constants is not None and
                rl_noise_multiplier is not None):

            rl_layer_weights = rl_layer_weights.detach().clone().requires_grad_(False)
            rl_norm_constants = rl_norm_constants.detach().clone().requires_grad_(False)

            # Align the number of layers
            n_layers = len(layer_outputs)
            if len(rl_layer_weights) != n_layers:
                if len(rl_layer_weights) < n_layers:
                    extra = n_layers - len(rl_layer_weights)
                    pad = torch.ones(extra, device=rl_layer_weights.device) / extra
                    rl_layer_weights = torch.cat([rl_layer_weights, pad])
                else:
                    rl_layer_weights = rl_layer_weights[:n_layers]
                rl_layer_weights = F.softmax(rl_layer_weights, dim=-1).detach()

            device = input_ids.device
            fused = apply_spah(
                layer_outputs=layer_outputs,
                layer_weights=rl_layer_weights,
                norm_constants=rl_norm_constants,
                noise_multiplier=rl_noise_multiplier,
                device=device,
                attention_outputs=attentions,
                attn_influence=rl_attn_influence,
            )
        else:
            fused = hidden_states[-1]  # use the last layer when no perturbation is applied

        # Extract the [CLS] vector (split point)
        # fused: [B, seq_len, hidden_size]; take the token at position 0
        cls_vec = fused[:, 0, :]  # [B, hidden_size]

        # For BERT/RoBERTa, apply the pooler (dense + tanh) to improve classification feature quality
        # DistilBERT has no pooler; when RoBERTa is loaded via AutoModelForSequenceClassification
        # the pooler attribute exists but is None (add_pooling_layer=False), so check explicitly for non-None
        pooler = getattr(self.encoder, "pooler", None)
        if self.model_type in ("bert", "roberta") and pooler is not None:
            cls_vec = pooler(fused)  # [B, hidden_size]

        return cls_vec, hidden_states, attentions


# ============================================================================ #
# Federated client
# ============================================================================ #

class FedClient:
    """
    Federated split-learning client.
    Each client holds:
      - local encoder (ClientEncoder)
      - local RL agent (SAC)
      - local data loader
      - local privacy accountant
    """

    def __init__(self, client_id: int, data_indices: list, dataset,
                 collate_fn, tokenizer, args, device: torch.device,
                 global_encoder_state: dict, hidden_size: int, num_layers: int):
        self.cid = client_id
        self.args = args
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # num_encoder_layers + 1 (embedding)
        self.num_blocks = args.num_blocks

        # Local dataset
        subset = Subset(dataset, data_indices)
        self.data_loader = DataLoader(
            subset,
            batch_size=args.client_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,  # keep the batch size consistent
        )
        self.data_iter = iter(self.data_loader)
        self.local_dataset_size = len(data_indices)

        # Local encoder (initialized from the global model)
        self._init_encoder(global_encoder_state)

        # Local optimizer & LR schedule
        self.optimizer = optim.AdamW(
            self.encoder.parameters(), lr=args.client_lr)

        # RL components
        num_layers_rl = num_layers  # encoder layers + embedding layer
        action_dim = num_layers_rl + self.num_blocks + 1 + 1  # layer_w + norms + attn + noise
        state_dim = 7 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + self.num_blocks  # 15 + K

        self.state_encoder = SharedStateEncoder(state_dim, hidden=64, embed_dim=64).to(device)
        self.actor = SACActor(64, action_dim, hidden=64).to(device)
        self.critic = SACCritic(64, action_dim, hidden=64).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=args.actor_lr * 0.5)
        self.sac_buffer = ReplayBuffer(
            max_size=args.sac_buffer_size,
            state_dim=64,
            action_dim=action_dim,
            device=device,
        )
        self.state_norm = RunningMeanStd(state_dim, device=device)

        # Privacy accountant
        self.privacy_accountant = NumericalPrivacyAccountant(delta=args.delta)
        self.sample_rate = args.client_batch_size / max(self.local_dataset_size, 1)

        # Training state
        self.global_step = 0
        self.recent_losses = deque(maxlen=args.rl_interval)
        self.prev_loss = None
        self.prev_eps = None
        self.utility_ema = None
        self.grad_norm = 0.0
        self.exhausted = False
        self.exhausted_round = None

        # Initial RL parameters (locked to the last layer)
        lw = torch.zeros(num_layers_rl, device=device)
        lw[-1] = 10.0
        lw[:-1] = -10.0
        self.layer_weights = lw
        self.norm_constants = torch.ones(self.num_blocks, device=device) * args.default_norm_c
        self.noise_multiplier = args.default_noise_multiplier
        self.attn_influence = 0.0

        # RL state buffer (used to compute the reward at the next step)
        self._stored_state = None
        self._stored_action = None
        self.prox_anchor = None  # FedProx: snapshot of the global encoder weights at the start of each round

        # ALF experiment: layer-weight log (one row per RL step)
        # Fields: fed_round, global_step, eps_ratio, w_0, w_1, ..., w_{L-1}
        self.alf_log_path = None  # set externally by the main training loop
        self.current_fed_round = 0

        # SCAFFOLD: local control variable c_i (stored on CPU, not part of the gradient computation)
        self.scaffold_ci = {
            k: torch.zeros_like(v, device="cpu")
            for k, v in self.encoder.state_dict().items()
        }
        self.scaffold_c = None
        self._round_start_state = None

    def _init_encoder(self, global_state: dict):
        """Initialize the local encoder from the global weights."""
        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_flash_attention_utils").setLevel(logging.ERROR)
        # Suppress the transformers 5.x "layers not sharded" message
        logging.getLogger("transformers").setLevel(logging.ERROR)

        config = AutoConfig.from_pretrained(self.args.model_name)
        config.num_labels = self.args.num_labels
        config.output_attentions = True

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config)

        model_type = config.model_type  # 'bert', 'roberta', 'distilbert'
        self.encoder = ClientEncoder(base_model, model_type).to(self.device)
        del base_model  # release unneeded weights such as the classifier head to save memory

        # Load the global encoder weights
        if global_state:
            self.encoder.load_state_dict(global_state, strict=True)

    def load_global_encoder(self, global_state: dict,
                            scaffold_global_c: dict = None):
        """Receive the global encoder weights broadcast by the server and reset the optimizer state (clear stale momentum)."""
        self.encoder.load_state_dict(global_state, strict=True)
        self.optimizer = optim.AdamW(
            self.encoder.parameters(), lr=self.args.client_lr)
        if self.args.fedprox_mu > 0:
            self.prox_anchor = {
                k: v.detach().clone() for k, v in self.encoder.state_dict().items()
            }
        else:
            self.prox_anchor = None
        # SCAFFOLD: save the initial state of this round (CPU) and the global control variable (CPU)
        if self.args.scaffold:
            self._round_start_state = {
                k: v.detach().cpu().clone()
                for k, v in self.encoder.state_dict().items()
            }
            if scaffold_global_c is not None:
                self.scaffold_c = {
                    k: v.cpu().clone() for k, v in scaffold_global_c.items()
                }

    def get_encoder_state(self) -> dict:
        """Return the local encoder weights (for FedAvg)."""
        return {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}

    def scaffold_update_ci(self, global_state: dict, local_steps: int):
        """
        SCAFFOLD control variate update in *parameter-drift* space.
        Store c_i as the average parameter drift (θ^t - θ_i) / K directly,
        NOT divided by η. This keeps c_i in the same scale as parameter
        differences (~1e-5), avoiding the magnitude explosion from 1/(η*K).

        c_i^+ = (θ^t - θ_i) / K
        """
        K = max(local_steps, 1)
        delta_ci = {}
        new_ci = {}
        local_state = {k: v.cpu() for k, v in self.encoder.state_dict().items()}
        for k in self.scaffold_ci:
            start_param = self._round_start_state[k]
            end_param = local_state[k]
            ci_old = self.scaffold_ci[k]
            ci_new = (start_param - end_param) / K
            delta_ci[k] = (ci_new - ci_old).clone()
            new_ci[k] = ci_new
        self.scaffold_ci = new_ci
        return delta_ci

    def _next_batch(self):
        """Take the next batch from the local data iterator; reset it when exhausted."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch

    def local_step(self, server_classifier: nn.Module,
                   server_classifier_opt: optim.Optimizer,
                   total_steps: int):
        """
        Run one local training step (the core split-learning step).

        Returns:
            loss_val (float): loss value of the current step
            correct (int): number of correct predictions
            batch_size (int): batch size
        """
        self.global_step += 1
        batch = self._next_batch()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device) if "token_type_ids" in batch else None
        labels = batch["label"].to(self.device) if "label" in batch else batch["labels"].to(self.device)
        B = input_ids.size(0)

        # ------------------------------------------------------------------ #
        # Step 1: RL agent decision (policy updated every rl_interval steps)
        # ------------------------------------------------------------------ #
        args = self.args
        # ALF analysis variant (fixed layer-selection policy, RL exploration of layer weights disabled)
        alf_strategy = getattr(args, "alf_strategy", "rap_sfl")
        use_rl = (not args.disable_rl) and alf_strategy == "rap_sfl"
        is_rl_step = (use_rl and
                      self.global_step >= args.sac_start_steps and
                      self.global_step % args.rl_interval == 0)

        # Under a fixed policy: every step uses the predefined layer_weights, and norm/noise use their defaults
        if alf_strategy != "rap_sfl":
            lw = torch.zeros(self.num_layers, device=self.device)
            if alf_strategy == "last_only":
                lw[-1] = 1.0
            elif alf_strategy == "first_only":
                lw[0] = 1.0
            elif alf_strategy == "middle_only":
                lw[self.num_layers // 2] = 1.0
            elif alf_strategy == "uniform":
                lw[:] = 1.0 / self.num_layers
            else:
                raise ValueError(f"Unknown alf_strategy: {alf_strategy}")
            self.layer_weights = lw
            self.norm_constants = torch.ones(
                self.num_blocks, device=self.device) * args.default_norm_c
            self.noise_multiplier = args.default_noise_multiplier
            self.attn_influence = 0.0

        if is_rl_step:
            avg_loss = (sum(self.recent_losses) / len(self.recent_losses)
                        if self.recent_losses else 0.0)
            utility = -avg_loss

            # Get the current encoder output without gradients, used to build the RL state
            with torch.no_grad():
                rl_fwd_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
                if token_type_ids is not None and self.encoder.model_type == "bert":
                    rl_fwd_kwargs["token_type_ids"] = token_type_ids
                enc_out = self.encoder.encoder(**rl_fwd_kwargs)
                hs = enc_out.hidden_states
                atts = enc_out.attentions

            attn_entropy = compute_attention_entropy(atts)
            norm_stats = compute_embedding_norm_stats(hs[-1])
            spent_eps = self.privacy_accountant.get_epsilon(delta=args.delta)
            batch_loss = self.recent_losses[-1] if self.recent_losses else 0.0

            state = build_state_vector(
                embedding_norm_stats=norm_stats,
                utility=utility,
                spent_eps=spent_eps,
                batch_loss=batch_loss,
                num_blocks=self.num_blocks,
                target_epsilon=args.epsilon,
                prev_loss=self.prev_loss,
                gradient_norm=self.grad_norm,
                attention_entropy=attn_entropy,
                device=self.device,
            )
            self.state_norm.update(state.unsqueeze(0))
            norm_state = self.state_norm.normalize(state.unsqueeze(0))
            enc_state = self.state_encoder(norm_state)

            action_raw, _ = self.actor.sample(enc_state)
            action_vec = action_raw.squeeze(0)

            progress = self.global_step / max(total_steps, 1)
            eps_ratio = spent_eps / args.epsilon if args.epsilon > 0 else 0.0
            self.layer_weights, self.norm_constants, self.noise_multiplier, self.attn_influence = \
                parse_action(
                    action_vector=action_vec,
                    num_layers=self.num_layers,
                    num_blocks=self.num_blocks,
                    progress=progress,
                    eps_ratio=eps_ratio,
                    min_noise_bound=args.default_noise_multiplier,
                    ablation_mode=args.ablation_mode,
                )

            self._stored_state = enc_state.squeeze(0).detach().cpu().numpy()
            self._stored_action = action_vec.detach().cpu().numpy()

            # ALF experiment: log the layer weights to CSV
            if self.alf_log_path is not None:
                lw_np = self.layer_weights.detach().cpu().numpy()
                row = [self.current_fed_round, self.global_step,
                       float(eps_ratio)] + [float(w) for w in lw_np]
                with open(self.alf_log_path, "a") as f:
                    f.write(",".join(f"{x}" for x in row) + "\n")

        elif self.global_step < args.sac_start_steps:
            # Warm-up phase: locked to the last layer
            lw = torch.zeros(self.num_layers, device=self.device)
            lw[-1] = 10.0; lw[:-1] = -10.0
            self.layer_weights = lw
            self.norm_constants = torch.ones(
                self.num_blocks, device=self.device) * args.default_norm_c * 1.5
            self.noise_multiplier = args.default_noise_multiplier
            self.attn_influence = 0.0

        # ------------------------------------------------------------------ #
        # Step 2: client forward pass (RAP-SFL perturbation)
        # ------------------------------------------------------------------ #
        lw = self.layer_weights.detach().clone().to(self.device)
        nc = self.norm_constants.detach().clone().to(self.device)

        # Client forward pass (with gradients)
        cls_vec, hidden_states, attentions = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            rl_layer_weights=lw,
            rl_norm_constants=nc,
            rl_noise_multiplier=self.noise_multiplier,
            rl_attn_influence=self.attn_influence,
        )
        # cls_vec: [B, hidden_size], part of the computation graph

        # ------------------------------------------------------------------ #
        # Step 3: create the "communication boundary" — simulates cross-network transfer
        # cls_vec_server needs requires_grad=True so that it can receive the server gradients
        # ------------------------------------------------------------------ #
        cls_vec_server = cls_vec.detach().requires_grad_(True)

        # ------------------------------------------------------------------ #
        # Step 4: server forward pass + loss computation + backward (only up to the split point)
        # ------------------------------------------------------------------ #
        server_classifier_opt.zero_grad()
        logits = server_classifier(cls_vec_server)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
        loss.backward()  # gradients stop at cls_vec_server

        grad_from_server = cls_vec_server.grad.clone()  # gradients returned by the server
        server_classifier_opt.step()  # server classifier head weight update

        # ------------------------------------------------------------------ #
        # Step 5: client backward pass (using the gradients returned by the server)
        # ------------------------------------------------------------------ #
        self.optimizer.zero_grad()
        cls_vec.backward(grad_from_server)  # continue backpropagation through the encoder

        # FedProx: add the gradient μ(θ - θ^t) of (μ/2)||θ - θ^t||^2 on the encoder parameters
        if args.fedprox_mu > 0 and self.prox_anchor is not None:
            for name, p in self.encoder.named_parameters():
                if p.grad is None:
                    continue
                anchor = self.prox_anchor[name].to(p.device)
                p.grad.add_(args.fedprox_mu * (p - anchor))

        # SCAFFOLD: g_corrected = g + scaffold_lr * (c - c_i)
        # c and c_i are stored in parameter-drift space (magnitude ~1e-5,
        # same order as gradients), so the correction is numerically stable.
        if args.scaffold and self.scaffold_c is not None:
            scaffold_lr = getattr(args, 'scaffold_lr', 1.0)
            for name, p in self.encoder.named_parameters():
                if p.grad is None or name not in self.scaffold_ci:
                    continue
                correction = (self.scaffold_c[name] - self.scaffold_ci[name]).to(p.device)
                p.grad.add_(scaffold_lr * correction)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

        # Compute the gradient norm (used in the RL state)
        g_norm = 0.0
        for p in self.encoder.parameters():
            if p.grad is not None:
                g_norm += p.grad.data.norm(2).item() ** 2
        self.grad_norm = g_norm ** 0.5

        self.optimizer.step()

        # ------------------------------------------------------------------ #
        # Step 6: update the privacy accountant & training state
        # ------------------------------------------------------------------ #
        loss_val = loss.item()
        self.recent_losses.append(loss_val)
        self.prev_loss = loss_val

        if not args.no_dp:
            self.privacy_accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate,
            )

        # ------------------------------------------------------------------ #
        # Step 7: RL reward computation & SAC update
        # ------------------------------------------------------------------ #
        if is_rl_step and self._stored_state is not None:
            current_utility = -loss_val
            current_eps = self.privacy_accountant.get_epsilon(delta=args.delta)
            ema_alpha = 0.1
            if self.utility_ema is None:
                self.utility_ema = current_utility

            if self.prev_eps is not None:
                delta_u = current_utility - self.utility_ema
                delta_eps = current_eps - self.prev_eps
                reward = compute_reward(
                    delta_utility=delta_u,
                    delta_epsilon=delta_eps,
                    current_step=self.global_step,
                    total_steps=total_steps,
                    target_epsilon=args.epsilon,
                    current_epsilon=current_eps,
                    reward_mode=args.reward_mode,
                    max_steps=total_steps,
                    steps_in_interval=args.rl_interval,
                )
                self.utility_ema = ema_alpha * current_utility + (1 - ema_alpha) * self.utility_ema
            else:
                reward = 0.0

            self.prev_eps = current_eps
            self.sac_buffer.add(
                self._stored_state, self._stored_action, reward,
                self._stored_state, done=False)

            if self.sac_buffer.size >= args.sac_batch_size:
                for _ in range(args.sac_updates_per_interval):
                    self._update_sac()

        # Count the correct classifications
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()

        return loss_val, correct, B

    def _update_sac(self):
        """SAC policy network update."""
        args = self.args
        s_b, a_b, r_b, s2_b, d_b = self.sac_buffer.sample(args.sac_batch_size)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2_b)
            q1_t, q2_t = self.critic_target(s2_b, a2)
            q_targ = torch.min(q1_t, q2_t) - args.alpha * logp2
            y = r_b + args.gamma * (1 - d_b) * q_targ

        q1_c, q2_c = self.critic(s_b, a_b)
        loss_q = F.smooth_l1_loss(q1_c, y) + F.smooth_l1_loss(q2_c, y)
        self.critic_opt.zero_grad(); loss_q.backward(); self.critic_opt.step()

        a_pi, logp_pi = self.actor.sample(s_b)
        q1_pi, q2_pi = self.critic(s_b, a_pi)
        loss_pi = (args.alpha * logp_pi - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad(); loss_pi.backward(); self.actor_opt.step()

        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1 - args.tau)
            p_t.data.add_(args.tau * p.data)

    def get_spent_epsilon(self) -> float:
        return self.privacy_accountant.get_epsilon(delta=self.args.delta)

    def is_exhausted(self) -> bool:
        if self.args.no_dp:
            return False
        if self.exhausted:
            return True
        return self.get_spent_epsilon() >= self.args.epsilon

    def mark_exhausted(self, round_idx: int = None):
        self.exhausted = True
        if round_idx is not None and self.exhausted_round is None:
            self.exhausted_round = round_idx


# ============================================================================ #
# FedAvg aggregation
# ============================================================================ #

def fed_avg(client_states: list, client_weights: list) -> dict:
    """
    FedAvg weighted average.
    client_weights: data volume of each client (used for the weighted average).
    """
    total_weight = sum(client_weights)
    avg_state = {}
    for key in client_states[0].keys():
        avg_state[key] = sum(
            w / total_weight * state[key].float()
            for state, w in zip(client_states, client_weights)
        )
    return avg_state


# ============================================================================ #
# Evaluation function
# ============================================================================ #

def evaluate_global(encoder_state: dict, server_classifier: nn.Module,
                    eval_loader: DataLoader, args, device: torch.device,
                    eval_encoder: "ClientEncoder" = None) -> tuple:
    """Evaluate model performance with the global encoder weights + server classifier head.
    Passing eval_encoder is recommended to avoid reloading the model every time (performance optimization).
    """
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if eval_encoder is None:
        # fallback: cold start (slow; recommended only for the first call)
        config = AutoConfig.from_pretrained(args.model_name)
        config.num_labels = args.num_labels
        config.output_attentions = True
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=config)
        eval_encoder = ClientEncoder(base_model, config.model_type).to(device)
        del base_model
    
    eval_encoder.load_state_dict(encoder_state)
    eval_encoder.eval()
    server_classifier.eval()

    all_preds, all_labels, all_losses = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            labels = (batch["label"] if "label" in batch else batch["labels"]).to(device)

            # No noise during evaluation (use the last layer, no perturbation)
            cls_vec, _, _ = eval_encoder(input_ids, attention_mask,
                                         token_type_ids=token_type_ids)
            logits = server_classifier(cls_vec)

            loss = nn.CrossEntropyLoss()(
                logits.view(-1, args.num_labels), labels.view(-1))
            preds = torch.argmax(logits, dim=-1)

            all_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = np.mean(all_losses)

    server_classifier.train()
    return acc, avg_loss


# ============================================================================ #
# Task configuration (dataset / text fields / number of labels)
# ============================================================================ #

TASK_CONFIGS = {
    # GLUE SST-2: single-sentence sentiment classification
    "sst2": {
        "dataset_args": ("glue", "sst2"),
        "text_fields": ["sentence"],
        "num_labels": 2,
        "eval_split": "validation",
    },
    # Standalone IMDB: single-sentence sentiment classification, evaluation uses the test split
    "imdb": {
        "dataset_args": ("imdb",),
        "text_fields": ["text"],
        "num_labels": 2,
        "eval_split": "test",
    },
    # GLUE QQP: sentence-pair question-equivalence classification
    "qqp": {
        "dataset_args": ("glue", "qqp"),
        "text_fields": ["question1", "question2"],
        "num_labels": 2,
        "eval_split": "validation",
    },
    # GLUE MNLI: natural language inference (3-way: entailment / contradiction / neutral)
    "mnli": {
        "dataset_args": ("glue", "mnli"),
        "text_fields": ["premise", "hypothesis"],
        "num_labels": 3,
        "eval_split": "validation_matched",
    },
}

SUPPORTED_MODELS = {
    "bert-base-uncased": "bert",
    "roberta-base": "roberta",
    "distilbert-base-uncased": "distilbert",
}


# ============================================================================ #
# Main training function
# ============================================================================ #

def train_rap_sfl(args):
    # Suppress the verbose transformers loading logs (LOAD REPORT / not sharded messages, etc.)
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    print("=" * 70)
    print("RAP-SFL: Reinforcement-driven Adaptive Perturbation at the Split Point")
    print("=" * 70)
    print(f"  Model: {args.model_name}")
    print(f"  Task: {args.task_name}")
    print(f"  Num clients: {args.num_clients}")
    print(f"  Clients per round: {args.clients_per_round}")
    print(f"  Local steps per round: {args.local_steps_per_round}")
    print(f"  Fed rounds: {args.fed_rounds}")
    print(f"  Dirichlet alpha: {args.dirichlet_alpha}")
    print(f"  Target epsilon: {args.epsilon}")
    print(f"  RL enabled: {not args.disable_rl}")
    print(f"  Ablation mode: {args.ablation_mode}")
    print(f"  FedProx mu: {args.fedprox_mu} (0 = disabled)")
    print(f"  SCAFFOLD: {args.scaffold}")
    print(f"  DP-FedAvg σ: {args.dp_fedavg_sigma} (0 = disabled)")
    print("=" * 70)

    # Random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------------- #
    # Load the dataset & tokenizer
    # ---------------------------------------------------------------------- #
    print("\n[1] Loading dataset...")

    if args.task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unsupported task '{args.task_name}'. "
            f"Choose from: {list(TASK_CONFIGS.keys())}"
        )
    task_cfg = TASK_CONFIGS[args.task_name]

    # If num_labels is not explicitly specified, read it from the task configuration
    if args.num_labels is None:
        args.num_labels = task_cfg["num_labels"]
        print(f"  Auto-set num_labels={args.num_labels} for task '{args.task_name}'")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load the raw dataset
    dataset = load_dataset(*task_cfg["dataset_args"])
    train_dataset_raw = dataset["train"]
    eval_dataset_raw = dataset[task_cfg["eval_split"]]

    # The IMDB test split contains a few unsupervised samples with label=-1, which must be filtered out
    if args.task_name == "imdb":
        eval_dataset_raw = eval_dataset_raw.filter(lambda x: x["label"] != -1)

    text_fields = task_cfg["text_fields"]

    def tokenize_fn(examples):
        if len(text_fields) == 1:
            return tokenizer(
                examples[text_fields[0]],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        else:
            # Sentence-pair tasks (QQP, etc.)
            return tokenizer(
                examples[text_fields[0]],
                examples[text_fields[1]],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )

    # Keep only input_ids, attention_mask, token_type_ids (if present) and label
    keep_cols = {"label"}
    cols_to_remove_train = [c for c in train_dataset_raw.column_names
                            if c not in keep_cols]
    cols_to_remove_eval = [c for c in eval_dataset_raw.column_names
                           if c not in keep_cols]

    train_tokenized = train_dataset_raw.map(
        tokenize_fn, batched=True,
        remove_columns=cols_to_remove_train,
        load_from_cache_file=False,
    )
    eval_tokenized = eval_dataset_raw.map(
        tokenize_fn, batched=True,
        remove_columns=cols_to_remove_eval,
        load_from_cache_file=False,
    )
    train_tokenized.set_format(type="torch")
    eval_tokenized.set_format(type="torch")

    def collate_fn(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            out[k] = torch.stack([b[k] for b in batch])
        return out

    eval_loader = DataLoader(
        eval_tokenized,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"  Train size: {len(train_tokenized)}, Eval size: {len(eval_tokenized)}")

    # ---------------------------------------------------------------------- #
    # Non-IID data partitioning
    # ---------------------------------------------------------------------- #
    print(f"\n[2] Partitioning data (Dirichlet alpha={args.dirichlet_alpha})...")
    client_data_indices = partition_dirichlet(
        train_tokenized, args.num_clients, args.dirichlet_alpha, seed=args.seed)
    print_partition_stats(client_data_indices, train_tokenized, num_classes=args.num_labels)

    active_client_sizes = [len(indices) for indices in client_data_indices if len(indices) >= args.client_batch_size]
    effective_clients_per_round = len(active_client_sizes) if (
        args.clients_per_round <= 0 or args.clients_per_round >= len(active_client_sizes)
    ) else args.clients_per_round
    estimated_steps_per_client = math.ceil(
        args.fed_rounds * args.local_steps_per_round * effective_clients_per_round / max(len(active_client_sizes), 1)
    )
    max_sample_rate = max((args.client_batch_size / size) for size in active_client_sizes) if active_client_sizes else 0.0

    if args.default_noise_multiplier is None and not args.no_dp:
        args.default_noise_multiplier = calibrate_noise_multiplier(
            target_epsilon=args.epsilon,
            delta=args.delta,
            sample_rate=max_sample_rate,
            total_steps=estimated_steps_per_client,
        )
        estimated_eps = estimate_epsilon_from_params(
            args.default_noise_multiplier, max_sample_rate, estimated_steps_per_client, args.delta
        )
        print(
            f"[Auto Privacy Calibration] sample_rate={max_sample_rate:.6f}, "
            f"steps/client≈{estimated_steps_per_client}, "
            f"noise_multiplier={args.default_noise_multiplier:.4f}, "
            f"estimated_max_epsilon≈{estimated_eps:.4f}"
        )

    # ---------------------------------------------------------------------- #
    # Initialize the global model
    # ---------------------------------------------------------------------- #
    print("[3] Initializing global model...")
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = args.num_labels
    config.output_attentions = True

    # Initialize a temporary encoder to obtain the global initial weights
    base_model_init = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config)
    tmp_encoder = ClientEncoder(base_model_init, config.model_type).to(device)
    global_encoder_state = {k: v.cpu().clone() for k, v in tmp_encoder.state_dict().items()}
    hidden_size = config.hidden_size
    # DistilBERT uses n_layers, BERT/RoBERTa use num_hidden_layers
    raw_layers = getattr(config, "num_hidden_layers",
                         getattr(config, "n_layers", 6))
    num_encoder_layers = raw_layers + 1  # +1 for embedding layer

    # If num_blocks is not set manually, infer it from the model config (DistilBERT=6, BERT/RoBERTa=12)
    if args.num_blocks is None:
        args.num_blocks = raw_layers
        print(f"  Auto-set num_blocks={args.num_blocks} from model config")

    print(f"  hidden_size={hidden_size}, encoder_layers={num_encoder_layers}, num_blocks={args.num_blocks}")
    del tmp_encoder, base_model_init

    # Server classifier head
    server_classifier = ServerClassifier(
        hidden_size=hidden_size, num_labels=args.num_labels).to(device)
    server_cls_opt = optim.AdamW(server_classifier.parameters(), lr=args.server_lr)

    # Pre-initialize the global eval encoder (avoids reloading the model on every evaluate_global call)
    print("[3.5] Pre-initializing eval encoder...")
    eval_config = AutoConfig.from_pretrained(args.model_name)
    eval_config.num_labels = args.num_labels
    eval_config.output_attentions = True
    eval_base = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=eval_config)
    eval_encoder = ClientEncoder(eval_base, eval_config.model_type).to(device)
    del eval_base

    # ---------------------------------------------------------------------- #
    # Initialize all federated clients
    # ---------------------------------------------------------------------- #
    print(f"[4] Creating {args.num_clients} federated clients...")
    clients = []
    for cid in range(args.num_clients):
        if len(client_data_indices[cid]) < args.client_batch_size:
            print(f"  Warning: Client {cid} has only {len(client_data_indices[cid])} samples "
                  f"(< batch_size={args.client_batch_size}), skipping.")
            clients.append(None)
            continue
        client = FedClient(
            client_id=cid,
            data_indices=client_data_indices[cid],
            dataset=train_tokenized,
            collate_fn=collate_fn,
            tokenizer=tokenizer,
            args=args,
            device=device,
            global_encoder_state=global_encoder_state,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
        )
        clients.append(client)
        print(f"  Client {cid} initialized: {len(client_data_indices[cid])} samples")

    active_clients = [c for c in clients if c is not None]
    print(f"  Active clients: {len(active_clients)}/{args.num_clients}")

    # SCAFFOLD: global control variable c (same shape as the encoder parameters, initialized to zero)
    scaffold_global_c = None
    if args.scaffold:
        scaffold_global_c = {
            k: torch.zeros_like(v) for k, v in global_encoder_state.items()
        }
        print("  [SCAFFOLD] Global control variate initialized.")

    # ALF experiment: initialize the layer-weight log CSV (one file per client)
    if getattr(args, "alf_logging", False):
        n_layers_log = num_encoder_layers
        header = ["fed_round", "global_step", "eps_ratio"] + \
                 [f"w_{i}" for i in range(n_layers_log)]
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in active_clients:
            c.alf_log_path = str(out_dir / f"alf_layer_weights_client{c.cid}.csv")
            with open(c.alf_log_path, "w") as f:
                f.write(",".join(header) + "\n")
        print(f"  [ALF Logging] Per-client CSVs written under {out_dir}")

    # ---------------------------------------------------------------------- #
    # Main federated training loop
    # ---------------------------------------------------------------------- #
    print("\n[5] Starting Federated Training...\n")

    # Record the convergence curve data
    history = {
        "round": [],
        "eval_accuracy": [],
        "eval_loss": [],
        "avg_train_loss": [],
        "avg_epsilon": [],
        "max_epsilon": [],
        "active_clients": [],
        "exhausted_clients": [],
    }

    # total_local_steps is used for progress normalization in the RL state and is always computed from fed_rounds,
    # so that in budget-driven mode the RL policy does not see degenerate states caused by round inflation.
    total_local_steps = args.fed_rounds * args.local_steps_per_round
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Determine the actual maximum number of rounds
    # Normal mode: fed_rounds is a hard upper bound
    # stop_by_budget mode: fed_rounds is only a "reference length", the actual bound is 3x larger,
    #   and the main stopping condition is exhaustion of every client's budget; this way, even if the fastest client freezes first,
    #   the remaining clients can keep training until their own ε=target is used up.
    # ---------------------------------------------------------------------- #
    if args.stop_by_budget and not args.no_dp:
        effective_fed_rounds = args.fed_rounds * 3
        print(f"\n  [Budget-driven mode] Primary stop: all clients exhaust ε={args.epsilon}")
        print(f"  Expected rounds (calibration): {args.fed_rounds}, safety cap: {effective_fed_rounds}")
    else:
        effective_fed_rounds = args.fed_rounds

    best_accuracy = 0.0
    no_improve_evals = 0

    for fed_round in range(1, effective_fed_rounds + 1):
        round_start = time.time()
        print(f"{'=' * 60}")
        print(f"[Round {fed_round}/{effective_fed_rounds}"
              f"{' (budget-driven)' if args.stop_by_budget and not args.no_dp else ''}]")

        trainable_clients = [c for c in active_clients if not c.is_exhausted()]
        exhausted_clients = [c for c in active_clients if c.is_exhausted()]
        if exhausted_clients:
            exhausted_ids = [c.cid for c in exhausted_clients]
            print(f"  Exhausted clients (frozen): {exhausted_ids}")
        print(f"  Trainable clients this round: {len(trainable_clients)}/{len(active_clients)}")

        if not trainable_clients:
            print("\n  All clients have exhausted their privacy budgets. Stopping.")
            break

        # ------------------------------------------------------------------ #
        # (A) Broadcast the global encoder weights to the participating clients
        # ------------------------------------------------------------------ #
        # Randomly sample the clients taking part in this round
        if args.clients_per_round <= 0 or args.clients_per_round >= len(trainable_clients):
            selected = list(trainable_clients)
        else:
            k = min(args.clients_per_round, len(trainable_clients))
            selected = random.sample(trainable_clients, k)
        print(f"  Selected clients: {[c.cid for c in selected]}")

        for c in selected:
            c.load_global_encoder(global_encoder_state,
                                  scaffold_global_c=scaffold_global_c)
            c.current_fed_round = fed_round

        # ------------------------------------------------------------------ #
        # (B) Local training: each client runs local_steps_per_round steps
        # ------------------------------------------------------------------ #
        round_losses = []
        round_corrects = 0
        round_total = 0
        newly_exhausted = []

        for c in selected:
            if c.is_exhausted():
                c.mark_exhausted(fed_round)
                continue
            client_loss_sum = 0.0
            client_correct = 0
            client_total = 0
            actual_local_steps = 0

            pbar = tqdm(
                range(args.local_steps_per_round),
                desc=f"  Client {c.cid}",
                leave=False,
            )
            for _ in pbar:
                if c.is_exhausted():
                    c.mark_exhausted(fed_round)
                    break
                loss_val, correct, bs = c.local_step(
                    server_classifier=server_classifier,
                    server_classifier_opt=server_cls_opt,
                    total_steps=total_local_steps,
                )
                client_loss_sum += loss_val
                client_correct += correct
                client_total += bs
                actual_local_steps += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                if c.is_exhausted():
                    c.mark_exhausted(fed_round)
                    newly_exhausted.append(c.cid)
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "status": "budget_exhausted"})
                    break

            if actual_local_steps == 0:
                print(f"  Client {c.cid}: skipped (privacy budget exhausted)")
                continue

            avg_client_loss = client_loss_sum / actual_local_steps
            client_acc = client_correct / max(client_total, 1)
            eps = c.get_spent_epsilon()
            print(f"  Client {c.cid}: avg_loss={avg_client_loss:.4f}, "
                  f"local_acc={client_acc:.4f}, ε={eps:.3f}")

            round_losses.append(avg_client_loss)
            round_corrects += client_correct
            round_total += client_total

        # ------------------------------------------------------------------ #
        # (C) Aggregation (FedAvg / SCAFFOLD / DP-FedAvg)
        # ------------------------------------------------------------------ #
        participating_clients = [
            c for c in selected if (not c.is_exhausted()) or c.exhausted_round == fed_round
        ]
        participating_states = [c.get_encoder_state() for c in participating_clients]
        participating_weights = [len(client_data_indices[c.cid]) for c in participating_clients]
        if participating_states:
            global_encoder_state = fed_avg(participating_states, participating_weights)

            # SCAFFOLD: collect Δc_i from each client and update the global control variable c
            if args.scaffold and scaffold_global_c is not None:
                n_active = len(active_clients)
                for pc in participating_clients:
                    actual_steps = pc.global_step  # approximate with the cumulative step count
                    delta_ci = pc.scaffold_update_ci(
                        global_encoder_state, local_steps=args.local_steps_per_round)
                    for k in scaffold_global_c:
                        scaffold_global_c[k] = scaffold_global_c[k] + delta_ci[k] / max(n_active, 1)
                print(f"  SCAFFOLD aggregation done. Aggregated {len(participating_clients)} clients.")
            else:
                print(f"  FedAvg done. Aggregated {len(participating_clients)} clients.")

            # DP-FedAvg: add Gaussian noise to the aggregated global model to realize CDP
            if args.dp_fedavg_sigma > 0:
                for k in global_encoder_state:
                    noise = torch.randn_like(global_encoder_state[k]) * args.dp_fedavg_sigma
                    global_encoder_state[k] = global_encoder_state[k] + noise
                print(f"  DP-FedAvg: added Gaussian noise σ={args.dp_fedavg_sigma} to global model.")
        else:
            print("  Aggregation skipped. No trainable clients contributed updates this round.")

        if newly_exhausted:
            print(f"  Newly exhausted this round: {sorted(set(newly_exhausted))}")
            # In budget-driven mode: a client has just exhausted its budget, so the training dynamics change
            # (after that client drops out, the per-round participation rate of the remaining clients rises and the gradient direction changes)
            # Reset the early-stopping counter so that early stopping is not triggered by the brief fluctuation at this transition point alone,
            # ensuring the remaining clients have enough rounds to train up to their own budget limits.
            if args.stop_by_budget and not args.no_dp and no_improve_evals > 0:
                print(f"  (Budget-driven: early-stop counter reset after client freeze, "
                      f"was {no_improve_evals})")
                no_improve_evals = 0

        # ------------------------------------------------------------------ #
        # (D) Periodic evaluation
        # ------------------------------------------------------------------ #
        avg_train_loss = np.mean(round_losses) if round_losses else 0.0
        all_eps = [c.get_spent_epsilon() for c in active_clients]
        avg_epsilon = np.mean(all_eps)
        max_epsilon = np.max(all_eps)

        if fed_round % args.eval_every == 0 or fed_round == args.fed_rounds:
            print(f"\n  [Eval] Round {fed_round}...")
            acc, eval_loss = evaluate_global(
                global_encoder_state, server_classifier, eval_loader, args, device,
                eval_encoder=eval_encoder)

            round_duration = time.time() - round_start
            print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
            print(f"  Eval Loss: {eval_loss:.4f}")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")
            print(f"  Avg ε: {avg_epsilon:.4f}")
            print(f"  Max ε: {max_epsilon:.4f}")
            print(f"  Round time: {round_duration:.1f}s")

            history["round"].append(fed_round)
            history["eval_accuracy"].append(float(acc))
            history["eval_loss"].append(float(eval_loss))
            history["avg_train_loss"].append(float(avg_train_loss))
            history["avg_epsilon"].append(float(avg_epsilon))
            history["max_epsilon"].append(float(max_epsilon))
            history["active_clients"].append(int(len(trainable_clients)))
            history["exhausted_clients"].append(int(len(exhausted_clients)))

            # Save the best model
            if acc > best_accuracy + args.early_stop_min_delta:
                best_accuracy = acc
                no_improve_evals = 0
                best_ckpt = {
                    "round": fed_round,
                    "accuracy": acc,
                    "eval_loss": eval_loss,
                    "epsilon": avg_epsilon,
                }
                torch.save(global_encoder_state, output_dir / "best_encoder.pt")
                server_classifier.cpu()
                torch.save(server_classifier.state_dict(), output_dir / "best_classifier.pt")
                server_classifier.to(device)
                if active_clients:
                    ref = active_clients[0]
                    rl_ckpt = {
                        "actor": ref.actor.state_dict(),
                        "encoder": ref.state_encoder.state_dict(),
                        "state_norm": {
                            "mean": ref.state_norm.mean.cpu().clone(),
                            "var": ref.state_norm.var.cpu().clone(),
                        },
                        "round": fed_round,
                        "client_id": ref.cid,
                    }
                    torch.save(rl_ckpt, output_dir / "rl_policy.pt")
                print(f"  *** New best accuracy: {acc:.4f} (saved) ***")
            else:
                no_improve_evals += 1
                print(f"  No improvement for {no_improve_evals} eval(s).")

            # Plot and save the convergence curves
            _plot_convergence(history, output_dir)

            if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                print(
                    f"\n  Early stopping triggered: no improvement greater than "
                    f"{args.early_stop_min_delta:.4f} for {no_improve_evals} evals."
                )
                break

        # Save the intermediate history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        remaining_clients = [c for c in active_clients if not c.is_exhausted()]
        if not remaining_clients and not args.no_dp:
            print("\n  All clients have exhausted their privacy budgets. Stopping.")
            break

    # ---------------------------------------------------------------------- #
    # Final results
    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Best accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)

    # Save the final history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    _plot_convergence(history, output_dir, final=True)

    return history


def _plot_convergence(history: dict, output_dir: Path, final: bool = False):
    """Plot the convergence curves (Accuracy & Loss vs. Federation Rounds)."""
    if not history["round"]:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Accuracy
    axes[0].plot(history["round"], history["eval_accuracy"], "b-o", markersize=4)
    axes[0].set_xlabel("Federation Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Convergence Curve: Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss
    axes[1].plot(history["round"], history["eval_loss"], "r-o",
                 markersize=4, label="Eval Loss")
    axes[1].plot(history["round"], history["avg_train_loss"], "g--s",
                 markersize=4, label="Train Loss")
    axes[1].set_xlabel("Federation Round")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Convergence Curve: Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Privacy Budget
    axes[2].plot(history["round"], history["avg_epsilon"], "m-^", markersize=4, label="Avg ε")
    if history.get("max_epsilon"):
        axes[2].plot(history["round"], history["max_epsilon"], "c--o", markersize=4, label="Max ε")
    axes[2].set_xlabel("Federation Round")
    axes[2].set_ylabel("Average ε")
    axes[2].set_title("Privacy Budget Consumption")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    suffix = "_final" if final else ""
    plt.savefig(output_dir / f"convergence_curves{suffix}.png", dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------- #
# Settings shared by every run reported in the paper.
#
# These are installed as parser defaults, so
#     python train_rap_sfl.py --preset <dataset>_<encoder>
# reproduces a paper configuration without any further flags. Explicit CLI
# arguments always override both COMMON_DEFAULTS and the chosen preset.
#
# stop_by_budget: training continues until every client has exhausted its own
# epsilon budget; --fed_rounds then acts as a reference length with a safety
# ceiling of fed_rounds x 3 (see the flag's help text).
# ---------------------------------------------------------------------------- #
COMMON_DEFAULTS = dict(
    num_clients=5,
    clients_per_round=0,          # 0 -> all clients participate every round
    dirichlet_alpha=0.5,          # non-IID partition
    fed_rounds=45,
    local_steps_per_round=6,
    eval_every=2,
    client_batch_size=32,
    eval_batch_size=32,
    seed=42,
    sac_buffer_size=2000,
    sac_batch_size=16,
    sac_updates_per_interval=1,
    alpha=0.2,
    gamma=0.99,
    reward_mode="soft_tanh",
    epsilon=8.0,
    delta=1e-5,
    stop_by_budget=True,
)


# ---------------------------------------------------------------------------- #
# Per-setting hyperparameter presets
#
# One entry per (dataset, encoder) pair reported in the paper. Values are the
# ones actually used for that setting; entries marked "not separately tuned"
# reuse the shared defaults and are listed as-is for transparency.
# ---------------------------------------------------------------------------- #
PRESETS = {
    "imdb_bert": {
        "max_seq_length": 256,
        "client_lr": 3e-5,
        "server_lr": 5e-5,
        "early_stop_patience": 14,
        "early_stop_min_delta": 0.0003,
        "rl_interval": 45,
        "sac_start_steps": 120,
        "actor_lr": 3e-5,
        "tau": 0.002,
        "default_norm_c": 5.0,
    },
    "imdb_distilbert": {
        "max_seq_length": 256,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 30,
        "sac_start_steps": 90,
        "actor_lr": 5e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "imdb_roberta": {
        "max_seq_length": 256,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 40,
        "sac_start_steps": 105,
        "actor_lr": 4e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "mnli_bert": {
        "max_seq_length": 128,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 30,
        "sac_start_steps": 90,
        "actor_lr": 5e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "mnli_distilbert": {
        "max_seq_length": 128,
        "client_lr": 3e-5,
        "server_lr": 5e-5,
        "early_stop_patience": 14,
        "early_stop_min_delta": 0.001,
        "rl_interval": 50,
        "sac_start_steps": 150,
        "actor_lr": 3e-5,
        "tau": 0.002,
        "default_norm_c": 6.0,
    },
    "mnli_roberta": {
        "max_seq_length": 128,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 35,
        "sac_start_steps": 100,
        "actor_lr": 5e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "qqp_bert": {
        "max_seq_length": 128,
        "client_lr": 3e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 16,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 35,
        "sac_start_steps": 110,
        "actor_lr": 4e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "qqp_distilbert": {
        "max_seq_length": 128,
        "client_lr": 5e-5,
        "server_lr": 1e-4,
        "early_stop_patience": 20,
        "early_stop_min_delta": 0.002,
        "rl_interval": 80,
        "sac_start_steps": 200,
        "actor_lr": 1e-5,
        "tau": 0.001,
        "default_norm_c": 8.0,
    },
    "qqp_roberta": {
        "max_seq_length": 128,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 14,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 35,
        "sac_start_steps": 100,
        "actor_lr": 4e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "sst2_bert": {
        "max_seq_length": 128,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 30,
        "sac_start_steps": 90,
        "actor_lr": 5e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "sst2_distilbert": {
        "max_seq_length": 128,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 30,
        "sac_start_steps": 90,
        "actor_lr": 5e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
    "sst2_roberta": {
        "max_seq_length": 128,
        "client_lr": 2e-5,
        "server_lr": 4e-5,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.0002,
        "rl_interval": 35,
        "sac_start_steps": 100,
        "actor_lr": 4e-5,
        "tau": 0.003,
        "default_norm_c": 4.0,
    },
}

# Why each preset differs from the shared defaults (tuning rationale).
PRESET_NOTES = {
    "imdb_bert": (
        "MAX_SEQ_LENGTH 256 for long reviews; RL slowed (RL_INTERVAL"
        "45) and clipping relaxed (C=5.0) to reduce DP+RL oscillation.",
    ),
    "imdb_distilbert": (
        "Shared defaults with MAX_SEQ_LENGTH 256 for long reviews;"
        "other values not separately tuned.",
    ),
    "imdb_roberta": (
        "RL_INTERVAL 35->40 and SAC_START_STEPS 90->105: longer warm-up"
        "keeps the agent from raising noise too early and stalling"
        "budget consumption.",
    ),
    "mnli_bert": "Shared defaults; not separately tuned.",
    "mnli_distilbert": (
        "Sentence-pair NLI with the 6-layer encoder: higher learning"
        "rates, relaxed early stopping (delta 1e-3), RL every 50 steps,"
        "C=6.0.",
    ),
    "mnli_roberta": (
        "RoBERTa schedule (RL_INTERVAL 35, SAC_START_STEPS 100) on top"
        "of the shared defaults.",
    ),
    "qqp_bert": (
        "Large corpus -> low sampling rate -> slow budget consumption;"
        "CLIENT_LR 2e-5->3e-5, EARLY_STOP_PATIENCE 14->16,"
        "SAC_START_STEPS 100->110.",
    ),
    "qqp_distilbert": (
        "Third tuning iteration. v2 policies collapsed (accuracy"
        "dropped to ~48%) because RL updated too often; v3 delays RL to"
        "every 80 steps, warms up for 200 steps on fixed noise, uses a"
        "conservative ACTOR_LR=1e-5, slow TAU=0.001, and relaxed C=8.0"
        "(a 6-layer encoder concentrates gradients).",
    ),
    "qqp_roberta": (
        "Shared defaults with the RoBERTa RL schedule; not separately"
        "tuned.",
    ),
    "sst2_bert": "Reference configuration; the other presets derive from it.",
    "sst2_distilbert": "Shared defaults; not separately tuned for this encoder.",
    "sst2_roberta": (
        "SAC_START_STEPS 90->100 and RL_INTERVAL 30->35 to damp early-"
        "round policy oscillation.",
    ),
}


# ============================================================================ #
# CLI entry point
# ============================================================================ #

def build_parser() -> argparse.ArgumentParser:
    """CLI definition. Defaults are filled in by apply_preset() at run time."""
    parser = argparse.ArgumentParser(
        description="RAP-SFL: Reinforcement-driven Adaptive Perturbation at the "
                    "Split Point of Split Federated Learning")

    parser.add_argument(
        "--preset", type=str, default="sst2_bert", choices=sorted(PRESETS),
        help=(
            "Per-(dataset, encoder) hyperparameter preset to load as defaults; "
            "see PRESETS / PRESET_NOTES. Any explicit flag below overrides it."
        ),
    )

    # Model & data
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased",
        choices=["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
        help="Pretrained model name; supports bert-base-uncased / roberta-base / distilbert-base-uncased",
    )
    parser.add_argument(
        "--task_name", type=str, default="sst2",
        choices=["sst2", "imdb", "qqp", "mnli"],
        help="Training task: sst2 / imdb / qqp / mnli (GLUE natural language inference, 3-way)",
    )
    parser.add_argument(
        "--num_labels", type=int, default=None,
        help="Number of classification labels; if not specified, inferred from task_name (2 for sst2/imdb/qqp, 3 for mnli)",
    )
    parser.add_argument("--max_seq_length", type=int, default=128)

    # Federated learning hyperparameters
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Total number of federated clients")
    parser.add_argument("--clients_per_round", type=int, default=0,
                        help="Number of clients participating in training per round; <=0 means all clients participate")
    parser.add_argument("--fed_rounds", type=int, default=50,
                        help="Total number of federated training rounds")
    parser.add_argument("--local_steps_per_round", type=int, default=10,
                        help="Local training steps per client per round")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5,
                        help="Dirichlet distribution parameter; smaller values mean a higher degree of non-IID (recommended 0.1~1.0)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N rounds")

    # Training hyperparameters
    parser.add_argument("--client_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--client_lr", type=float, default=3e-5)
    parser.add_argument("--server_lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=4,
                        help="Stop early after several consecutive evaluations without significant improvement on the validation set; <=0 disables it")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002,
                        help="Minimum accuracy increment regarded as a significant improvement")

    # RL hyperparameters
    parser.add_argument(
        "--num_blocks", type=int, default=None,
        help="Number of Transformer blocks (the RL action/state dimensions depend on this value); if not specified, inferred from the model config",
    )
    parser.add_argument("--rl_interval", type=int, default=10)
    parser.add_argument("--sac_start_steps", type=int, default=30)
    parser.add_argument("--sac_buffer_size", type=int, default=2000)
    parser.add_argument("--sac_batch_size", type=int, default=16)
    parser.add_argument("--sac_updates_per_interval", type=int, default=1)
    parser.add_argument("--actor_lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--reward_mode", type=str, default="log_ratio_asym")
    parser.add_argument("--disable_rl", action="store_true",
                        help="Disable RL and use a static noise policy")
    parser.add_argument(
        "--ablation_mode", type=str, default="none",
        choices=["none", "no_alf", "no_sana"],
        help=(
            "RAP-SFL component ablation: none=full method; "
            "no_alf=disable Adaptive Layer Fusion, use only the last layer; "
            "no_sana=disable Semantics-Aware Noise, use uniform clipping/noise and disable attention-based semantic modulation"
        ),
    )
    parser.add_argument(
        "--fedprox_mu", type=float, default=0.0,
        help="FedProx proximal-term coefficient μ; adds (μ/2)||θ-θ^t||^2 to the local objective (θ^t is the global encoder at the start of the round). 0 disables it",
    )
    parser.add_argument(
        "--scaffold", action="store_true",
        help="Enable SCAFFOLD control-variate gradient correction (Karimireddy et al., ICML 2020)",
    )
    parser.add_argument(
        "--scaffold_lr", type=float, default=1.0,
        help="Scaling coefficient for the SCAFFOLD gradient correction; the control variates are already of the same order as the gradients in parameter-drift space",
    )
    parser.add_argument(
        "--dp_fedavg_sigma", type=float, default=0.0,
        help="DP-FedAvg: standard deviation σ of the Gaussian noise added to the global model after aggregation. 0 disables it (no CDP protection)",
    )

    # Privacy hyperparameters
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--default_norm_c", type=float, default=4.0)
    parser.add_argument("--default_noise_multiplier", type=float, default=None,
                        help="Default noise multiplier; if not specified, estimated automatically from the target epsilon")
    parser.add_argument("--no_dp", action="store_true",
                        help="Disable differential privacy entirely (for the Non-Private baseline comparison)")
    parser.add_argument(
        "--stop_by_budget", action="store_true",
        help=(
            "Budget-driven stopping mode: training continues until every client has exhausted its own ε budget. "
            "fed_rounds is only a reference length; the actual safe upper bound is fed_rounds×3. "
            "When a client exhausts its budget early and is frozen, the remaining clients keep training until each reaches its own ε limit, "
            "and the early-stopping counter is reset after each new freeze event to prevent false triggers caused by the change in training dynamics."
        ),
    )

    # Other
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--alf_logging", action="store_true",
                        help="Enable ALF layer-weight CSV logging (produces the per-layer fusion weights plotted in the paper)")
    parser.add_argument(
        "--alf_strategy", type=str, default="rap_sfl",
        choices=["rap_sfl", "last_only", "first_only", "middle_only", "uniform"],
        help=(
            "ALF analysis variants: rap_sfl=full RL-driven ALF; "
            "last_only/first_only/middle_only=fixed single layer; "
            "uniform=uniform fusion over all layers (w_l=1/L). "
            "For any value other than rap_sfl, RL exploration of layer_weights is disabled automatically."
        ),
    )

    return parser


def apply_preset(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Two-pass parse: resolve --preset, then install COMMON_DEFAULTS and that
    preset's values as parser defaults so explicit CLI flags still win."""
    pre, _ = parser.parse_known_args(argv)
    parser.set_defaults(**COMMON_DEFAULTS)
    parser.set_defaults(**PRESETS[pre.preset])
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = apply_preset(build_parser())
    train_rap_sfl(args)
