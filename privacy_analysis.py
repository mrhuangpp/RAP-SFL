#!/usr/bin/env python3
"""
Privacy Attack Evaluation for RAP-SFL
=============================================
Adapts attacks to the Federated Split Learning scenario where:
  - Clients send perturbed [CLS] embeddings (smashed data) to the server
  - The server is honest-but-curious and attempts to recover client input text

Attack methods:
  1. Embedding Inversion (NN Accuracy + Semantic Similarity)
  2. Membership Inference Attack (Entropy / Confidence ASR)
  3. SIP Forward Inversion Attack (ROUGE-L / Token F1)         -- adapted from BiSR
  4. NaMoE Noise-Adaptive Inversion (ROUGE-L / Token F1)       -- adapted from BiSR
  5. Mutual Information I(X;Z) via MINE                        -- Belghazi et al. 2018

References:
  BiSR: Chen et al., "Unveiling the Vulnerability of Private Fine-Tuning in
    Split-Based Frameworks for Large Language Models", CCS 2024.
  MINE: Belghazi et al., "Mutual Information Neural Estimation", ICML 2018.
"""

import argparse
import json
import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from train_rap_sfl import (
    ClientEncoder,
    ServerClassifier,
    TASK_CONFIGS,
    SUPPORTED_MODELS,
)
from rap_sfl_components import (
    SharedStateEncoder,
    SACActor,
    compute_embedding_norm_stats,
    build_state_vector,
    parse_action,
    apply_spah,
    compute_attention_entropy,
)

logger = logging.getLogger(__name__)


# ============================================================================ #
# SIP Inversion Models (adapted for encoder-only classification LLMs)
# ============================================================================ #

class GRUInverter(nn.Module):
    """
    GRU-based inversion model that maps smashed data (perturbed [CLS] or
    full-sequence embeddings) back to token logits over the vocabulary.

    In our RAP-SFL scenario, the client sends a perturbed [CLS] vector
    (shape [B, hidden_size]) to the server.  The attacker attempts to
    reconstruct the full input token sequence from this single vector.

    Architecture:
        [CLS] vector -> expand to seq_len -> GRU -> MLP -> vocab logits
    """

    def __init__(self, hidden_size, vocab_size, max_seq_len=128,
                 gru_hidden=256, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.expand = nn.Linear(hidden_size, hidden_size * max_seq_len)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, vocab_size)

    def forward(self, cls_vec):
        """
        Args:
            cls_vec: [B, hidden_size] — the smashed data (perturbed CLS)
        Returns:
            logits: [B, max_seq_len, vocab_size]
        """
        B = cls_vec.size(0)
        expanded = self.expand(cls_vec)  # [B, hidden_size * seq_len]
        expanded = expanded.view(B, self.max_seq_len, self.hidden_size)
        hidden, _ = self.gru(expanded)
        hidden = self.dropout(hidden)
        return self.head(hidden)


class FullSeqGRUInverter(nn.Module):
    """
    GRU-based inversion model for full-sequence smashed data.
    Used when the split point transmits the entire sequence embedding
    (shape [B, seq_len, hidden_size]) instead of just [CLS].
    """

    def __init__(self, hidden_size, vocab_size, gru_hidden=256, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, vocab_size)

    def forward(self, seq_embedding):
        """
        Args:
            seq_embedding: [B, seq_len, hidden_size]
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        hidden, _ = self.gru(seq_embedding)
        hidden = self.dropout(hidden)
        return self.head(hidden)


class NaMoEInverter(nn.Module):
    """
    Noise-adaptive Mixture of Experts inversion model.
    Each expert is a GRU trained on a specific noise level.
    A gating network selects the appropriate expert based on the input.
    """

    def __init__(self, hidden_size, vocab_size, max_seq_len=128,
                 expert_noise_scales=None, gru_hidden=256, dropout=0.1):
        super().__init__()
        if expert_noise_scales is None:
            expert_noise_scales = [0.0, 0.5, 1.0, 2.0]
        self.expert_noise_scales = expert_noise_scales
        self.num_experts = len(expert_noise_scales)
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        self.expand = nn.Linear(hidden_size, hidden_size * max_seq_len)
        self.experts = nn.ModuleList([
            nn.GRU(input_size=hidden_size, hidden_size=gru_hidden,
                   batch_first=True)
            for _ in expert_noise_scales
        ])
        self.dropout = nn.Dropout(dropout)

        self.gating_mlp = nn.Linear(hidden_size, gru_hidden)
        self.gating_out = nn.Linear(gru_hidden, self.num_experts)

        self.head = nn.Linear(gru_hidden, vocab_size)

    def forward(self, cls_vec):
        B = cls_vec.size(0)
        expanded = self.expand(cls_vec).view(B, self.max_seq_len, self.hidden_size)

        exp_outputs = []
        for expert in self.experts:
            h, _ = expert(expanded)
            h = self.dropout(h)
            exp_outputs.append(h)
        exp_stack = torch.stack(exp_outputs, dim=1)  # [B, num_experts, seq, gru_h]

        gate_h = F.relu(self.gating_mlp(cls_vec))  # [B, gru_h]
        gate_logits = self.gating_out(gate_h)       # [B, num_experts]
        weights = F.softmax(gate_logits, dim=-1)    # [B, num_experts]

        # Weighted sum: [B, seq, gru_h]
        output = torch.einsum('besd,be->bsd', exp_stack, weights)
        return self.head(output)

    def train_experts_independently(self, expert_idx, expanded_input, target_ids):
        """Train a single expert on data with a specific noise level."""
        h, _ = self.experts[expert_idx](expanded_input)
        h = self.dropout(h)
        logits = self.head(h)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
        )
        return loss, logits


# ============================================================================ #
# Victim Model Wrapper for RAP-SFL
# ============================================================================ #

class RapSFLVictim(nn.Module):
    """
    Wraps the trained RAP-SFL model to produce smashed data (perturbed
    embeddings) as the server would observe during training/inference.
    """

    def __init__(self, model_name, model_type, encoder_state, server_state,
                 args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.victim_kind = getattr(args, "victim_kind", "rap_sfl")

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = args.num_labels
        config.output_attentions = True
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config)
        self.encoder = ClientEncoder(base_model, model_type).to(device)
        del base_model

        self.encoder.load_state_dict(encoder_state)
        self.encoder.eval()

        self.server_classifier = ServerClassifier(
            config.hidden_size, args.num_labels).to(device)
        self.server_classifier.load_state_dict(server_state)
        self.server_classifier.eval()

        self.config = config
        self.hidden_size = config.hidden_size
        self.model_type = model_type

        # RL components (for generating realistic noise)
        self._load_rl_components(args)

    def _load_rl_components(self, args):
        """Load RL policy for generating realistic DP noise."""
        num_layers = self.config.num_hidden_layers + 1
        num_blocks = args.num_blocks
        action_dim = num_layers + num_blocks + 1 + 1
        state_dim = 15 + num_blocks

        self.rl_encoder = SharedStateEncoder(state_dim, hidden=64, embed_dim=64).to(self.device)
        self.rl_actor = SACActor(64, action_dim, hidden=64).to(self.device)
        self.num_layers = num_layers
        self.num_blocks = num_blocks

        from rap_sfl_components import RunningMeanStd
        self.state_norm = RunningMeanStd(state_dim, device=self.device)

        rl_path = os.path.join(args.model_path, "rl_policy.pt") if hasattr(args, 'model_path') else None
        if rl_path and os.path.exists(rl_path):
            ckpt = torch.load(rl_path, map_location=self.device)
            self.rl_actor.load_state_dict(ckpt['actor'])
            self.rl_encoder.load_state_dict(ckpt['encoder'])
            if 'state_norm' in ckpt:
                self.state_norm.mean = ckpt['state_norm']['mean'].to(self.device)
                self.state_norm.var = ckpt['state_norm']['var'].to(self.device)
            logger.info(f"Loaded RL policy from {rl_path}")
        else:
            logger.warning("RL policy not found, using default noise parameters.")

        self.default_noise_multiplier = getattr(args, 'default_noise_multiplier', 1.0)
        self.default_norm_c = getattr(args, 'default_norm_c', 3.0)

    def get_smashed_data(self, input_ids, attention_mask, token_type_ids=None,
                         noise_multiplier=None):
        """
        Generate smashed data as the server would observe.

        Args:
            noise_multiplier: Override noise level.
                - If victim_kind == "non_private", LDP perturbation is bypassed
                  entirely and the clean CLS / last-layer output is returned,
                  regardless of this argument. This matches the assumption that
                  non-private baselines (Centralized / FedAvg / SplitFed) do
                  not inject any client-side LDP noise.
                - If victim_kind == "dp_fedavg", the server-side aggregated
                  weights already absorbed Gaussian noise during training, so
                  again no extra forward LDP noise is added at attack time.
                - If victim_kind == "rap_sfl" and noise_multiplier is None, the
                  RL policy is used to derive (layer_weights, clip, noise).
                  If noise_multiplier is given, the static last-layer fallback
                  is used with the supplied scalar — only meaningful when the
                  attacker explicitly probes a specific noise level (e.g.
                  NaMoE expert training).
        Returns:
            cls_vec: [B, hidden_size] — (perturbed) CLS embedding (smashed data)
            full_seq: [B, seq_len, hidden_size] — full (perturbed) sequence
        """
        with torch.no_grad():
            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            if token_type_ids is not None and self.model_type == "bert":
                fwd_kwargs["token_type_ids"] = token_type_ids

            enc_out = self.encoder.encoder(**fwd_kwargs)
            hidden_states = enc_out.hidden_states
            attentions = enc_out.attentions
            layer_outputs = list(hidden_states)

            # Non-private / DP-FedAvg victims: no client-side LDP noise.
            # The attacker observes the clean smashed feature, which is
            # exactly what the SFL server sees in those baselines.
            if self.victim_kind in ("non_private", "dp_fedavg"):
                fused = hidden_states[-1]
                cls_vec = fused[:, 0, :]
                pooler = getattr(self.encoder.encoder, "pooler", None)
                if self.model_type in ("bert", "roberta") and pooler is not None:
                    cls_vec = pooler(fused)
                return cls_vec, fused

            if noise_multiplier is not None:
                nm = noise_multiplier
                num_layers = len(layer_outputs)
                lw = torch.zeros(num_layers, device=self.device)
                lw[-1] = 10.0
                lw[:-1] = -10.0
                lw = F.softmax(lw, dim=-1)
                nc = torch.ones(self.num_blocks, device=self.device) * self.default_norm_c
                attn_inf = 0.0
            else:
                norm_stats = compute_embedding_norm_stats(hidden_states[-1])
                attn_entropy = compute_attention_entropy(attentions)
                state = build_state_vector(
                    embedding_norm_stats=norm_stats,
                    utility=0.0,
                    spent_eps=self.args.epsilon,
                    batch_loss=0.0,
                    num_blocks=self.num_blocks,
                    target_epsilon=self.args.epsilon,
                    prev_loss=0.0,
                    gradient_norm=0.0,
                    attention_entropy=attn_entropy,
                    device=self.device,
                )
                norm_state = self.state_norm.normalize(state.unsqueeze(0))
                enc_state = self.rl_encoder(norm_state)
                action_mu, _ = self.rl_actor(enc_state)
                action_vec = action_mu.squeeze(0)

                lw, nc, nm, attn_inf = parse_action(
                    action_vector=action_vec,
                    num_layers=self.num_layers,
                    num_blocks=self.num_blocks,
                    progress=1.0,
                    eps_ratio=1.0,
                    min_noise_bound=self.default_noise_multiplier,
                )

            fused = apply_spah(
                layer_outputs=layer_outputs,
                layer_weights=lw,
                norm_constants=nc,
                noise_multiplier=nm,
                device=self.device,
                attention_outputs=attentions,
                attn_influence=attn_inf,
            )

            cls_vec = fused[:, 0, :]
            pooler = getattr(self.encoder.encoder, "pooler", None)
            if self.model_type in ("bert", "roberta") and pooler is not None:
                cls_vec = pooler(fused)

            return cls_vec, fused

    def get_clean_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        """Get clean (unperturbed) embeddings for comparison."""
        with torch.no_grad():
            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            if token_type_ids is not None and self.model_type == "bert":
                fwd_kwargs["token_type_ids"] = token_type_ids
            enc_out = self.encoder.encoder(**fwd_kwargs)
            return enc_out.hidden_states[-1]


# ============================================================================ #
# Evaluation Metrics
# ============================================================================ #

def compute_rouge_l(pred_tokens, ref_tokens):
    """Compute ROUGE-L F1 between two token sequences."""
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def compute_token_f1(pred_tokens, ref_tokens):
    """Compute token-level F1 (set-based overlap)."""
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    if len(pred_set) == 0 or len(ref_set) == 0:
        return 0.0
    overlap = len(pred_set & ref_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================================ #
# Attack 1: Embedding Inversion
# ============================================================================ #

def run_embedding_inversion(victim, tokenizer, eval_data, text_fields,
                            device, max_samples=200, max_seq_len=128):
    """Nearest-Neighbor token inversion + Semantic Similarity on smashed data."""
    logger.info("=" * 50)
    logger.info("Attack 1: Embedding Inversion (NN + Cosine Sim)")
    logger.info("=" * 50)

    word_embeddings = victim.encoder.encoder.get_input_embeddings().weight  # [V, H]

    correct_tokens, total_tokens = 0, 0
    total_cosine_sim = 0.0

    for idx in tqdm(range(min(max_samples, len(eval_data))), desc="EmbInversion"):
        item = eval_data[idx]
        if len(text_fields) == 1:
            text = item[text_fields[0]]
        else:
            text = item[text_fields[0]] + " " + item[text_fields[1]]

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_seq_len, padding="max_length").to(device)
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        ttids = inputs.get("token_type_ids", None)

        _, full_perturbed = victim.get_smashed_data(input_ids, mask, ttids)
        clean = victim.get_clean_embeddings(input_ids, mask, ttids)

        cos_sim = F.cosine_similarity(full_perturbed, clean, dim=-1)  # [1, seq]
        active_cos = (cos_sim * mask).sum().item()
        active_count = mask.sum().item()
        total_cosine_sim += active_cos

        seq_len = input_ids.size(1)
        for i in range(seq_len):
            if mask[0, i] == 0:
                continue
            vec = full_perturbed[0, i]
            sim = F.cosine_similarity(vec.unsqueeze(0), word_embeddings, dim=1)
            if torch.argmax(sim).item() == input_ids[0, i].item():
                correct_tokens += 1
            total_tokens += 1

    nn_acc = correct_tokens / max(total_tokens, 1)
    sem_sim = total_cosine_sim / max(total_tokens, 1)

    logger.info(f"  NN Accuracy:        {nn_acc:.4f}")
    logger.info(f"  Semantic Similarity: {sem_sim:.4f}")
    return {"nn_accuracy": nn_acc, "semantic_similarity": sem_sim}


# ============================================================================ #
# Attack 2: Membership Inference Attack (original, adapted)
# ============================================================================ #

def run_mia(victim, tokenizer, train_data, test_data, text_fields,
            device, max_samples=500, max_seq_len=128):
    """Entropy & Confidence based MIA on the RAP-SFL model."""
    logger.info("=" * 50)
    logger.info("Attack 2: Membership Inference Attack (MIA)")
    logger.info("=" * 50)

    def collect_metrics(data, desc):
        entropies, confidences = [], []
        for idx in tqdm(range(min(max_samples, len(data))), desc=desc):
            item = data[idx]
            if len(text_fields) == 1:
                text = item[text_fields[0]]
            else:
                text = item[text_fields[0]] + " " + item[text_fields[1]]
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_seq_len).to(device)
            ttids = inputs.get("token_type_ids", None)
            cls_vec, _ = victim.get_smashed_data(
                inputs["input_ids"], inputs["attention_mask"], ttids)
            logits = victim.server_classifier(cls_vec)
            probs = F.softmax(logits, dim=-1)
            confidences.append(torch.max(probs).item())
            entropies.append((-torch.sum(probs * torch.log(probs + 1e-9))).item())
        return entropies, confidences

    mem_ent, mem_conf = collect_metrics(train_data, "MIA-Members")
    non_ent, non_conf = collect_metrics(test_data, "MIA-NonMembers")

    def calc_asr(mem_scores, non_scores, reverse=False):
        all_scores = mem_scores + non_scores
        all_labels = [1] * len(mem_scores) + [0] * len(non_scores)
        thresholds = sorted(all_scores)[::max(1, len(all_scores) // 50)]
        best_acc = 0.0
        for t in thresholds:
            preds = [1 if (s < t if reverse else s > t) else 0 for s in all_scores]
            best_acc = max(best_acc, accuracy_score(all_labels, preds))
        return best_acc

    asr_ent = calc_asr(mem_ent, non_ent, reverse=True)
    asr_conf = calc_asr(mem_conf, non_conf, reverse=False)

    logger.info(f"  MIA ASR (Entropy):    {asr_ent:.4f}")
    logger.info(f"  MIA ASR (Confidence): {asr_conf:.4f}")
    return {"asr_entropy": asr_ent, "asr_confidence": asr_conf}


# ============================================================================ #
# Attack 3: SIP Forward Inversion (adapted from BiSR for RAP-SFL)
# ============================================================================ #

def train_sip_inverter(victim, tokenizer, aux_data, text_fields, device, args):
    """
    Train an SIP inversion model on auxiliary data.
    The attacker uses the pre-trained Bottom model (semi-white-box access)
    to generate smashed data, then trains a decoder to invert it.
    """
    logger.info("Training SIP Inversion Model...")

    max_seq_len = args.max_seq_len
    inverter = GRUInverter(
        hidden_size=victim.hidden_size,
        vocab_size=victim.config.vocab_size,
        max_seq_len=max_seq_len,
        gru_hidden=256,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(inverter.parameters(), lr=args.sip_lr,
                                  weight_decay=1e-5)
    num_samples = len(aux_data) if args.sip_train_samples <= 0 else \
        min(args.sip_train_samples, len(aux_data))

    for epoch in range(args.sip_epochs):
        inverter.train()
        total_loss = 0.0
        indices = list(range(num_samples))
        random.shuffle(indices)

        pbar = tqdm(range(0, num_samples, args.sip_batch_size),
                    desc=f"SIP Epoch {epoch + 1}/{args.sip_epochs}")
        for start in pbar:
            batch_indices = indices[start:start + args.sip_batch_size]
            if len(batch_indices) == 0:
                continue

            batch_texts = []
            for i in batch_indices:
                item = aux_data[i]
                if len(text_fields) == 1:
                    batch_texts.append(item[text_fields[0]])
                else:
                    batch_texts.append(
                        item[text_fields[0]] + " " + item[text_fields[1]])

            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                            max_length=max_seq_len, padding="max_length").to(device)
            input_ids = enc["input_ids"]
            mask = enc["attention_mask"]
            ttids = enc.get("token_type_ids", None)

            with torch.no_grad():
                cls_vec, _ = victim.get_smashed_data(
                    input_ids, mask, ttids,
                    noise_multiplier=victim.default_noise_multiplier)

            logits = inverter(cls_vec)  # [B, seq_len, vocab]

            target = input_ids.clone()
            target[mask == 0] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1),
                ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        logger.info(f"  SIP Epoch {epoch + 1} avg loss: "
                    f"{total_loss / max(num_samples // args.sip_batch_size, 1):.4f}")

    return inverter


def evaluate_sip(inverter, victim, tokenizer, eval_data, text_fields,
                 device, max_samples=200, max_seq_len=128):
    """Evaluate SIP inversion attack using ROUGE-L and Token F1."""
    logger.info("Evaluating SIP Inversion Attack...")
    inverter.eval()

    rouge_scores, token_f1_scores = [], []

    for idx in tqdm(range(min(max_samples, len(eval_data))), desc="SIP-Eval"):
        item = eval_data[idx]
        if len(text_fields) == 1:
            text = item[text_fields[0]]
        else:
            text = item[text_fields[0]] + " " + item[text_fields[1]]

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, padding="max_length").to(device)
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"]
        ttids = enc.get("token_type_ids", None)

        with torch.no_grad():
            cls_vec, _ = victim.get_smashed_data(input_ids, mask, ttids)
            logits = inverter(cls_vec)

        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
        ref_ids = input_ids[0].cpu().tolist()
        mask_list = mask[0].cpu().tolist()

        pred_active = [t for t, m in zip(pred_ids, mask_list) if m == 1]
        ref_active = [t for t, m in zip(ref_ids, mask_list) if m == 1]

        rouge_scores.append(compute_rouge_l(pred_active, ref_active))
        token_f1_scores.append(compute_token_f1(pred_active, ref_active))

    avg_rouge = np.mean(rouge_scores)
    avg_f1 = np.mean(token_f1_scores)
    logger.info(f"  SIP ROUGE-L F1:  {avg_rouge:.4f}")
    logger.info(f"  SIP Token F1:    {avg_f1:.4f}")
    return {"rouge_l": avg_rouge, "token_f1": avg_f1}


def run_sip_attack(victim, tokenizer, train_data, eval_data, text_fields,
                   device, args):
    """Full SIP attack pipeline: train on aux data, evaluate on eval data."""
    logger.info("=" * 50)
    logger.info("Attack 3: SIP Forward Inversion Attack")
    logger.info("=" * 50)

    inverter = train_sip_inverter(
        victim, tokenizer, train_data, text_fields, device, args)
    results = evaluate_sip(
        inverter, victim, tokenizer, eval_data, text_fields, device,
        max_samples=args.eval_samples, max_seq_len=args.max_seq_len)
    return results


# ============================================================================ #
# Attack 4: NaMoE Noise-Adaptive Inversion
# ============================================================================ #

def train_namoe_inverter(victim, tokenizer, aux_data, text_fields, device, args):
    """
    Train a NaMoE inversion model with multiple noise-aware experts.

    Phase 1: Train each expert independently on data perturbed with its
             designated noise level.
    Phase 2: Freeze experts, train gating network on randomly-noised data.
    Phase 3: Joint fine-tuning of all parameters.
    """
    logger.info("Training NaMoE Inversion Model...")

    noise_scales = [float(s) for s in args.namoe_noise_scales.split(",")]
    max_seq_len = args.max_seq_len

    namoe = NaMoEInverter(
        hidden_size=victim.hidden_size,
        vocab_size=victim.config.vocab_size,
        max_seq_len=max_seq_len,
        expert_noise_scales=noise_scales,
        gru_hidden=256,
        dropout=0.1,
    ).to(device)

    num_samples = len(aux_data) if args.sip_train_samples <= 0 else \
        min(args.sip_train_samples, len(aux_data))

    def get_batch(batch_indices):
        batch_texts = []
        for i in batch_indices:
            item = aux_data[i]
            if len(text_fields) == 1:
                batch_texts.append(item[text_fields[0]])
            else:
                batch_texts.append(
                    item[text_fields[0]] + " " + item[text_fields[1]])
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, padding="max_length").to(device)
        return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids", None)

    # Phase 1: Expert training
    logger.info("  Phase 1: Training experts independently...")
    for param in namoe.gating_mlp.parameters():
        param.requires_grad = False
    for param in namoe.gating_out.parameters():
        param.requires_grad = False

    opt1 = torch.optim.Adam(
        [p for p in namoe.parameters() if p.requires_grad], lr=args.sip_lr)

    for epoch in range(args.namoe_expert_epochs):
        namoe.train()
        indices = list(range(num_samples))
        random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        for start in range(0, num_samples, args.sip_batch_size):
            batch_idx = indices[start:start + args.sip_batch_size]
            if len(batch_idx) == 0:
                continue
            input_ids, mask, ttids = get_batch(batch_idx)
            target = input_ids.clone()
            target[mask == 0] = -100

            opt1.zero_grad()
            loss = torch.tensor(0.0, device=device)
            B = input_ids.size(0)

            for ei, ns in enumerate(noise_scales):
                with torch.no_grad():
                    cls_vec, _ = victim.get_smashed_data(
                        input_ids, mask, ttids, noise_multiplier=ns)
                expanded = namoe.expand(cls_vec).view(B, max_seq_len, victim.hidden_size)
                exp_loss, _ = namoe.train_experts_independently(
                    ei, expanded, target)
                loss = loss + exp_loss

            loss.backward()
            opt1.step()
            total_loss += loss.item()
            steps += 1

        logger.info(f"    Expert Epoch {epoch + 1} avg loss: "
                    f"{total_loss / max(steps, 1):.4f}")

    # Phase 2: Gating training
    logger.info("  Phase 2: Training gating network...")
    for param in namoe.experts.parameters():
        param.requires_grad = False
    for param in namoe.gating_mlp.parameters():
        param.requires_grad = True
    for param in namoe.gating_out.parameters():
        param.requires_grad = True

    opt2 = torch.optim.Adam(
        [p for p in namoe.parameters() if p.requires_grad], lr=args.sip_lr)

    for epoch in range(args.namoe_gating_epochs):
        namoe.train()
        indices = list(range(num_samples))
        random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        for start in range(0, num_samples, args.sip_batch_size):
            batch_idx = indices[start:start + args.sip_batch_size]
            if len(batch_idx) == 0:
                continue
            input_ids, mask, ttids = get_batch(batch_idx)
            target = input_ids.clone()
            target[mask == 0] = -100

            random_ns = random.choice(noise_scales)
            with torch.no_grad():
                cls_vec, _ = victim.get_smashed_data(
                    input_ids, mask, ttids, noise_multiplier=random_ns)

            logits = namoe(cls_vec)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1),
                ignore_index=-100)

            opt2.zero_grad()
            loss.backward()
            opt2.step()
            total_loss += loss.item()
            steps += 1

        logger.info(f"    Gating Epoch {epoch + 1} avg loss: "
                    f"{total_loss / max(steps, 1):.4f}")

    # Phase 3: Joint fine-tuning
    logger.info("  Phase 3: Joint fine-tuning...")
    for param in namoe.parameters():
        param.requires_grad = True

    opt3 = torch.optim.Adam(namoe.parameters(), lr=args.sip_lr * 0.5)

    for epoch in range(args.namoe_ft_epochs):
        namoe.train()
        indices = list(range(num_samples))
        random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        for start in range(0, num_samples, args.sip_batch_size):
            batch_idx = indices[start:start + args.sip_batch_size]
            if len(batch_idx) == 0:
                continue
            input_ids, mask, ttids = get_batch(batch_idx)
            target = input_ids.clone()
            target[mask == 0] = -100

            random_ns = random.choice(noise_scales + [0.0])
            with torch.no_grad():
                cls_vec, _ = victim.get_smashed_data(
                    input_ids, mask, ttids, noise_multiplier=random_ns)

            logits = namoe(cls_vec)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1),
                ignore_index=-100)

            opt3.zero_grad()
            loss.backward()
            opt3.step()
            total_loss += loss.item()
            steps += 1

        logger.info(f"    FT Epoch {epoch + 1} avg loss: "
                    f"{total_loss / max(steps, 1):.4f}")

    return namoe


def run_namoe_attack(victim, tokenizer, train_data, eval_data, text_fields,
                     device, args):
    """Full NaMoE attack pipeline."""
    logger.info("=" * 50)
    logger.info("Attack 4: NaMoE Noise-Adaptive Inversion Attack")
    logger.info("=" * 50)

    namoe = train_namoe_inverter(
        victim, tokenizer, train_data, text_fields, device, args)

    logger.info("Evaluating NaMoE Inversion Attack...")
    namoe.eval()

    rouge_scores, token_f1_scores = [], []
    max_seq_len = args.max_seq_len

    for idx in tqdm(range(min(args.eval_samples, len(eval_data))),
                    desc="NaMoE-Eval"):
        item = eval_data[idx]
        if len(text_fields) == 1:
            text = item[text_fields[0]]
        else:
            text = item[text_fields[0]] + " " + item[text_fields[1]]

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, padding="max_length").to(device)
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"]
        ttids = enc.get("token_type_ids", None)

        with torch.no_grad():
            cls_vec, _ = victim.get_smashed_data(input_ids, mask, ttids)
            logits = namoe(cls_vec)

        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
        ref_ids = input_ids[0].cpu().tolist()
        mask_list = mask[0].cpu().tolist()

        pred_active = [t for t, m in zip(pred_ids, mask_list) if m == 1]
        ref_active = [t for t, m in zip(ref_ids, mask_list) if m == 1]

        rouge_scores.append(compute_rouge_l(pred_active, ref_active))
        token_f1_scores.append(compute_token_f1(pred_active, ref_active))

    avg_rouge = np.mean(rouge_scores)
    avg_f1 = np.mean(token_f1_scores)
    logger.info(f"  NaMoE ROUGE-L F1:  {avg_rouge:.4f}")
    logger.info(f"  NaMoE Token F1:    {avg_f1:.4f}")
    return {"rouge_l": avg_rouge, "token_f1": avg_f1}


# ============================================================================ #
# Attack 5: Mutual Information Estimation  I(X; Z)
# ============================================================================ #

class MINENetwork(nn.Module):
    """
    MINE (Mutual Information Neural Estimation, Belghazi et al. 2018).
    T_θ(x, z) is a discriminator network that maximizes the Donsker-Varadhan lower bound:
        I(X; Z) >= E_joint[T] - log(E_marginal[exp(T)])
    """
    def __init__(self, x_dim: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1))


def _get_input_features(input_ids: torch.Tensor,
                        word_embeddings: torch.Tensor) -> torch.Tensor:
    """Map discrete token ids into the continuous embedding space and take mean pooling as the sentence representation."""
    with torch.no_grad():
        emb = word_embeddings[input_ids]  # [B, seq, dim]
        return emb.mean(dim=1)  # [B, dim]


def run_mutual_information(victim, tokenizer, eval_data, text_fields,
                           device, max_samples=500, max_seq_len=128,
                           mine_epochs=100, mine_lr=1e-4, mine_hidden=256):
    """
    Estimate the mutual information I(X;Z) between the raw input X and the perturbed [CLS] vector Z received by the server, using MINE.

    Principle:
      - Representation of X: mean-pool the raw input tokens through the word-embedding layer to get the sentence vector x ∈ R^d
      - Representation of Z: the [CLS] vector z ∈ R^d sent to the server after RAP-SFL perturbation
      - MINE trains a statistic network T_θ(x, z) that maximizes the Donsker-Varadhan lower bound:
            I(X;Z) >= E_{p(x,z)}[T_θ] - log( E_{p(x)p(z)}[e^{T_θ}] )
      - Joint samples (x, z) come from the same example
      - Marginal samples (x, z') are obtained by shuffling z

    Interpretation:
      - I(X;Z) ≈ 0 → the perturbed Z carries almost no information about the raw X → good privacy protection
      - Large I(X;Z) → Z still contains a large amount of raw information → severe leakage
    """
    logger.info("=" * 50)
    logger.info("Metric: Mutual Information I(X;Z) via MINE")
    logger.info("=" * 50)

    word_emb = victim.encoder.encoder.get_input_embeddings().weight  # [V, H]
    x_dim = word_emb.shape[1]
    z_dim = victim.hidden_size

    # ---- Collect (x, z) pairs ---- #
    x_list, z_list = [], []
    n = min(max_samples, len(eval_data))

    for idx in tqdm(range(n), desc="MI-collect"):
        item = eval_data[idx]
        if len(text_fields) == 1:
            text = item[text_fields[0]]
        else:
            text = item[text_fields[0]] + " " + item[text_fields[1]]

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_seq_len, padding="max_length").to(device)
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        ttids = inputs.get("token_type_ids", None)

        x_feat = _get_input_features(input_ids, word_emb)  # [1, x_dim]
        cls_vec, _ = victim.get_smashed_data(input_ids, mask, ttids)  # [1, z_dim]

        x_list.append(x_feat.cpu())
        z_list.append(cls_vec.cpu())

    X = torch.cat(x_list, dim=0)  # [N, x_dim]
    Z = torch.cat(z_list, dim=0)  # [N, z_dim]

    # Standardize the features for MINE training
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-8)
    Z = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-8)

    X = X.to(device)
    Z = Z.to(device)

    # ---- MINE training ---- #
    mine_net = MINENetwork(x_dim, z_dim, hidden=mine_hidden).to(device)
    optimizer = torch.optim.Adam(mine_net.parameters(), lr=mine_lr)
    N = X.shape[0]
    batch_size = min(128, N)

    mi_history = []
    ema_mi = 0.0

    for epoch in range(mine_epochs):
        perm = torch.randperm(N)
        epoch_mi = []

        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            x_b = X[idx]
            z_b = Z[idx]

            # Joint samples T(x, z)
            t_joint = mine_net(x_b, z_b)

            # Marginal samples: shuffle z (breaks the pairing)
            z_shuffle = z_b[torch.randperm(batch_size)]
            t_marginal = mine_net(x_b, z_shuffle)

            # MINE loss: -( E[T_joint] - log(E[exp(T_marginal)]) )
            # Use the EMA-based log-sum-exp trick to avoid gradient bias (Belghazi et al. 2018)
            et = torch.exp(t_marginal)
            mi_lb = t_joint.mean() - torch.log(et.mean() + 1e-8)
            loss = -mi_lb

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mine_net.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_mi.append(mi_lb.item())

        avg_mi = np.mean(epoch_mi) if epoch_mi else 0.0
        mi_history.append(avg_mi)
        ema_mi = 0.9 * ema_mi + 0.1 * avg_mi if epoch > 0 else avg_mi

        if (epoch + 1) % 20 == 0:
            logger.info(f"  MINE Epoch {epoch + 1}/{mine_epochs}: "
                        f"I(X;Z) ≈ {avg_mi:.4f} (EMA={ema_mi:.4f})")

    # Average the last 20 epochs as the final estimate
    tail = mi_history[-20:] if len(mi_history) >= 20 else mi_history
    final_mi = float(np.mean(tail))
    final_mi = max(final_mi, 0.0)  # MI is non-negative

    # Also compute the reference upper bound without protection: I(X; X_clean_cls)
    logger.info("  Computing reference MI (clean, no DP) ...")
    z_clean_list = []
    for idx in range(n):
        item = eval_data[idx]
        if len(text_fields) == 1:
            text = item[text_fields[0]]
        else:
            text = item[text_fields[0]] + " " + item[text_fields[1]]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_seq_len, padding="max_length").to(device)
        z_clean = victim.get_clean_embeddings(
            inputs["input_ids"], inputs["attention_mask"],
            inputs.get("token_type_ids", None))[:, 0, :]  # [1, H]
        z_clean_list.append(z_clean.cpu())

    Z_clean = torch.cat(z_clean_list, dim=0)
    Z_clean = (Z_clean - Z_clean.mean(0, keepdim=True)) / (Z_clean.std(0, keepdim=True) + 1e-8)
    Z_clean = Z_clean.to(device)

    mine_clean = MINENetwork(x_dim, z_dim, hidden=mine_hidden).to(device)
    opt_clean = torch.optim.Adam(mine_clean.parameters(), lr=mine_lr)

    mi_clean_history = []
    for epoch in range(mine_epochs):
        perm = torch.randperm(N)
        epoch_mi_c = []
        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            x_b = X[idx]
            z_b = Z_clean[idx]
            t_joint = mine_clean(x_b, z_b)
            z_shuffle = z_b[torch.randperm(batch_size)]
            t_marginal = mine_clean(x_b, z_shuffle)
            et = torch.exp(t_marginal)
            mi_lb = t_joint.mean() - torch.log(et.mean() + 1e-8)
            loss = -mi_lb
            opt_clean.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mine_clean.parameters(), max_norm=5.0)
            opt_clean.step()
            epoch_mi_c.append(mi_lb.item())
        if epoch_mi_c:
            mi_clean_history.append(np.mean(epoch_mi_c))

    tail_clean = mi_clean_history[-20:] if len(mi_clean_history) >= 20 else mi_clean_history
    mi_clean = max(float(np.mean(tail_clean)), 0.0) if tail_clean else 0.0

    privacy_leakage = final_mi / (mi_clean + 1e-8) if mi_clean > 0 else 0.0

    logger.info(f"\n  === Mutual Information Results ===")
    logger.info(f"  I(X; Z_perturbed) = {final_mi:.4f}  (with DP)")
    logger.info(f"  I(X; Z_clean)     = {mi_clean:.4f}  (no DP, reference)")
    logger.info(f"  Privacy Leakage Ratio = {privacy_leakage:.4f}  "
                f"(perturbed / clean, lower = better)")

    return {
        "mi_perturbed": final_mi,
        "mi_clean": mi_clean,
        "privacy_leakage_ratio": privacy_leakage,
    }


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="RAP-SFL Privacy Attack Evaluation")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained RAP-SFL output directory "
                             "(containing best_encoder.pt, best_classifier.pt, "
                             "and optionally rl_policy.pt)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--task_name", type=str, default="sst2",
                        choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--num_labels", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--attacks", type=str, default="all",
                        help="Comma-separated attacks: "
                             "emb_inv,mia,sip,namoe,mi or 'all'")

    parser.add_argument("--mine_epochs", type=int, default=100,
                        help="Number of MINE training epochs (mutual information estimation)")
    parser.add_argument("--mine_lr", type=float, default=1e-4,
                        help="Learning rate of the MINE discriminator network")
    parser.add_argument("--mine_hidden", type=int, default=256,
                        help="Hidden dimension of the MINE network")

    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--mia_samples", type=int, default=500)

    parser.add_argument("--sip_lr", type=float, default=1e-3)
    parser.add_argument("--sip_epochs", type=int, default=20,
                        help="SIP training epochs (BiSR paper: 20)")
    parser.add_argument("--sip_batch_size", type=int, default=6,
                        help="SIP batch size (BiSR paper: 6)")
    parser.add_argument("--sip_train_samples", type=int, default=0,
                        help="SIP training samples, 0=use all available data "
                             "(BiSR paper: full training set)")

    parser.add_argument("--namoe_noise_scales", type=str,
                        default="0.0,0.3,0.6,1.0,1.5,2.0,4.0",
                        help="Comma-separated noise scales for NaMoE experts")
    parser.add_argument("--namoe_expert_epochs", type=int, default=20,
                        help="NaMoE expert training epochs (BiSR paper: 20)")
    parser.add_argument("--namoe_gating_epochs", type=int, default=15,
                        help="NaMoE gating training epochs (BiSR paper: 15)")
    parser.add_argument("--namoe_ft_epochs", type=int, default=4,
                        help="NaMoE joint fine-tune epochs (BiSR paper: 4)")

    parser.add_argument("--default_noise_multiplier", type=float, default=1.0)
    parser.add_argument("--default_norm_c", type=float, default=3.0)
    parser.add_argument("--victim_kind", type=str, default="rap_sfl",
                        choices=["non_private", "dp_fedavg", "rap_sfl"],
                        help="Tells the attacker how to materialize smashed "
                             "data: 'non_private' = clean encoder output "
                             "(no LDP, used for Centralized/FedAvg/SplitFed); "
                             "'dp_fedavg' = clean encoder output for the "
                             "DP-FedAvg victim, whose Gaussian noise lives "
                             "in the aggregated weights, not in the forward "
                             "pass; 'rap_sfl' = drive the perturbation by "
                             "the trained SAC policy (or fall back to "
                             "last-layer static noise if rl_policy.pt is "
                             "missing).")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ------------------------------------------------------------------ #
    # Load task config and data
    # ------------------------------------------------------------------ #
    task_cfg = TASK_CONFIGS[args.task_name]
    text_fields = task_cfg["text_fields"]
    if args.num_labels is None:
        args.num_labels = task_cfg["num_labels"]

    logger.info(f"Loading dataset: {args.task_name}")
    ds = load_dataset(*task_cfg["dataset_args"])
    train_data = ds["train"]
    eval_data = ds[task_cfg["eval_split"]]
    if args.task_name == "imdb":
        eval_data = eval_data.filter(lambda x: x["label"] != -1)

    # ------------------------------------------------------------------ #
    # Load victim model
    # ------------------------------------------------------------------ #
    logger.info(f"Loading victim model from {args.model_path}")
    model_name = args.model_name
    model_type = SUPPORTED_MODELS.get(model_name, "bert")

    encoder_path = os.path.join(args.model_path, "best_encoder.pt")
    server_path = os.path.join(args.model_path, "best_classifier.pt")

    if os.path.exists(encoder_path) and os.path.exists(server_path):
        encoder_state = torch.load(encoder_path, map_location=device)
        server_state = torch.load(server_path, map_location=device)
    else:
        logger.warning(
            f"Model checkpoints not found at {args.model_path}. "
            f"Expected best_encoder.pt and best_classifier.pt. "
            f"Using pre-trained weights (no fine-tuning, no DP noise).")
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = args.num_labels
        config.output_attentions = True
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config)
        tmp_enc = ClientEncoder(base, model_type)
        encoder_state = tmp_enc.state_dict()
        tmp_srv = ServerClassifier(config.hidden_size, args.num_labels)
        server_state = tmp_srv.state_dict()
        del base, tmp_enc, tmp_srv

    victim = RapSFLVictim(
        model_name=model_name,
        model_type=model_type,
        encoder_state=encoder_state,
        server_state=server_state,
        args=args,
        device=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------ #
    # Run selected attacks
    # ------------------------------------------------------------------ #
    if args.attacks == "all":
        attack_list = ["emb_inv", "mia", "sip", "namoe", "mi"]
    else:
        attack_list = [a.strip() for a in args.attacks.split(",")]

    all_results = {}

    print("\n" + "=" * 60)
    print("RAP-SFL Privacy Attack Evaluation")
    print(f"  Model: {model_name} | Task: {args.task_name} | ε={args.epsilon}")
    print("=" * 60)

    if "emb_inv" in attack_list:
        r = run_embedding_inversion(
            victim, tokenizer, eval_data, text_fields, device,
            max_samples=args.eval_samples, max_seq_len=args.max_seq_len)
        all_results["embedding_inversion"] = r

    if "mia" in attack_list:
        r = run_mia(
            victim, tokenizer, train_data, eval_data, text_fields, device,
            max_samples=args.mia_samples, max_seq_len=args.max_seq_len)
        all_results["mia"] = r

    if "sip" in attack_list:
        r = run_sip_attack(
            victim, tokenizer, train_data, eval_data, text_fields,
            device, args)
        all_results["sip_inversion"] = r

    if "namoe" in attack_list:
        r = run_namoe_attack(
            victim, tokenizer, train_data, eval_data, text_fields,
            device, args)
        all_results["namoe_inversion"] = r

    if "mi" in attack_list:
        r = run_mutual_information(
            victim, tokenizer, eval_data, text_fields, device,
            max_samples=args.eval_samples,
            max_seq_len=args.max_seq_len,
            mine_epochs=args.mine_epochs,
            mine_lr=args.mine_lr,
            mine_hidden=args.mine_hidden)
        all_results["mutual_information"] = r

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ATTACK RESULTS SUMMARY")
    print("=" * 60)
    for attack_name, metrics in all_results.items():
        print(f"\n  [{attack_name}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
    print("=" * 60)

    out_path = os.path.join(args.model_path, "privacy_attack_results.json")
    try:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {out_path}")
    except Exception:
        logger.warning(f"Could not save results to {out_path}")
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
