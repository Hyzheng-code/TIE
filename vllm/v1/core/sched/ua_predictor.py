# SPDX-License-Identifier: Apache-2.0
# Uncertainty-Aware (UA) output length predictor.
# Supports log-t (fixed ν), log-t (dynamic ν), and log-normal distributions,
# selectable via MODEL_TYPE.

import os
import re
import json
import math
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from transformers import AutoConfig, AutoModel, AutoTokenizer

from vllm.v1.core.sched.ua_score_calculator import Calu_lognorm_score, Calu_logt_score

# Model type: "logt" | "logt_dynamic" | "lognorm"
MODEL_TYPE = "logt"

MODEL_CONFIGS = {
    "logt": {
        "model_path": "xxx",
        "description": "Log-t distribution (fixed ν=3.5)"
    },
}

if MODEL_TYPE not in MODEL_CONFIGS:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'lognorm', 'logt', or 'logt_dynamic'.")

CURRENT_CONFIG = MODEL_CONFIGS[MODEL_TYPE]
MODEL_PATH = CURRENT_CONFIG["model_path"]

_MODEL = None
_TOKENIZER = None
_NORMALIZE_STATS = None
_DEVICE = None

MAX_LENGTH = 2048  # must match the value used during training

# Max running batch size B, corresponding to --max-num-seqs.
# Used in the adaptive β formula (paper Eq. 12).
# Injected by the launch script via the UA_GPU_BATCH_SIZE environment variable.
GPU_BATCH_SIZE = int(os.environ.get('UA_GPU_BATCH_SIZE', '32'))


def _compute_score(mu: float, sigma: float, waiting_count: int, nu: float = None) -> float:
    """Compute the TIE scheduling score: E[X] + β·CVaR_α[X].

    β is adapted to system pressure (paper Eq. 12):
        β = min(0.5, max(0.1, 0.1 · Lq / B))
    where Lq = waiting_count and B = GPU_BATCH_SIZE.
    """
    beta = min(0.5, max(0.1, 0.1 * waiting_count / GPU_BATCH_SIZE))

    if MODEL_TYPE == "logt":
        score = Calu_logt_score.mean_add_n_cvar(mu, sigma, a=90, n=beta)
    elif MODEL_TYPE == "logt_dynamic":
        raise NotImplementedError("MODEL_TYPE='logt_dynamic' is not supported.")
    elif MODEL_TYPE == "lognorm":
        score = Calu_lognorm_score.mean_add_n_cvar(mu, sigma, a=90, n=beta)
    return score


def denormalize_predictions(
    mu_normalized: float,
    sigma_normalized: float,
    stats: dict,
    nu_normalized: float = None
) -> Tuple[float, ...]:
    """Reverse the z-score normalization and log1p transform applied during training."""
    mu = mu_normalized * stats['mu_std'] + stats['mu_mean']

    sigma_log = sigma_normalized * stats['sigma_log_std'] + stats['sigma_log_mean']
    sigma = np.expm1(sigma_log)  # inverse of log1p
    sigma = max(sigma, 1e-6)

    if MODEL_TYPE == "logt_dynamic":
        if nu_normalized is None:
            raise ValueError("logt_dynamic mode requires nu_normalized.")
        nu_log = nu_normalized * stats['nu_log_std'] + stats['nu_log_mean']
        nu = np.expm1(nu_log)
        nu = np.clip(nu, 0.1, 100)
        return mu, sigma, nu
    else:
        return mu, sigma


def extract_original_prompt(full_prompt: str) -> str:
    """Extract the user prompt from a formatted chat string."""
    patterns = [
        r'User:\s*(.+?)(?:\n\nAssistant:|\nAssistant:|Assistant:)',
        r'User:\s*(.+?)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, full_prompt, re.DOTALL)
        if match:
            result = match.group(1).strip()
            if result:
                return result

    if 'User: ' in full_prompt:
        after_user = full_prompt.split('User: ', 1)[1]
        for sep in ['\n\nAssistant:', '\nAssistant:', 'Assistant:']:
            if sep in after_user:
                after_user = after_user.split(sep)[0]
                break
        return after_user.strip()

    return full_prompt.strip()


class BasePredictionModel(nn.Module):
    """DeBERTa encoder with multi-pooling feature extractor and dual MLP heads
    for predicting log-t distribution parameters μ and σ (paper §3.2)."""

    def __init__(self, model_name_or_path, hidden_dim=256):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)

        self.hidden_size = self.config.hidden_size
        self.model_type = self.config.model_type

        # Multi-pooling: CLS + mean-pooled + max-pooled representations
        feature_dim = self.hidden_size * 3

        # Independent feature extractor for μ
        self.mu_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Independent feature extractor for σ
        self.sigma_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Prediction heads
        self.mu_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.sigma_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # CLS token
        cls_output = hidden_states[:, 0, :]

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_output = sum_embeddings / sum_mask

        # Max pooling
        hidden_states_masked = hidden_states * mask_expanded + (1.0 - mask_expanded) * -1e9
        max_output = torch.max(hidden_states_masked, dim=1)[0]

        combined = torch.cat([cls_output, mean_output, max_output], dim=-1)

        mu_features = self.mu_feature_extractor(combined)
        sigma_features = self.sigma_feature_extractor(combined)

        mu = self.mu_predictor(mu_features).squeeze(-1)
        sigma = self.sigma_predictor(sigma_features).squeeze(-1)

        return mu, sigma


class LogNormPredictionModel(BasePredictionModel):
    def __init__(self, model_name_or_path, hidden_dim=256):
        super().__init__(model_name_or_path, hidden_dim)
        self.distribution_type = "lognorm"


class LogTPredictionModel(BasePredictionModel):
    def __init__(self, model_name_or_path, hidden_dim=256):
        super().__init__(model_name_or_path, hidden_dim)
        self.distribution_type = "logt"


class LogTDynamicPredictionModel(nn.Module):
    """Log-t model with a learned degrees-of-freedom parameter ν (ablation variant)."""

    def __init__(self, model_name_or_path, hidden_dim=384):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)

        self.hidden_size = self.config.hidden_size
        self.model_type = self.config.model_type
        self.distribution_type = "logt_dynamic"

        feature_dim = self.hidden_size * 3
        dropout_rate = 0.15

        self.mu_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.sigma_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.nu_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.mu_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.sigma_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.nu_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.mu_feature_extractor, self.sigma_feature_extractor,
                       self.nu_feature_extractor, self.mu_predictor,
                       self.sigma_predictor, self.nu_predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.2)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        cls_output = hidden_states[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_output = sum_embeddings / sum_mask

        hidden_states_masked = hidden_states * mask_expanded + (1.0 - mask_expanded) * -1e9
        max_output = torch.max(hidden_states_masked, dim=1)[0]

        combined = torch.cat([cls_output, mean_output, max_output], dim=-1)

        mu_features = self.mu_feature_extractor(combined)
        sigma_features = self.sigma_feature_extractor(combined)
        nu_features = self.nu_feature_extractor(combined)

        mu = self.mu_predictor(mu_features).squeeze(-1)
        sigma = self.sigma_predictor(sigma_features).squeeze(-1)
        nu = self.nu_predictor(nu_features).squeeze(-1)

        mu = torch.clamp(mu, -15, 15)
        sigma = torch.clamp(sigma, -15, 15)
        nu = torch.clamp(nu, -15, 15)

        return mu, sigma, nu


def load_model(
    model_path: str = MODEL_PATH,
    device_id: int = 0
) -> None:
    """Load the predictor model and tokenizer into global state.

    Paths to configure (see README):
      - model_path: directory containing best_model.pth and normalize_stats.json
      - encoder_path: pretrained DeBERTa encoder directory
    """
    global _MODEL, _TOKENIZER, _NORMALIZE_STATS, _DEVICE

    if _MODEL is not None:
        return

    print(f"[UA] Loading model from: {model_path} (type={MODEL_TYPE.upper()})")

    if device_id >= 0 and torch.cuda.is_available():
        _DEVICE = torch.device(f"cuda:{device_id}")
    else:
        _DEVICE = torch.device("cpu")

    model_file = Path(model_path) / "best_model.pth"
    stats_file = Path(model_path) / "normalize_stats.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not stats_file.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {stats_file}")

    with open(stats_file, 'r') as f:
        _NORMALIZE_STATS = json.load(f)

    checkpoint = torch.load(model_file, map_location=_DEVICE, weights_only=False)

    # Set encoder_path to the pretrained DeBERTa directory (see README)
    encoder_path = "xxx"

    if MODEL_TYPE == "logt":
        _MODEL = LogTPredictionModel(model_name_or_path=encoder_path, hidden_dim=256)
    elif MODEL_TYPE == "logt_dynamic":
        _MODEL = LogTDynamicPredictionModel(model_name_or_path=encoder_path, hidden_dim=384)
    else:
        _MODEL = LogNormPredictionModel(model_name_or_path=encoder_path, hidden_dim=256)

    _MODEL.load_state_dict(checkpoint['model_state_dict'])
    _MODEL.to(_DEVICE)
    _MODEL.eval()

    print(f"[UA] Model loaded ({MODEL_TYPE}, {sum(p.numel() for p in _MODEL.parameters()):,} params).")

    _TOKENIZER = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)


def predict_length_from_token_ids(
    token_ids: list[int],
    tokenizer,
    waiting_count: int = 0
) -> int:
    """Predict the TIE score for a single request given its prompt token IDs."""
    global _MODEL, _TOKENIZER, _NORMALIZE_STATS, _DEVICE

    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    try:
        original_prompt_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt_text = extract_original_prompt(original_prompt_text)
        prompt_text = prompt_text.removeprefix("user\n\n")
        prompt_text = prompt_text.removesuffix("assistant")

        encoding = _TOKENIZER(
            prompt_text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(_DEVICE)
        attention_mask = encoding['attention_mask'].to(_DEVICE)

        with torch.no_grad():
            outputs = _MODEL(input_ids, attention_mask)

            if MODEL_TYPE == "logt_dynamic":
                mu_normalized, sigma_normalized, nu_normalized = outputs
                nu_normalized = nu_normalized.item()
            else:
                mu_normalized, sigma_normalized = outputs
                nu_normalized = None

        mu_normalized = mu_normalized.item()
        sigma_normalized = sigma_normalized.item()

        denorm_result = denormalize_predictions(
            mu_normalized, sigma_normalized, _NORMALIZE_STATS,
            nu_normalized=nu_normalized
        )

        if MODEL_TYPE == "logt_dynamic":
            mu, sigma, nu = denorm_result
        else:
            mu, sigma = denorm_result
            nu = None

        expected_length = _compute_score(mu, sigma, waiting_count, nu=nu)
        return max(1, int(round(expected_length)))

    except Exception as e:
        print(f"[UA] Prediction failed: {e}")
        return 100


def predict_length_from_token_ids_batch(
    token_ids_list: list[list[int]],
    tokenizer,
    waiting_count: int = 0
) -> list[int]:
    """Predict TIE scores for a batch of requests."""
    global _MODEL, _TOKENIZER, _NORMALIZE_STATS, _DEVICE

    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    batch_size = len(token_ids_list)
    if batch_size == 0:
        return []

    try:
        original_prompt_texts = tokenizer.batch_decode(token_ids_list, skip_special_tokens=True)

        prompt_texts = []
        for text in original_prompt_texts:
            extracted = extract_original_prompt(text)
            extracted = extracted.removeprefix("user\n\n")
            extracted = extracted.removesuffix("assistant")
            prompt_texts.append(extracted)

        encoding = _TOKENIZER(
            prompt_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(_DEVICE)
        attention_mask = encoding['attention_mask'].to(_DEVICE)

        with torch.no_grad():
            outputs = _MODEL(input_ids, attention_mask)

            if MODEL_TYPE == "logt_dynamic":
                mu_normalized, sigma_normalized, nu_normalized = outputs
            else:
                mu_normalized, sigma_normalized = outputs
                nu_normalized = None

        results = []
        for i in range(batch_size):
            mu_n = mu_normalized[i].item()
            sigma_n = sigma_normalized[i].item()

            if MODEL_TYPE == "logt_dynamic":
                nu_n = nu_normalized[i].item()
                denorm_result = denormalize_predictions(mu_n, sigma_n, _NORMALIZE_STATS, nu_normalized=nu_n)
                mu, sigma, nu = denorm_result
            else:
                denorm_result = denormalize_predictions(mu_n, sigma_n, _NORMALIZE_STATS)
                mu, sigma = denorm_result
                nu = None

            expected_length = _compute_score(mu, sigma, waiting_count, nu=nu)
            results.append(max(1, int(round(expected_length))))

        return results

    except Exception as e:
        print(f"[UA] Batch prediction failed: {e}")
        return [100] * batch_size


def predict_length_from_prompt_text(prompt_text: str) -> Tuple[float, ...]:
    """Predict distribution parameters and TIE score from raw prompt text."""
    global _MODEL, _TOKENIZER, _NORMALIZE_STATS, _DEVICE

    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    encoding = _TOKENIZER(
        prompt_text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(_DEVICE)
    attention_mask = encoding['attention_mask'].to(_DEVICE)

    with torch.no_grad():
        outputs = _MODEL(input_ids, attention_mask)

        if MODEL_TYPE == "logt_dynamic":
            mu_normalized, sigma_normalized, nu_normalized = outputs
            mu_normalized = mu_normalized.item()
            sigma_normalized = sigma_normalized.item()
            nu_normalized = nu_normalized.item()
        else:
            mu_normalized, sigma_normalized = outputs
            mu_normalized = mu_normalized.item()
            sigma_normalized = sigma_normalized.item()
            nu_normalized = None

    denorm_result = denormalize_predictions(
        mu_normalized, sigma_normalized, _NORMALIZE_STATS,
        nu_normalized=nu_normalized
    )

    if MODEL_TYPE == "logt_dynamic":
        mu, sigma, nu = denorm_result
        expected_length = _compute_score(mu, sigma, 0, nu=nu)
        return mu, sigma, nu, int(round(expected_length))
    else:
        mu, sigma = denorm_result
        expected_length = _compute_score(mu, sigma, 0)
        return mu, sigma, int(round(expected_length))
