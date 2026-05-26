# %%
"""
Log-t distribution parameter prediction model.
Predicts logt_mu and logt_sigma for a fixed degrees-of-freedom ν=3.5.
Architecture: encoder (DeBERTa or similar) + multi-pooling MLP heads.

Log-t distribution: if Y ~ Student-t(ν, μ, σ), then X = exp(Y) follows the Log-t distribution.
"""

import os
# Must be set before all other imports
os.environ['MKL_SERVICE_FORCE_INTEL'] = '7'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import time
import random
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# ==================== Configuration ====================
# TODO: Set to your Log-t distribution data CSV file path
DATA_PATH = "xxx"
TEST_PART = 0.4

# TODO: Set to your pre-trained encoder model path (e.g. DeBERTa)
MODEL_PATH = 'xxx'

# Training hyperparameters
NUM_EPOCHS = 20
ENCODER_TUNING_EPOCHS = 12  # Epochs 1-12: fine-tune encoder; epochs 13-20: freeze encoder
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
MAX_LENGTH = 512

LR_ENCODER_TUNING = 2e-5   # Learning rate while encoder is trainable
LR_FREEZE_ENCODER = 5e-5   # Learning rate after encoder is frozen

HIDDEN_DIM = 256

# GPU configuration
USE_GPU_IDS = [6,7]  # GPU indices to use, e.g. [0,1]
DEVICE = torch.device(f'cuda:{USE_GPU_IDS[0]}' if torch.cuda.is_available() else 'cpu')

# TODO: Set to your desired model checkpoint output directory
BASE_SAVE_DIR = Path("xxx")

# Generate a unique save path to avoid overwriting previous runs
def get_unique_save_dir(base_dir, data_name, model_name):
    """Return a non-conflicting subdirectory under base_dir."""
    dir_name = f"{data_name}_{model_name}"
    save_path = base_dir / dir_name
    if not save_path.exists():
        return save_path
    counter = 1
    while True:
        new_dir_name = f"{dir_name}_{counter}"
        save_path = base_dir / new_dir_name
        if not save_path.exists():
            return save_path
        counter += 1

data_filename = Path(DATA_PATH).stem
model_name = Path(MODEL_PATH).name
SAVE_DIR = get_unique_save_dir(BASE_SAVE_DIR, data_filename, model_name)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Model will be saved to: {SAVE_DIR}")

RANDOM_SEED = random.randint(0, 10000)
print(f"Random seed for this run: {RANDOM_SEED}")
config = {
    'random_seed': RANDOM_SEED,
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'distribution': 'Log-t (fixed ν=3.5)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}
with open(SAVE_DIR / 'training_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# ==================== Dataset ====================
class LogTDataset(Dataset):
    """Dataset for Log-t distribution parameter prediction.

    Log-t distribution: if Y ~ Student-t(ν, μ, σ), then X = exp(Y) follows the Log-t distribution.
    Fixed degrees of freedom: ν=3.5.

    Applies normalization to targets:
      - mu: standardized directly
      - sigma: log1p-transformed, then standardized
    """
    def __init__(self, df, tokenizer, max_length, normalize_stats=None):
        """
        Args:
            df: DataFrame with columns 'prompt', 'logt_mu', 'logt_sigma'
            tokenizer: tokenizer for text encoding
            max_length: maximum sequence length
            normalize_stats: dict with keys 'mu_mean', 'mu_std', 'sigma_log_mean', 'sigma_log_std'.
                             If None, statistics are computed from df (training set).
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if normalize_stats is None:
            # Training set: compute normalization statistics
            sigma_log = np.log1p(df['logt_sigma'].values)  # log(1+x)

            self.mu_mean = df['logt_mu'].mean()
            self.mu_std = df['logt_mu'].std()
            self.sigma_log_mean = sigma_log.mean()
            self.sigma_log_std = sigma_log.std()

            print(f"\nNormalization statistics:")
            print(f"  mu: mean={self.mu_mean:.4f}, std={self.mu_std:.4f}")
            print(f"  sigma (after log1p): mean={self.sigma_log_mean:.4f}, std={self.sigma_log_std:.4f}")
        else:
            # Validation / test set: reuse training-set statistics
            self.mu_mean = normalize_stats['mu_mean']
            self.mu_std = normalize_stats['mu_std']
            self.sigma_log_mean = normalize_stats['sigma_log_mean']
            self.sigma_log_std = normalize_stats['sigma_log_std']

    def get_stats(self):
        """Return normalization statistics for use by validation/test datasets."""
        return {
            'mu_mean': self.mu_mean,
            'mu_std': self.mu_std,
            'sigma_log_mean': self.sigma_log_mean,
            'sigma_log_std': self.sigma_log_std
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
        logt_mu = float(row['logt_mu'])
        logt_sigma = float(row['logt_sigma'])

        # Tokenization
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Target transformation and normalization
        # mu: standardize directly
        mu_normalized = (logt_mu - self.mu_mean) / (self.mu_std + 1e-8)

        # sigma: log1p transform to avoid log(0), then standardize
        sigma_log = np.log1p(logt_sigma)
        sigma_normalized = (sigma_log - self.sigma_log_mean) / (self.sigma_log_std + 1e-8)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'logt_mu': torch.tensor(mu_normalized, dtype=torch.float),
            'logt_sigma': torch.tensor(sigma_normalized, dtype=torch.float),
            # Raw values retained for evaluation de-normalization
            'logt_mu_raw': torch.tensor(logt_mu, dtype=torch.float),
            'logt_sigma_raw': torch.tensor(logt_sigma, dtype=torch.float)
        }


# ==================== Model ====================
class LogTPredictionModel(nn.Module):
    """
    Log-t distribution parameter prediction model.
    Outputs: logt_mu and logt_sigma (fixed ν=3.5).

    Architecture: encoder → [CLS, Mean, Max] multi-pooling → independent
    feature extractors and prediction heads for mu and sigma.
    """
    def __init__(self, model_name_or_path, hidden_dim=256):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)

        self.hidden_size = self.config.hidden_size
        self.model_type = self.config.model_type

        print(f"Encoder type: {self.model_type}")
        print(f"Hidden size: {self.hidden_size}")

        # Multi-pooling concatenation: CLS + Mean + Max
        feature_dim = self.hidden_size * 3

        # Independent feature extractors for mu and sigma
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

        self._init_weights()

        print(f"\nModel architecture:")
        print(f"  Encoder: {self.model_type}")
        print(f"  Target distribution: Log-t (fixed ν=3.5)")
        print(f"  Pooling: [CLS/BOS] + Mean + Max")
        print(f"  mu feature extractor: {feature_dim} -> {hidden_dim} -> {hidden_dim}")
        print(f"  sigma feature extractor: {feature_dim} -> {hidden_dim} -> {hidden_dim}")
        print(f"  mu predictor head: {hidden_dim} -> {hidden_dim//2} -> 1")
        print(f"  sigma predictor head: {hidden_dim} -> {hidden_dim//2} -> 1")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        predictor_params = sum(p.numel() for p in self.mu_feature_extractor.parameters()) + \
                          sum(p.numel() for p in self.sigma_feature_extractor.parameters()) + \
                          sum(p.numel() for p in self.mu_predictor.parameters()) + \
                          sum(p.numel() for p in self.sigma_predictor.parameters())
        print(f"  Prediction head parameters: {predictor_params:,}\n")

    def _init_weights(self):
        for module in [self.mu_feature_extractor, self.sigma_feature_extractor,
                       self.mu_predictor, self.sigma_predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # Multi-pooling
        # 1. CLS/BOS token
        cls_output = hidden_states[:, 0, :]

        # 2. Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_output = sum_embeddings / sum_mask

        # 3. Max pooling
        hidden_states_masked = hidden_states * mask_expanded + (1.0 - mask_expanded) * -1e9
        max_output = torch.max(hidden_states_masked, dim=1)[0]

        combined = torch.cat([cls_output, mean_output, max_output], dim=-1)

        mu_features = self.mu_feature_extractor(combined)
        sigma_features = self.sigma_feature_extractor(combined)

        mu = self.mu_predictor(mu_features).squeeze(-1)
        sigma = self.sigma_predictor(sigma_features).squeeze(-1)

        return mu, sigma

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")


# ==================== Training ====================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()

    # Freeze encoder at the configured epoch boundary
    if epoch == ENCODER_TUNING_EPOCHS:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}: freezing encoder, switching to lr={LR_FREEZE_ENCODER}")
        print(f"{'='*80}")

        encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder
        for param in encoder.parameters():
            param.requires_grad = False

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_FREEZE_ENCODER,
            weight_decay=0.01,
            eps=1e-8
        )

        remaining_steps = (NUM_EPOCHS - epoch) * len(dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=remaining_steps
        )

    total_loss = 0
    mu_predictions, mu_targets = [], []
    sigma_predictions, sigma_targets = [], []

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_mu = batch['logt_mu'].to(device)
        target_sigma = batch['logt_sigma'].to(device)

        optimizer.zero_grad()
        pred_mu, pred_sigma = model(input_ids, attention_mask)

        loss_mu = criterion(pred_mu, target_mu)
        loss_sigma = criterion(pred_sigma, target_sigma)

        # Dynamic sigma loss weight: linearly increases from 3 to 4 over training
        sigma_weight = 3 + (epoch / NUM_EPOCHS) * 1.0
        loss = 1.0 * loss_mu + sigma_weight * loss_sigma

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN/Inf loss at batch {batch_idx}, skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        mu_predictions.extend(pred_mu.detach().cpu().numpy())
        mu_targets.extend(target_mu.cpu().numpy())
        sigma_predictions.extend(pred_sigma.detach().cpu().numpy())
        sigma_targets.extend(target_sigma.cpu().numpy())

        if batch_idx % 50 == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss_mu': f'{loss_mu.item():.4f}',
                'loss_sigma': f'{loss_sigma.item():.4f}',
                'sigma_w': f'{sigma_weight:.2f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

    avg_loss = total_loss / len(dataloader)

    mu_preds = np.array(mu_predictions)
    mu_targs = np.array(mu_targets)
    sigma_preds = np.array(sigma_predictions)
    sigma_targs = np.array(sigma_targets)

    mae_mu = np.mean(np.abs(mu_preds - mu_targs))
    mae_sigma = np.mean(np.abs(sigma_preds - sigma_targs))

    print(f"\n  Train stats:")
    print(f"    mu    - pred: [{mu_preds.min():.3f}, {mu_preds.max():.3f}], "
          f"true: [{mu_targs.min():.3f}, {mu_targs.max():.3f}], MAE: {mae_mu:.4f}")
    print(f"    sigma - pred: [{sigma_preds.min():.3f}, {sigma_preds.max():.3f}], "
          f"true: [{sigma_targs.min():.3f}, {sigma_targs.max():.3f}], MAE: {mae_sigma:.4f}")

    return avg_loss, mae_mu, mae_sigma, optimizer, scheduler


def evaluate(model, dataloader, criterion, device, normalize_stats):
    """
    Evaluate the model, reporting metrics in the original (de-normalized) space.

    Args:
        model: trained model
        dataloader: DataLoader for the evaluation set
        criterion: loss function (applied in normalized space)
        device: compute device
        normalize_stats: normalization statistics dict from LogTDataset.get_stats()
    """
    model.eval()
    total_loss = 0
    mu_predictions, mu_targets = [], []
    sigma_predictions, sigma_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_mu = batch['logt_mu'].to(device)    # normalized
            target_sigma = batch['logt_sigma'].to(device)  # normalized

            pred_mu, pred_sigma = model(input_ids, attention_mask)

            # Loss in normalized space
            loss_mu = criterion(pred_mu, target_mu)
            loss_sigma = criterion(pred_sigma, target_sigma)
            loss = loss_mu + loss_sigma

            total_loss += loss.item()

            # De-normalize: mu
            pred_mu_original = pred_mu.cpu().numpy() * normalize_stats['mu_std'] + normalize_stats['mu_mean']
            target_mu_original = target_mu.cpu().numpy() * normalize_stats['mu_std'] + normalize_stats['mu_mean']

            # De-normalize: sigma (reverse standardization, then reverse log1p via expm1)
            pred_sigma_log = pred_sigma.cpu().numpy() * normalize_stats['sigma_log_std'] + normalize_stats['sigma_log_mean']
            pred_sigma_original = np.expm1(pred_sigma_log)

            target_sigma_log = target_sigma.cpu().numpy() * normalize_stats['sigma_log_std'] + normalize_stats['sigma_log_mean']
            target_sigma_original = np.expm1(target_sigma_log)

            mu_predictions.extend(pred_mu_original)
            mu_targets.extend(target_mu_original)
            sigma_predictions.extend(pred_sigma_original)
            sigma_targets.extend(target_sigma_original)

    avg_loss = total_loss / len(dataloader)

    mu_preds = np.array(mu_predictions)
    mu_targs = np.array(mu_targets)
    sigma_preds = np.array(sigma_predictions)
    sigma_targs = np.array(sigma_targets)

    def compute_metrics(preds, targets, name=''):
        mae = np.mean(np.abs(preds - targets))
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)

        # MAPE: exclude near-zero targets to avoid division instability
        mask = np.abs(targets) > 0.01
        if mask.sum() > 0:
            mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100
        else:
            mape = float('inf')
            print(f"  Warning ({name}): all target values near zero, MAPE undefined.")

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }

    metrics_mu = compute_metrics(mu_preds, mu_targs, 'mu')
    metrics_sigma = compute_metrics(sigma_preds, sigma_targs, 'sigma')

    return {
        'loss': avg_loss,
        'mu': metrics_mu,
        'sigma': metrics_sigma
    }, (mu_preds, sigma_preds), (mu_targs, sigma_targs)


# ==================== Main training loop ====================
def compute_sample_weights(df):
    """
    Compute per-sample weights to up-weight high-mu and high-sigma samples,
    improving tail coverage.
    """
    weights = np.ones(len(df))

    high_mu_mask = df['logt_mu'] > 5.5
    weights[high_mu_mask] = 1.5

    high_sigma_mask = df['logt_sigma'] > 1.0
    weights[high_sigma_mask] = 1.5

    extreme_mask = (df['logt_mu'] > 6.0) | (df['logt_sigma'] > 1.3)
    weights[extreme_mask] = 2.0

    print(f"\nSample weighting:")
    print(f"  Normal samples  (weight=1.0): {np.sum(weights == 1.0)} ({np.sum(weights == 1.0)/len(weights)*100:.1f}%)")
    print(f"  High-value      (weight=1.5): {np.sum(weights == 1.5)} ({np.sum(weights == 1.5)/len(weights)*100:.1f}%)")
    print(f"  Extreme samples (weight=2.0): {np.sum(weights == 2.0)} ({np.sum(weights == 2.0)/len(weights)*100:.1f}%)")

    return weights


if __name__ == "__main__":
    print("="*80)
    print("Log-t Distribution Parameter Prediction Model Training")
    print("Target distribution: Log-t (fixed ν=3.5)")
    print("Predicted parameters: logt_mu, logt_sigma")
    print("="*80)

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset: {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  logt_mu: min={df['logt_mu'].min():.3f}, max={df['logt_mu'].max():.3f}, "
          f"mean={df['logt_mu'].mean():.3f}")
    print(f"  logt_sigma: min={df['logt_sigma'].min():.3f}, max={df['logt_sigma'].max():.3f}, "
          f"mean={df['logt_sigma'].mean():.3f}")

    if 'ks_p_value' in df.columns:
        print(f"  ks_p_value: min={df['ks_p_value'].min():.4f}, max={df['ks_p_value'].max():.4f}, "
              f"mean={df['ks_p_value'].mean():.4f}")
    if 'valid_sample_count' in df.columns:
        print(f"  valid_sample_count: min={df['valid_sample_count'].min()}, max={df['valid_sample_count'].max()}, "
              f"mean={df['valid_sample_count'].mean():.1f}")

    train_df, temp_df = train_test_split(df, test_size=TEST_PART, random_state=RANDOM_SEED, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Save splits for reproducibility
    test_df.to_csv(SAVE_DIR / 'test_set.csv', index=False)
    val_df.to_csv(SAVE_DIR / 'val_set.csv', index=False)
    print(f"Test set saved to: {SAVE_DIR / 'test_set.csv'}")
    print(f"Val set saved to:  {SAVE_DIR / 'val_set.csv'}")

    print(f"\n{'='*80}")
    print(f"Model configuration")
    print(f"{'='*80}")
    print(f"Model path:  {MODEL_PATH}")
    print(f"Device:      {DEVICE}")
    print(f"GPUs:        {USE_GPU_IDS if torch.cuda.device_count() > 1 else 'CPU or single GPU'}")
    print(f"Batch size:  {TRAIN_BATCH_SIZE}")
    print(f"Hidden dim:  {HIDDEN_DIM}")
    print(f"LR:          {LR_ENCODER_TUNING} (encoder tuning) / {LR_FREEZE_ENCODER} (frozen encoder)")
    print(f"{'='*80}\n")

    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LogTPredictionModel(MODEL_PATH, hidden_dim=HIDDEN_DIM)

    if torch.cuda.device_count() > 1:
        print(f"Using {len(USE_GPU_IDS)} GPUs: {USE_GPU_IDS}")
        model = nn.DataParallel(model, device_ids=USE_GPU_IDS)
    model = model.to(DEVICE)

    print("\nCreating datasets...")

    # Training set: compute normalization statistics
    train_dataset = LogTDataset(train_df, tokenizer, MAX_LENGTH, normalize_stats=None)
    normalize_stats = train_dataset.get_stats()

    # Validation and test sets: reuse training statistics
    val_dataset = LogTDataset(val_df, tokenizer, MAX_LENGTH, normalize_stats=normalize_stats)
    test_dataset = LogTDataset(test_df, tokenizer, MAX_LENGTH, normalize_stats=normalize_stats)

    # Save normalization statistics for inference
    with open(SAVE_DIR / 'normalize_stats.json', 'w') as f:
        json.dump(normalize_stats, f, indent=2)
    print(f"Normalization stats saved to: {SAVE_DIR / 'normalize_stats.json'}")

    # Weighted sampler to improve tail coverage
    train_weights = compute_sample_weights(train_df)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                             sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"DataLoader batches: {len(train_loader)} train / {len(val_loader)} val / {len(test_loader)} test\n")

    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LR_ENCODER_TUNING,
        weight_decay=0.01,
        eps=1e-8
    )

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Training configuration:")
    print(f"  Loss:            MSE")
    print(f"  Optimizer:       AdamW (lr={LR_ENCODER_TUNING}, wd=0.01)")
    print(f"  Scheduler:       Linear warmup ({warmup_steps} steps)")
    print(f"  Gradient clip:   max_norm=1.0")
    print(f"  Sample weighting: high mu/sigma samples up-weighted\n")

    print("\n" + "="*80)
    print("Starting training")
    print("="*80 + "\n")

    train_losses, val_losses = [], []
    train_mae_mus, train_mae_sigmas = [], []
    val_mae_mus, val_mae_sigmas = [], []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*80}")

        train_loss, train_mae_mu, train_mae_sigma, optimizer, scheduler = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE, epoch
        )
        train_losses.append(train_loss)
        train_mae_mus.append(train_mae_mu)
        train_mae_sigmas.append(train_mae_sigma)

        val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE, normalize_stats)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        val_mae_mus.append(val_metrics['mu']['mae'])
        val_mae_sigmas.append(val_metrics['sigma']['mae'])

        print(f"\n  Train - Loss: {train_loss:.4f}, MAE_mu: {train_mae_mu:.4f}, MAE_sigma: {train_mae_sigma:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE_mu: {val_metrics['mu']['mae']:.4f}, "
              f"MAE_sigma: {val_metrics['sigma']['mae']:.4f}")
        print(f"          R2_mu: {val_metrics['mu']['r2']:.4f}, R2_sigma: {val_metrics['sigma']['r2']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'distribution': 'Log-t (fixed ν=3.5)'
            }
            torch.save(checkpoint, SAVE_DIR / "best_model.pth")
            print(f"  Best model saved (val_loss: {val_loss:.4f})")

# %%
# Load best model and evaluate on test set
checkpoint = torch.load(SAVE_DIR / "best_model.pth", weights_only = False)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint['model_state_dict'])

test_metrics, (test_mu_preds, test_sigma_preds), (test_mu_targs, test_sigma_targs) = evaluate(
    model, test_loader, criterion, DEVICE, normalize_stats
)

print(f"\nTest set results:")
print(f"  Loss: {test_metrics['loss']:.4f}")
print(f"\n  mu metrics:")
print(f"    MAE:  {test_metrics['mu']['mae']:.4f}")
print(f"    RMSE: {test_metrics['mu']['rmse']:.4f}")
print(f"    MAPE: {test_metrics['mu']['mape']:.2f}%")
print(f"    R2:   {test_metrics['mu']['r2']:.4f}")
print(f"\n  sigma metrics:")
print(f"    MAE:  {test_metrics['sigma']['mae']:.4f}")
print(f"    RMSE: {test_metrics['sigma']['rmse']:.4f}")
print(f"    MAPE: {test_metrics['sigma']['mape']:.2f}%")
print(f"    R2:   {test_metrics['sigma']['r2']:.4f}")

# Save test results
test_results = {
    'test_metrics': test_metrics,
    'best_epoch': checkpoint['epoch'],
    'distribution': 'Log-t (fixed ν=3.5)'
}
with open(SAVE_DIR / 'test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2, default=float)

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_mae_mus': train_mae_mus,
    'train_mae_sigmas': train_mae_sigmas,
    'val_mae_mus': val_mae_mus,
    'val_mae_sigmas': val_mae_sigmas
}


# %%
# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_mae_mus, label='Train MAE')
axes[1].plot(val_mae_mus, label='Val MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('MAE (mu)')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(train_mae_sigmas, label='Train MAE')
axes[2].plot(val_mae_sigmas, label='Val MAE')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MAE')
axes[2].set_title('MAE (sigma)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves.png', dpi=150)
plt.show()
plt.close()

# Plot prediction scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(test_mu_targs, test_mu_preds, alpha=0.5, s=10)
min_val = min(min(test_mu_targs), min(test_mu_preds))
max_val = max(max(test_mu_targs), max(test_mu_preds))
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
axes[0].set_xlabel('True mu')
axes[0].set_ylabel('Predicted mu')
axes[0].set_title(f'Log-t mu (R2={test_metrics["mu"]["r2"]:.4f})')
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(test_sigma_targs, test_sigma_preds, alpha=0.5, s=10)
min_val = min(min(test_sigma_targs), min(test_sigma_preds))
max_val = max(max(test_sigma_targs), max(test_sigma_preds))
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
axes[1].set_xlabel('True sigma')
axes[1].set_ylabel('Predicted sigma')
axes[1].set_title(f'Log-t sigma (R2={test_metrics["sigma"]["r2"]:.4f})')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'prediction_scatter.png', dpi=150)
plt.show()
plt.close()

print(f"\nTraining complete.")
print(f"Output directory: {SAVE_DIR}")
print(f"  - best_model.pth:        best model checkpoint")
print(f"  - normalize_stats.json:  normalization statistics (required for inference)")
print(f"  - training_config.json:  training configuration")
print(f"  - test_results.json:     test set evaluation results")
print(f"  - training_curves.png:   loss and MAE curves")
print(f"  - prediction_scatter.png: predicted vs. true scatter plots")
