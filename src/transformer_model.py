"""
Transformer Possession Value Model — The Architect Framework

Processes possession chains as sequences of actions, predicts terminal xG,
and extracts attention-based credit assignment to quantify each action's
contribution to the final outcome.

Architecture:
  - Action embedding: type (one-hot) + spatial (x,y coords) + temporal features
  - Positional encoding (learned)
  - 4-layer Transformer encoder, 4 heads, dim=64
  - Regression head: mean-pool → linear → predicted xG
  - Attention extraction for credit assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import time
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# Action type vocabulary
ACTION_TYPES = [
    'Pass', 'Carry', 'Shot', 'Dribble', 'Ball Receipt*',
    'Ball Recovery', 'Clearance', 'Miscontrol', 'Dispossessed',
    'Interception', 'Foul Won', 'Goal Keeper', 'Other'
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
N_ACTION_TYPES = len(ACTION_TYPES)

MAX_SEQ_LEN = 30


class PossessionDataset(Dataset):
    """Dataset of possession chains for the Transformer."""

    def __init__(self, chains_df, max_len=MAX_SEQ_LEN):
        self.max_len = max_len
        self.sequences = []
        self.targets = []
        self.chain_meta = []

        for _, chain in chains_df.iterrows():
            seq = self._build_sequence(chain)
            if seq is None:
                continue
            self.sequences.append(seq)
            self.targets.append(chain['terminal_xg'])
            self.chain_meta.append({
                'match_id': chain['match_id'],
                'team': chain['team'],
                'players': chain['players'],
                'action_types': chain['action_types'],
                'n_actions': chain['n_actions'],
                'ended_in_shot': chain['ended_in_shot'],
            })

    def _build_sequence(self, chain):
        """Convert a chain into a feature tensor."""
        action_types = chain['action_types']
        locations = chain['locations']
        durations = chain['durations']
        n = min(len(action_types), self.max_len)

        # Take last max_len actions (most relevant to outcome)
        if len(action_types) > self.max_len:
            offset = len(action_types) - self.max_len
            action_types = action_types[offset:]
            locations = locations[offset:]
            durations = durations[offset:]

        features = []
        prev_x, prev_y = 60, 40  # center of pitch as default

        for i in range(n):
            # Action type one-hot
            atype = action_types[i]
            aidx = ACTION_TO_IDX.get(atype, ACTION_TO_IDX['Other'])
            type_onehot = [0.0] * N_ACTION_TYPES
            type_onehot[aidx] = 1.0

            # Spatial features
            loc = locations[i]
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                x, y = float(loc[0]) / 120.0, float(loc[1]) / 80.0  # normalize
                if np.isnan(x) or np.isnan(y):
                    x, y = prev_x / 120.0, prev_y / 80.0
                else:
                    prev_x, prev_y = float(loc[0]), float(loc[1])
            else:
                x, y = prev_x / 120.0, prev_y / 80.0

            # Temporal: duration of action
            dur = durations[i] if i < len(durations) and durations[i] is not None else 0
            dur = float(dur) if not (isinstance(dur, float) and np.isnan(dur)) else 0
            dur = min(dur / 10.0, 1.0)  # normalize, cap at 10s

            # Distance to goal (normalized)
            dist_to_goal = 1.0 - x  # 0 = at goal, 1 = far from goal

            # Feature vector: type_onehot (13) + x + y + dur + dist_to_goal = 17
            feat = type_onehot + [x, y, dur, dist_to_goal]
            features.append(feat)

        if len(features) == 0:
            return None

        # Pad to max_len
        feat_dim = len(features[0])
        while len(features) < self.max_len:
            features.append([0.0] * feat_dim)

        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        # Attention mask: 1 for real tokens, 0 for padding
        mask = (seq.sum(dim=-1) != 0).float()
        return seq, mask, target


class PossessionTransformer(nn.Module):
    """Transformer model for possession chain value prediction."""

    def __init__(self, input_dim=17, d_model=64, nhead=4, num_layers=4,
                 d_ff=128, max_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned positional encoding
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # For attention extraction
        self.attn_weights = None

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) — 1 for real, 0 for padding
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        h = self.input_proj(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.pos_embed(positions)

        # Create key_padding_mask (True = ignore)
        key_padding_mask = (mask == 0) if mask is not None else None

        # Transformer encoder
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Mean pool over non-padded positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq, 1)
            h_masked = h * mask_expanded
            h_pooled = h_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h_pooled = h.mean(dim=1)

        # Predict xG
        output = self.output_head(h_pooled).squeeze(-1)
        return output

    def extract_attention(self, x, mask=None):
        """Extract attention weights from the last layer."""
        batch_size, seq_len, _ = x.shape
        h = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.pos_embed(positions)

        key_padding_mask = (mask == 0) if mask is not None else None

        # Run through all layers except last, then manually run last layer
        for layer in self.transformer.layers[:-1]:
            h = layer(h, src_key_padding_mask=key_padding_mask)

        # Last layer — extract attention
        last_layer = self.transformer.layers[-1]

        # Self-attention sublayer
        h_norm = last_layer.norm1(h)
        attn_out, attn_weights = last_layer.self_attn(
            h_norm, h_norm, h_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True
        )
        # attn_weights: (batch, seq, seq)

        return attn_weights


def train_model(chains_df, epochs=80, lr=1e-3, batch_size=64, val_split=0.2):
    """Train the Transformer possession value model."""
    print("="*70)
    print("TRAINING TRANSFORMER POSSESSION VALUE MODEL")
    print("="*70)

    # Create dataset
    dataset = PossessionDataset(chains_df)
    print(f"  Dataset: {len(dataset)} chains")

    # Train/val split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model
    model = PossessionTransformer(input_dim=17, d_model=64, nhead=4,
                                   num_layers=4, d_ff=128, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        n_batches = 0
        for seq, mask, target in train_loader:
            optimizer.zero_grad()
            pred = model(seq, mask)
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for seq, mask, target in val_loader:
                pred = model(seq, mask)
                loss = F.mse_loss(pred, target)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss = val_loss / n_val_batches
        val_losses.append(val_loss)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "possession_transformer.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:>3}: train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  "
                  f"{'*best*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start
    print(f"\n  Training complete in {elapsed:.0f}s")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Baseline (predict mean): {np.var(chains_df['terminal_xg']):.6f}")

    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / "possession_transformer.pt", weights_only=True))
    return model, dataset


def extract_action_values(model, dataset, chains_df):
    """
    For each chain, extract attention-based action values.
    Value(action_i) = attention_weight(i) * terminal_xG
    """
    print("\nExtracting attention-based action values...")
    model.eval()

    all_values = []
    loader = DataLoader(dataset, batch_size=32)

    chain_idx = 0
    for seq, mask, target in loader:
        with torch.no_grad():
            attn = model.extract_attention(seq, mask)  # (batch, seq, seq)

        batch_size = seq.shape[0]
        for i in range(batch_size):
            if chain_idx >= len(dataset.chain_meta):
                break

            meta = dataset.chain_meta[chain_idx]
            xg = dataset.targets[chain_idx]
            n_real = int(mask[i].sum().item())

            # Average attention received by each position (column-wise mean)
            attn_i = attn[i, :n_real, :n_real]  # real tokens only
            action_importance = attn_i.mean(dim=0).numpy()  # how much each action is attended to

            # Normalize to sum to 1
            total = action_importance.sum()
            if total > 0:
                action_importance = action_importance / total

            # Assign values
            players = meta['players']
            action_types = meta['action_types']
            n_orig = meta['n_actions']

            # If chain was truncated, offset
            offset = max(0, n_orig - MAX_SEQ_LEN)

            for j in range(n_real):
                orig_idx = offset + j
                if orig_idx >= len(players):
                    break

                pos_from_end = n_orig - 1 - orig_idx

                all_values.append({
                    'match_id': meta['match_id'],
                    'team': meta['team'],
                    'player': players[orig_idx],
                    'action_type': action_types[orig_idx] if orig_idx < len(action_types) else 'Unknown',
                    'action_index': orig_idx,
                    'position_from_end': pos_from_end,
                    'attention_weight': float(action_importance[j]),
                    'action_value': float(action_importance[j] * xg),
                    'terminal_xg': xg,
                    'chain_length': n_orig,
                    'ended_in_shot': meta['ended_in_shot'],
                })

            chain_idx += 1

    values_df = pd.DataFrame(all_values)
    values_df.to_parquet(PROCESSED_DIR / "action_values.parquet", index=False)
    print(f"  Saved {len(values_df)} action values")
    return values_df


def compare_credit_assignment(values_df):
    """Compare attention-based credit vs naive baselines."""
    print(f"\n{'='*70}")
    print("CREDIT ASSIGNMENT COMPARISON")
    print(f"{'='*70}")

    shot_values = values_df[values_df['ended_in_shot']].copy()

    # Group by chain and compute credit fractions
    chain_groups = shot_values.groupby(['match_id', 'team', 'terminal_xg'])

    # For each chain, compute what fraction of credit goes to pre-assist zone (pos 2-5)
    methods = {'attention': [], 'equal': [], 'linear_decay': [], 'exp_decay': []}

    for (mid, team, xg), group in chain_groups:
        n = len(group)
        if n < 3 or xg == 0:
            continue

        # Attention-based
        attn_credits = group['attention_weight'].values
        pre_assist_mask = (group['position_from_end'] >= 2) & (group['position_from_end'] <= 5)
        methods['attention'].append(attn_credits[pre_assist_mask].sum())

        # Equal weighting
        equal_credits = np.ones(n) / n
        methods['equal'].append(equal_credits[pre_assist_mask.values].sum())

        # Linear decay (closer to end = more credit)
        linear = np.arange(1, n + 1, dtype=float)
        linear = linear / linear.sum()
        methods['linear_decay'].append(linear[pre_assist_mask.values].sum())

        # Exponential decay (gamma=0.95)
        gamma = 0.95
        exp = np.array([gamma ** (n - 1 - i) for i in range(n)])
        exp = exp / exp.sum()
        methods['exp_decay'].append(exp[pre_assist_mask.values].sum())

    print("\n  % of chain credit assigned to pre-assist zone (positions 2-5 from shot):")
    for method, fracs in methods.items():
        print(f"    {method:<20} mean: {np.mean(fracs)*100:.1f}%  "
              f"median: {np.median(fracs)*100:.1f}%")

    # Per-player PACV under attention vs equal
    print(f"\n  Player PACV comparison (attention vs equal weighting):")
    for name in ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']:
        player = shot_values[shot_values['player'] == name]
        pre_assist = player[(player['position_from_end'] >= 2) & (player['position_from_end'] <= 5)]

        attn_pacv = pre_assist['action_value'].sum()
        equal_pacv = (pre_assist['terminal_xg'] / pre_assist['chain_length']).sum()

        print(f"    {name:<25} Attention PACV: {attn_pacv:.4f}  "
              f"Equal PACV: {equal_pacv:.4f}  "
              f"Ratio: {attn_pacv/equal_pacv:.2f}x" if equal_pacv > 0 else "")

    # Where does attention concentrate?
    print(f"\n  Mean attention weight by position from shot:")
    pos_attn = shot_values.groupby('position_from_end')['attention_weight'].mean()
    for pos in range(min(10, len(pos_attn))):
        if pos in pos_attn.index:
            label = {0:'shot', 1:'assist', 2:'pre-asst 2', 3:'pre-asst 3',
                     4:'pre-asst 4', 5:'pre-asst 5'}.get(pos, f'pos {pos}')
            bar = '█' * int(pos_attn[pos] * 200)
            print(f"    {label:<12} {pos_attn[pos]:.4f} {bar}")


if __name__ == "__main__":
    chains = pd.read_parquet(PROCESSED_DIR / "possession_chains.parquet")
    MODEL_DIR.mkdir(exist_ok=True)

    model, dataset = train_model(chains, epochs=80, lr=1e-3, batch_size=64)
    values_df = extract_action_values(model, dataset, chains)
    compare_credit_assignment(values_df)
