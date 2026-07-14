"""
TLOB-Transformer-Hawkes hybrid for options LOB direction prediction.

Architecture:
1. TLOB encoder produces per-timestep hidden states from LOB sequence.
2. Transformer-Hawkes intensity head takes the encoder output and models
   intensity of the next event, conditioned on TTE via additive bias.
3. Direction prediction head reads the joint hidden state.

Joint loss: point process negative log-likelihood + direction cross-entropy.
"""

import torch
from torch import nn
import numpy as np
from einops import rearrange

from models.bin import BiN
from models.mlplob import MLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):
    T = token_sequence_size
    d = token_embedding_dim
    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)
    denominators = torch.pow(n, 2 * torch.arange(0, d // 2) / d)
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)
    return embeddings.to(DEVICE, non_blocking=True)


class TLOBEncoder(nn.Module):
    """
    Reused TLOB encoder core, outputs (B, seq_len, hidden_dim) sequence representation.

    Same as TLOB but strips the final classification head; returns the encoded sequence
    so the Hawkes head can consume it.
    """

    def __init__(
        self,
        num_features: int,
        seq_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        is_sin_emb: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_size = seq_size

        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)

        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        # Interleaved feature-axis and time-axis transformer layers (TLOB dual attention)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_TransformerBlock(hidden_dim, num_heads, hidden_dim))
            self.layers.append(_TransformerBlock(seq_size, num_heads, seq_size))

    def forward(self, x):
        # x: (B, seq_len, num_features)
        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")
        x = self.emb_layer(x)
        x = x + self.pos_encoder

        # Interleaved feature-attention, time-attention
        # After feature attention: (B, seq_len, hidden_dim)
        # After time attention:    (B, hidden_dim, seq_len) which we transpose back
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            x = x.permute(0, 2, 1)  # swap the last two dims each round, matching TLOB
        # After all layers, x is (B, seq_len, hidden_dim) if num_layers is even
        # (which it is: 2 layers per iteration, always even count)
        return x


class _TransformerBlock(nn.Module):
    """Single dual-attention transformer block (feature or time axis)."""

    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.attention = nn.MultiheadAttention(
            hidden_dim * num_heads, num_heads, batch_first=True, device=DEVICE
        )
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        res = x
        q, k, v = self.q(x), self.k(x), self.v(x)
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, att


class TransformerHawkesHead(nn.Module):
    """
    Transformer Hawkes intensity head with TTE-conditioned additive bias.

    Given encoded LOB sequence hidden states, produces:
    - Log-intensity for K event types at query time (used in NLL loss)
    - Baseline direction logits for direction prediction

    TTE conditioning is a learned function TTE -> bias vector, added to the intensity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_event_types: int = 3,
        num_layers: int = 2,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_event_types = num_event_types
        self.hidden_dim = hidden_dim

        # Transformer layers on top of TLOB output. Each takes the sequence and refines it.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                device=DEVICE,
            )
            for _ in range(num_layers)
        ])

        # Intensity head: from pooled hidden state -> log-intensity per event type
        self.intensity_proj = nn.Linear(hidden_dim, num_event_types)

        # TTE-conditioned additive bias: TTE (scalar) -> bias vector for intensity
        # Small MLP: [tte, tte^2] -> num_event_types bias values
        self.tte_bias_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, num_event_types),
        )

        # Direction prediction head from pooled hidden state (down/stable/up)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, encoder_output, tte):
        """
        Args:
            encoder_output: (B, seq_len, hidden_dim) from TLOBEncoder
            tte: (B,) time-to-expiry in days (raw, unnormalized)
        Returns:
            log_intensity: (B, num_event_types) log-intensity for next-event prediction
            direction_logits: (B, 3) direction prediction logits
        """
        # Refine sequence with additional transformer layers
        x = encoder_output
        for layer in self.transformer_layers:
            x = layer(x)

        # Pool to single vector per batch item (use last-token, matching Transformer Hawkes convention)
        pooled = x[:, -1, :]  # (B, hidden_dim)

        # Intensity from pooled hidden state
        base_intensity = self.intensity_proj(pooled)  # (B, num_event_types)

        # TTE conditioning: additive bias derived from TTE
        # Use log(TTE + 1) and (log(TTE + 1))^2 as features for the bias network
        log_tte = torch.log(tte.unsqueeze(-1) + 1.0)  # (B, 1)
        tte_features = torch.cat([log_tte, log_tte ** 2], dim=-1)  # (B, 2)
        tte_bias = self.tte_bias_mlp(tte_features)  # (B, num_event_types)

        log_intensity = base_intensity + tte_bias

        # Direction prediction from same pooled hidden state
        direction_logits = self.direction_head(pooled)

        return log_intensity, direction_logits


class TLOBHawkes(nn.Module):
    """
    Full model: TLOB encoder + Transformer Hawkes head with TTE conditioning.
    """

    def __init__(
        self,
        num_features: int,
        seq_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_event_types: int = 3,
        hawkes_layers: int = 2,
        is_sin_emb: bool = True,
    ):
        super().__init__()
        self.encoder = TLOBEncoder(
            num_features=num_features,
            seq_size=seq_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            is_sin_emb=is_sin_emb,
        )
        self.hawkes_head = TransformerHawkesHead(
            hidden_dim=hidden_dim,
            num_event_types=num_event_types,
            num_layers=hawkes_layers,
            num_heads=num_heads,
        )

    def forward(self, x, tte):
        """
        Args:
            x: (B, seq_len, num_features) LOB sequence
            tte: (B,) time-to-expiry in days
        Returns:
            log_intensity: (B, num_event_types)
            direction_logits: (B, 3)
        """
        h = self.encoder(x)
        return self.hawkes_head(h, tte)


def hawkes_nll_loss(log_intensity, next_event_types, next_event_times):
    """
    Simplified Hawkes-style negative log-likelihood loss for the next event.

    Given predicted log-intensity per event type and the actual next event type + time gap,
    compute NLL under a marked point process approximation.

    Args:
        log_intensity: (B, K) predicted log-intensity per event type
        next_event_types: (B,) actual next event type (0..K-1)
        next_event_times: (B,) time gap to next event (used for expected intensity integral)

    Returns:
        NLL scalar
    """
    # log-likelihood of observed event type k at time delta:
    # ll = log_intensity[k] - integral_0^delta lambda(t) dt
    # Approximation: assume constant intensity over the interval -> integral ~ sum(exp(log_intensity)) * delta

    B = log_intensity.shape[0]
    log_lambda_k = log_intensity[torch.arange(B), next_event_types]  # (B,)
    total_intensity = torch.exp(log_intensity).sum(dim=-1)  # (B,)
    integral_term = total_intensity * next_event_times  # (B,)

    ll = log_lambda_k - integral_term
    return -ll.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)