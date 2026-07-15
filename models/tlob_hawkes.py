"""
TLOB-Transformer-Hawkes hybrid for options LOB direction prediction.

Architecture:
1. Shared encoder: TLOB dual-attention transformer stack.
2. Direction head: TLOB-style flattening + MLP (uses full seq_len * hidden_dim).
3. Hawkes head: separate branch with additional transformer layers,
   TTE-conditioned additive bias on the log-intensity.

Direction and Hawkes branches share ONLY the encoder, not any post-encoder layers.
"""

import torch
from torch import nn
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


class _TLOBBlock(nn.Module):
    """Single dual-attention transformer block (feature or time axis), from TLOB."""

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
        x, _ = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x


class TLOBHawkes(nn.Module):
    """
    Full model: TLOB encoder + direction head + Hawkes head.

    The two heads share ONLY the encoder. Direction and Hawkes paths diverge
    at the encoder output and don't share any subsequent parameters.
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
        self.hidden_dim = hidden_dim
        self.seq_size = seq_size
        self.num_event_types = num_event_types

        # --- Encoder (shared, exactly matching TLOB's structure) ---
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        # TLOB's interleaved feature-axis and time-axis blocks
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.encoder_layers.append(_TLOBBlock(hidden_dim, num_heads, hidden_dim))
                self.encoder_layers.append(_TLOBBlock(seq_size, num_heads, seq_size))
            else:
                # Last iteration compresses to hidden_dim//4 and seq_size//4
                self.encoder_layers.append(_TLOBBlock(hidden_dim, num_heads, hidden_dim // 4))
                self.encoder_layers.append(_TLOBBlock(seq_size, num_heads, seq_size // 4))

        # --- Direction head (matches vanilla TLOB exactly) ---
        # After encoder: shape (B, seq_size//4, hidden_dim//4). Flatten and MLP-down.
        direction_input_dim = (hidden_dim // 4) * (seq_size // 4)
        self.direction_layers = nn.ModuleList()
        d = direction_input_dim
        while d > 128:
            self.direction_layers.append(nn.Linear(d, d // 4))
            self.direction_layers.append(nn.GELU())
            d = d // 4
        self.direction_layers.append(nn.Linear(d, 3))

        # --- Hawkes head (separate branch, its own transformer layers) ---
        # Reads from encoder pre-final-compression to keep full hidden_dim available.
        # We pool over the sequence dimension with mean pooling here.
        self.hawkes_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                device=DEVICE,
            )
            for _ in range(hawkes_layers)
        ])
        self.intensity_proj = nn.Linear(hidden_dim, num_event_types)

        # TTE-conditioned additive bias
        self.tte_bias_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, num_event_types),
        )

    def forward(self, x, tte):
        """
        Args:
            x: (B, seq_size, num_features)
            tte: (B,) time-to-expiry in days
        Returns:
            log_intensity: (B, num_event_types)
            direction_logits: (B, 3)
        """
        # --- Encoder path ---
        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")
        x = self.emb_layer(x)
        x = x + self.pos_encoder

        # Grab intermediate encoder output for Hawkes head (after N-1 encoder layer pairs)
        # This is the last full-dimensional representation before final compression.
        hawkes_input = None

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = x.permute(0, 2, 1)  # swap dims after each block (matches TLOB)
            # Capture Hawkes input after the second-to-last layer pair
            # (which is the last layer with full hidden_dim x seq_size shape)
            if i == len(self.encoder_layers) - 3:
                hawkes_input = x.clone()  # (B, seq_size, hidden_dim)

        # --- Direction head (matches TLOB exactly) ---
        direction_x = rearrange(x, "b s f -> b (f s) 1").reshape(x.shape[0], -1)
        for layer in self.direction_layers:
            direction_x = layer(direction_x)
        direction_logits = direction_x

        # --- Hawkes head (separate branch from hawkes_input) ---
        # hawkes_input shape: (B, seq_size, hidden_dim)
        h = hawkes_input
        for layer in self.hawkes_transformer:
            h = layer(h)
        # Mean pool over sequence
        h_pooled = h.mean(dim=1)  # (B, hidden_dim)
        base_intensity = self.intensity_proj(h_pooled)

        # TTE-conditioned additive bias
        log_tte = torch.log(tte.unsqueeze(-1) + 1.0)
        tte_features = torch.cat([log_tte, log_tte ** 2], dim=-1)
        tte_bias = self.tte_bias_mlp(tte_features)

        log_intensity = base_intensity + tte_bias

        return log_intensity, direction_logits


def hawkes_nll_loss(log_intensity, next_event_types, next_event_times):
    """
    Simplified Hawkes NLL: constant-intensity approximation over next-event interval.
    """
    B = log_intensity.shape[0]
    log_lambda_k = log_intensity[torch.arange(B), next_event_types]
    total_intensity = torch.exp(log_intensity).sum(dim=-1)
    integral_term = total_intensity * next_event_times
    ll = log_lambda_k - integral_term
    return -ll.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)