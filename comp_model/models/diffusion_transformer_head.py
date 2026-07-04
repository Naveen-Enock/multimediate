"""Conditional diffusion transformer head.

The head denoises the engagement chunk directly as a short token sequence and cross-attends to the
full trunk context window, keeping per-frame temporal structure available throughout denoising.

  * ``y_t`` — noisy engagement chunk ``(B, K, 1)``. The chunk spans the K-frame horizon; each frame
    is one denoising token.
  * ``timesteps`` — diffusion step ``(B,)``; embedded (sinusoidal -> MLP) to ``d_model`` and used
    as the DiT-style AdaLN conditioning signal for every sub-layer.
  * ``X_cond`` — dyadic context ``(B, W, 2d)``: reactive and anticipatory streams concatenated
    along the feature axis so the time dimension stays at W. A learned ``Linear(2d, d)``
    projection (``ctx_proj``) maps X_cond to ``d`` before the cross-attention K/V. The K engagement
    tokens act as queries and attend over all W context frames as keys/values.

Each of the (default 4) transformer blocks runs: AdaLN self-attention over the K engagement tokens,
AdaLN cross-attention (Q = engagement tokens, K/V = ``X_cond``), then an AdaLN FFN. All three
sub-layers are AdaLN-Zero gated (gate zero-init) so the head starts as identity.

The denoiser uses the velocity (v) parameterization ``v = √ᾱ_t·ε − √(1−ᾱ_t)·y₀``.

Training (`noise_pred`) samples ``t`` + noise, forms ``y_t``, and returns the predicted velocity
``v_pred``, target ``v``, ``y_t``, and ``ᾱ_t`` for ``DiffusionHybridCCCLoss``. Inference (`sample`)
maps the predicted velocity back to ``(ŷ₀, ε̂)`` and runs deterministic DDIM over ``sample_steps``,
clamping to the engagement range [0, 1].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.normalize import load_label_stats, load_label_stats_per_dataset


class SinusoidalPosEmb(nn.Module):
    """Standard transformer sinusoidal embedding of a scalar (here: the diffusion timestep)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)           # (B, dim)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DiT AdaLN modulation: ``(1 + scale) ⊙ x + shift`` with per-sample (B, d) params."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTCrossBlock(nn.Module):
    """One conditioned denoiser block: self-attn over chunk tokens + cross-attn to ``X_cond`` + FFN.

    All three sub-layers are affine-free LayerNorm modulated by the timestep embedding (shift/scale)
    and gated (AdaLN-Zero, gate zero-init -> identity at start). The 9 modulation params per block
    come from a single ``SiLU -> Linear`` projection of the conditioning vector.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        # cond -> (shift, scale, gate) x (self-attn, cross-attn, ffn) = 9 * d_model.
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, ctx: torch.Tensor,
                ctx_key_pad: torch.Tensor | None) -> torch.Tensor:
        """``x: (B, K, d)`` queries; ``cond: (B, d)``; ``ctx: (B, L, d)`` K/V;
        ``ctx_key_pad: (B, L)`` True = ignore."""
        (sa_s, sa_c, sa_g, ca_s, ca_c, ca_g,
         ff_s, ff_c, ff_g) = self.adaln(cond).chunk(9, dim=-1)
        h = _modulate(self.norm1(x), sa_s, sa_c)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa_g.unsqueeze(1) * sa
        h = _modulate(self.norm2(x), ca_s, ca_c)
        ca, _ = self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_key_pad, need_weights=False)
        x = x + ca_g.unsqueeze(1) * ca
        h = _modulate(self.norm3(x), ff_s, ff_c)
        return x + ff_g.unsqueeze(1) * self.ffn(h)


class DiffusionTransformerHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        d = int(m["d_model"])
        heads = int(m["attention_heads"])
        dropout = float(m["dropout"])
        ffn_mult = int(m.get("ffn_mult", 4))
        self.K = int(config["window"]["K"])
        dc = config.get("diffusion", {}) or {}
        self.train_steps = int(dc.get("train_steps", 1000))
        self.sample_steps = int(dc.get("sample_steps", 20))
        beta_start = float(dc.get("beta_start", 1e-4))
        beta_end = float(dc.get("beta_end", 0.02))
        num_layers = int(dc.get("num_layers", 4))
        betas = torch.linspace(beta_start, beta_end, self.train_steps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

        # Per-dataset target standardization: each window's target is standardized by its own
        # dataset's (mean, std). ``ds_mean``/``ds_std`` are a table over datasets with a trailing
        # pooled-fallback slot for unknown datasets (pooled defaults to (0, 1) = identity pre-fit).
        # Stats are train-only (fit_norm_stats).
        stats_path = (config.get("data", {}) or {}).get("norm_stats")
        pooled_mean, pooled_std = load_label_stats(stats_path)
        per_ds = load_label_stats_per_dataset(stats_path)
        self.ds_names = sorted(per_ds.keys())
        self._ds_index = {name: i for i, name in enumerate(self.ds_names)}
        means = [per_ds[d][0] for d in self.ds_names] + [pooled_mean]
        stds = [per_ds[d][1] for d in self.ds_names] + [pooled_std]
        self.register_buffer("ds_mean", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("ds_std", torch.tensor(stds, dtype=torch.float32))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.in_proj = nn.Linear(1, d)                               # engagement scalar -> token
        self.pos_emb = nn.Parameter(torch.randn(1, self.K, d) * d ** -0.5)  # 1D frame order
        self.ctx_proj = nn.Linear(2 * d, d)                          # X_cond (B,W,2d) -> (B,W,d)
        self.blocks = nn.ModuleList([
            DiTCrossBlock(d, heads, ffn_mult, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_proj = nn.Linear(d, 1)
        nn.init.zeros_(self.out_proj.weight)                         # start predicting ~zero velocity
        nn.init.zeros_(self.out_proj.bias)

    # ── core denoiser ─────────────────────────────────────────────────────────
    def forward(self, y_t: torch.Tensor, timesteps: torch.Tensor, X_cond: torch.Tensor,
                cond_frames_valid: torch.Tensor | None = None) -> torch.Tensor:
        """``y_t: (B, K, 1)``; ``timesteps: (B,)``; ``X_cond: (B, W, 2d)`` -> predicted
        velocity ``v (B, K, 1)``."""
        cond = self.time_mlp(timesteps)                              # (B, d)
        x = self.in_proj(y_t) + self.pos_emb                         # (B, K, d)
        ctx = self.ctx_proj(X_cond)                                  # (B, W, d)
        key_pad = None
        if cond_frames_valid is not None:
            key_pad = cond_frames_valid <= 0.5                       # (B, W) True = ignore
            key_pad = key_pad.clone()
            key_pad[:, 0] = False                                    # keep >=1 key -> no NaN row
        for blk in self.blocks:
            x = blk(x, cond, ctx, key_pad)
        return self.out_proj(self.out_norm(x))                       # (B, K, 1)

    def dataset_stats(self, dataset_ids: list[str], n_repeat: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-node ``(mean, std)`` ``(B·n_repeat, 1)`` for a length-B list of dataset-id strings.

        The head runs on the flattened ``(B·N)`` node axis, so each session's stats are repeated
        over its ``n_repeat`` nodes (``repeat_interleave`` matches the ``(B, N) -> B·N`` flatten).
        Unknown datasets fall back to the trailing pooled slot.
        """
        fallback = len(self.ds_names)
        idx = torch.tensor([self._ds_index.get(d, fallback) for d in dataset_ids],
                           device=self.ds_mean.device, dtype=torch.long)
        idx = idx.repeat_interleave(n_repeat)                         # (B·n_repeat,)
        return self.ds_mean[idx].unsqueeze(1), self.ds_std[idx].unsqueeze(1)

    @staticmethod
    def _v_to_y0_eps(v: torch.Tensor, y_t: torch.Tensor,
                     acp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the velocity parameterization: ``v, y_t -> (ŷ₀, ε̂)`` (all standardized).
        ``v``/``y_t``: ``(B, K)``; ``acp``: ``(B, 1)``."""
        sa, sb = acp.sqrt(), (1.0 - acp).sqrt()
        y0 = sa * y_t - sb * v                                       # v -> y₀
        eps = sb * y_t + sa * v                                      # v -> ε
        return y0, eps

    # ── training / inference wrappers (operate on (B, K) chunks) ───────────────
    def noise_pred(self, X_cond: torch.Tensor, cond_frames_valid: torch.Tensor,
                   y0: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict:
        """Training step: sample ``t`` + noise, form ``y_t``, predict the velocity ``v``.

        Returns a dict with the predicted velocity ``v_pred`` and its target ``v`` ``(B, K)``, the
        noisy chunk ``y_t`` ``(B, K)``, the schedule coefficient ``alpha_bar_t`` ``(B, 1)``, and the
        per-sample standardization stats ``y_mean``/``y_std`` ``(B, 1)``. ``y0`` enters in the raw
        engagement scale and is standardized here by its per-dataset ``(y_mean, y_std)``; ``y_t`` /
        ``v_pred`` are therefore in standardized space.
        """
        B = y0.shape[0]
        y0 = (y0 - y_mean) / y_std                                   # per-dataset -> ~unit variance
        t = torch.randint(0, self.train_steps, (B,), device=y0.device)
        noise = torch.randn_like(y0)
        acp = self.alphas_cumprod[t].unsqueeze(1)                    # (B, 1)
        y_t = acp.sqrt() * y0 + (1.0 - acp).sqrt() * noise           # (B, K)
        v = acp.sqrt() * noise - (1.0 - acp).sqrt() * y0             # velocity target
        v_pred = self.forward(y_t.unsqueeze(-1), t, X_cond, cond_frames_valid).squeeze(-1)
        return {"v_pred": v_pred, "v": v, "y_t": y_t, "alpha_bar_t": acp,
                "y_mean": y_mean, "y_std": y_std}

    @torch.no_grad()
    def sample(self, X_cond: torch.Tensor, cond_frames_valid: torch.Tensor,
               y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        """Inference: deterministic DDIM over ``sample_steps`` -> ``Y_t (B, K)`` in [0, 1].

        Denoising runs in the standardized space the head was trained in; the recovered ŷ₀ is only
        un-standardized by its per-sample ``(y_mean, y_std)`` ``(B, 1)`` (and clamped to the [0, 1]
        engagement range) on the final step, so the intermediate DDIM trajectory is never clipped in
        the wrong space. The predicted velocity is mapped back to ``(ŷ₀, ε̂)`` for the DDIM update.
        """
        B = X_cond.shape[0]
        x = torch.randn(B, self.K, device=X_cond.device)
        steps = torch.linspace(self.train_steps - 1, 0, self.sample_steps).round().long().tolist()
        for i, t_cur in enumerate(steps):
            t = torch.full((B,), t_cur, device=X_cond.device, dtype=torch.long)
            v = self.forward(x.unsqueeze(-1), t, X_cond, cond_frames_valid).squeeze(-1)
            acp_t = self.alphas_cumprod[t_cur].view(1, 1)
            y0, eps = self._v_to_y0_eps(v, x, acp_t)                 # standardized ŷ₀, ε̂
            if i < len(steps) - 1:
                acp_next = self.alphas_cumprod[steps[i + 1]]
                x = acp_next.sqrt() * y0 + (1.0 - acp_next).sqrt() * eps
            else:
                x = (y0 * y_std + y_mean).clamp(0.0, 1.0)            # back to raw engagement scale
        return x
