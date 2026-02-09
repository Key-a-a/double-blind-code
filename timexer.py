import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_dim, d_model):
        super().__init__()
        self.patch_size = patch_size
        # Linear 层映射：patch_size * in_dim → d_model
        self.proj = nn.Linear(patch_size * in_dim, d_model)

    def forward(self, x):
        # x: [B, T, D]  D==in_dim
        B, T, D = x.shape
        ps = self.patch_size


        pad_len = (-T) % ps
        if pad_len > 0:

            pad = x[:, :1, :].repeat(1, pad_len, 1)
            x = torch.cat([pad, x], dim=1)
            T = T + pad_len

        num_patches = T // ps
        x = x.reshape(B, num_patches, ps * D)
        return self.proj(x)


class TimeXerBlock(nn.Module):
    def __init__(self, patch_size, in_dim, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_size, in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )



    def forward(self, x):
        B, T, D = x.shape

        patches = self.patch_embed(x)  # [B, P, d_model]
        patches_t = patches.permute(1, 0, 2)  # [P, B, d]
        attn_out_t = self.transformer(patches_t)  # [P, B, d]
        attn_out = attn_out_t.permute(1, 0, 2)  # [B, P, d]

        x_rec = attn_out.repeat_interleave(self.patch_embed.patch_size, dim=1)  # [B, P*patch_size, d]

        x_rec = x_rec[:, :T, :]  # [B, T, d_model]

        x_rec_t = x_rec.permute(1, 0, 2)  # [T, B, d]
        attn2_t, _ = self.cross_attn(x_rec_t, x_rec_t, x_rec_t)
        attn2 = attn2_t.permute(1, 0, 2)  # [B, T, d_model]
        x2 = self.norm1(x_rec + attn2)


        x3 = self.norm2(x2 + self.ff(x2))
        return x3
