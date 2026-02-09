import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from timexer import TimeXerBlock


def bntc_to_bcnt(x: torch.Tensor) -> torch.Tensor:

    return x.permute(0, 3, 1, 2).contiguous()

def bcnt_to_bntc(x: torch.Tensor) -> torch.Tensor:

    return x.permute(0, 2, 3, 1).contiguous()

def conv2d_bntc(conv: nn.Module, x_bntc: torch.Tensor) -> torch.Tensor:
    return bcnt_to_bntc(conv(bntc_to_bcnt(x_bntc)))

def bn2d_bntc(bn: nn.Module, x_bntc: torch.Tensor) -> torch.Tensor:
    return bcnt_to_bntc(bn(bntc_to_bcnt(x_bntc)))

def pad_time_left_bntc(x: torch.Tensor, pad_len: int) -> torch.Tensor:
    # replicate pad on the left along time axis
    if pad_len <= 0:
        return x
    left = x[:, :, 0:1, :].repeat(1, 1, pad_len, 1)
    return torch.cat([left, x], dim=2)


# ============================================================
# Basic operators (BNTC)
# ============================================================

class d_nconv(nn.Module):
    """Neighborhood propagation over the node/edge dimension.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B,V,T,C]
        if A.dim() == 3:

            out = torch.einsum('bvtc,bvw->bwtc', x, A)
        elif A.dim() == 2:
            out = torch.einsum('bvtc,vw->bwtc', x, A)
        else:
            raise ValueError(f"A must be 2D or 3D, got {A.dim()}D")
        return out.contiguous()


class linear_(nn.Module):
    """1×1 channel projection used by DGCN/HGCN blocks.

    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()

        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x_bntc: torch.Tensor) -> torch.Tensor:

        return conv2d_bntc(self.mlp, x_bntc)

class DGCN(nn.Module):
    """Dynamic Graph Convolution Network (DGCN) block (paper Eq. (26))."""
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 3, order: int = 2):
        super().__init__()
        self.d_nconv = d_nconv()
        c_in_eff = (order * support_len + 1) * c_in
        self.mlp = linear_(c_in_eff, c_out)
        self.dropout = float(dropout)
        self.order = int(order)

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        # x: [B,N,T,C_in]
        out = [x]
        for a in support:
            x1 = self.d_nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class EdgeTemporalSSM(nn.Module):
    """Edge temporal State Space Module (paper Eq. (30)).

    Input:  x_e [B, E, T, C_in]
    Output: y   [B, E, T, C_out]
    Note: We keep the public layout as BNTC, where N := E (hyperedges).
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.a = nn.Parameter(torch.zeros(1, out_channels, 1, 1))  # stored as [1,C,1,1] for Conv2d; converted to BNTC in forward
        self.b = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.g = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.d = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_e: torch.Tensor) -> torch.Tensor:
        # x_e: [B, E, T, C_in] (BNTC with N:=E)
        x = conv2d_bntc(self.proj_in, x_e)  # [B, E, T, C_out]

        # a: [1,C,1,1] -> [1,1,1,C] for BNTC broadcast
        a = torch.tanh(self.a).permute(0, 2, 3, 1).contiguous()

        B, E, T, C = x.shape
        s = torch.zeros(B, E, 1, C, device=x.device, dtype=x.dtype)

        u = conv2d_bntc(self.b, x)  # [B, E, T, C]
        outs = []
        for t in range(T):
            s = a * s + u[:, :, t:t+1, :]
            y_t = conv2d_bntc(self.g, s) + conv2d_bntc(self.d, x[:, :, t:t+1, :])
            outs.append(y_t)
        y = torch.cat(outs, dim=2)  # [B, E, T, C]

        y = F.gelu(bn2d_bntc(self.bn, y))
        y = self.drop(y) + x
        return y

class HGCN(nn.Module):
    """Hypergraph Convolution in hyperedge domain (paper Eq. (31))."""
    def __init__(self, c_in: int, c_out: int, dropout: float, order: int = 2):
        super().__init__()
        self.d_nconv = d_nconv()
        c_in_eff = (order + 1) * c_in
        self.mlp = linear_(c_in_eff, c_out)
        self.dropout = float(dropout)
        self.order = int(order)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # x: [B,E,T,C_in], G: [E,E] or [B,E,E]
        out = [x]
        x1 = self.d_nconv(x, G)
        out.append(x1)
        for _ in range(2, self.order + 1):
            x2 = self.d_nconv(x1, G)
            out.append(x2)
            x1 = x2
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# ============================================================
# TADG: Topology-Aware Dual Graph Gating (paper Eq. (34)–(41))
# ============================================================

class TADG(nn.Module):
    """Topology-Aware Dual Graph Gating (TADG).

    Inputs:
      - X_g: [B, N, T, C_g]  (DGCN branch)
      - X_h: [B, N, T, C_h]  (ATSH branch)
    Output:
      - Y:   [B, N, T, C_mid]
    """
    def __init__(self, c_g: int, c_h: int, c_mid: int, topo_in: int, dropout: float = 0.0):
        super().__init__()

        self.proj_g = nn.Conv2d(c_g, c_mid, 1, bias=True)
        self.proj_h = nn.Conv2d(c_h, c_mid, 1, bias=True)
        self.proj_topo = nn.Conv2d(topo_in, c_mid, 1, bias=True)
        self.gate = nn.Conv2d(3 * c_mid, 2, 1, bias=True)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x_g: torch.Tensor, x_h: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:

        x_g_p = conv2d_bntc(self.proj_g, x_g)  # [B,N,T,c_mid]
        x_h_p = conv2d_bntc(self.proj_h, x_h)  # [B,N,T,c_mid]

        # align time
        Tg, Th = x_g_p.size(2), x_h_p.size(2)
        if Tg != Th:
            T = min(Tg, Th)
            x_g_p = x_g_p[:, :, -T:, :]
            x_h_p = x_h_p[:, :, -T:, :]

        # temporal mean pool (Eq. 36–37) -> [B,N,1,C]
        Xg_bar = x_g_p.mean(dim=2, keepdim=True)
        Xh_bar = x_h_p.mean(dim=2, keepdim=True)

        topo = topo.to(x_g_p.device).type_as(x_g_p)
        topo_p = conv2d_bntc(self.proj_topo, topo)  # [B,N,1,c_mid]

        # concat then 1×1 conv then softmax (Eq. 38–40)
        cat_ctx = torch.cat([Xg_bar, Xh_bar, topo_p], dim=-1)  # [B,N,1,3*c_mid]
        w = torch.softmax(conv2d_bntc(self.gate, cat_ctx), dim=-1)  # [B,N,1,2]

        w_g = w[..., 0:1]  # [B,N,1,1]
        w_h = w[..., 1:2]

        # Broadcast weights to time and fuse (Eq. 41)
        Y = w_g * x_g_p + w_h * x_h_p
        return self.dropout(Y)


# ============================================================
# SADGC: Shared Attention Directed Graph Construction (Eq. 9–24)
# ============================================================

class SADGC(nn.Module):
    """Shared Attention Directed Graph Construction (SADGC) Module."""
    def __init__(self, in_channels: int, num_nodes: int, num_timesteps: int,
                 topk: int = 3, dropedge: float = 0.05):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_nodes = int(num_nodes)
        self.num_timesteps = max(1, int(num_timesteps))
        self.topk = int(topk) if topk is not None else 0
        self.dropedge = float(dropedge)

        r = max(1, self.in_channels // 4)

        self.W_shared = nn.Conv2d(self.in_channels, r, kernel_size=1, bias=False)
        self.W_phi = nn.Conv2d(r, self.in_channels, kernel_size=1, bias=True)
        self.W_mu = nn.Conv2d(r, 1, kernel_size=1, bias=True)

        self.d = max(1, self.in_channels // 2)
        self.W_Q = nn.Linear(self.in_channels, self.d, bias=False)
        self.W_K = nn.Linear(self.in_channels, self.d, bias=False)
        self.scale = math.sqrt(self.d)

    @staticmethod
    def _row_topk(A: torch.Tensor, k: int) -> torch.Tensor:
        if k is None or k <= 0:
            return A
        B, N, _ = A.shape
        k = max(1, min(int(k), N))
        _, idx = torch.topk(A, k=k, dim=-1)
        mask = torch.zeros_like(A).scatter_(-1, idx, 1.0)
        return A * mask

    def forward(self, x_wav: torch.Tensor, indices=None):
        # x_wav: [B, N, T, C]
        B, N, T, C = x_wav.shape
        T_delta = min(self.num_timesteps, T)

        # (9) temporal slicing
        X_w = x_wav[:, :, -T_delta:, :]  # [B, N, Td, C]

        # (10) channel attention
        avg = X_w.mean(dim=2, keepdim=True)  # [B, N, 1, C]
        z_ca = F.relu(conv2d_bntc(self.W_shared, avg))      # [B, N, 1, r]
        w = torch.sigmoid(conv2d_bntc(self.W_phi, z_ca))    # [B, N, 1, C]
        X_hat = X_w * w                                    # [B, N, Td, C]

        # (11) time attention along Td
        z_ta = F.relu(conv2d_bntc(self.W_shared, X_hat))    # [B, N, Td, r]
        alpha = torch.softmax(conv2d_bntc(self.W_mu, z_ta).squeeze(-1), dim=2)  # [B, N, Td]

        # (12) node-context representation
        X_c = (alpha.unsqueeze(-1) * X_hat).sum(dim=2)      # [B, N, C]

        # (13)-(16) forward/backward edge scores
        Q = self.W_Q(X_c)  # [B, N, d]
        K = self.W_K(X_c)  # [B, N, d]
        E_f = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [B, N, N]
        E_b = torch.matmul(K, Q.transpose(1, 2)) / self.scale  # [B, N, N]

        S_f = torch.sigmoid(E_f)
        S_b = torch.sigmoid(E_b)

        # (19)-(20) Top-K sparsification
        S_f = self._row_topk(S_f, self.topk)
        S_b = self._row_topk(S_b, self.topk)

        # (21)-(22) DropEdge
        if self.training and self.dropedge > 0.0:
            keep = (torch.rand_like(S_f) > self.dropedge).float()
            S_f = S_f * keep
            keep = (torch.rand_like(S_b) > self.dropedge).float()
            S_b = S_b * keep

        return S_f, S_b

class MCWP(nn.Module):
    """Multi-Level Cascaded Wavelet Processing Module (MCWP).
    """
    def __init__(self, channels: int):
        super().__init__()
        h = 1.0 / math.sqrt(2.0)
        self.register_buffer('haar_low',  torch.tensor([h,  h],  dtype=torch.float32).view(1, 1, 1, 2))
        self.register_buffer('haar_high', torch.tensor([h, -h], dtype=torch.float32).view(1, 1, 1, 2))

        # 1×1 conv projections for components (paper Eq. (3)–(4))
        self.wproj_A3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.wproj_D1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.wproj_D2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.wproj_D3 = nn.Conv2d(channels, channels, kernel_size=1)

        # learnable fusion weights (paper Eq. (7))
        self.alpha_A3 = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.alpha_D1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.alpha_D2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.alpha_D3 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _dwt1_bntc(self, x_bntc: torch.Tensor):
        """One-level Haar DWT along the temporal dimension.
        """
        x_int = bntc_to_bcnt(x_bntc)  # internal [B,C,N,T]
        B, C, N, T = x_int.shape
        weight_l = self.haar_low.repeat(C, 1, 1, 1)   # [C,1,1,2]
        weight_h = self.haar_high.repeat(C, 1, 1, 1)
        L = F.conv2d(x_int, weight=weight_l, bias=None, stride=(1, 2), padding=(0, 0), groups=C)
        H = F.conv2d(x_int, weight=weight_h, bias=None, stride=(1, 2), padding=(0, 0), groups=C)
        return bcnt_to_bntc(L), bcnt_to_bntc(H)

    @staticmethod
    def _linear_interp_time_bntc(z_bntc: torch.Tensor, tgt_T: int) -> torch.Tensor:
        """Linear interpolation along time for BNTC tensor."""
        B, N, T_src, C = z_bntc.shape
        if T_src == tgt_T:
            return z_bntc

        z1 = z_bntc.permute(0, 1, 3, 2).contiguous().view(B * N, C, T_src)
        z1 = F.interpolate(z1, size=tgt_T, mode='linear', align_corners=False)
        z2 = z1.view(B, N, C, tgt_T).permute(0, 1, 3, 2).contiguous()
        return z2

    def forward(self, x_p: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x_p: [B, N, T, C]
        Returns:
            x_wav: [B, N, T, C]
        """
        B, N, T, C = x_p.shape

        pad = (8 - (T % 8)) % 8
        if pad > 0:
            left = x_p[:, :, :1, :].repeat(1, 1, pad, 1)  # [B,N,pad,C]
            x_pad = torch.cat([left, x_p], dim=2)         # [B,N,T_pad,C]
        else:
            x_pad = x_p
        T_pad = x_pad.size(2)

        # (2) multi-level Haar decomposition
        A1, D1 = self._dwt1_bntc(x_pad)
        A2, D2 = self._dwt1_bntc(A1)
        A3, D3 = self._dwt1_bntc(A2)

        # (3)-(4) 1×1 projections at each scale (implemented via Conv2d helper)
        A3_hat = conv2d_bntc(self.wproj_A3, A3)
        D1_hat = conv2d_bntc(self.wproj_D1, D1)
        D2_hat = conv2d_bntc(self.wproj_D2, D2)
        D3_hat = conv2d_bntc(self.wproj_D3, D3)

        # (5)-(6) interpolate back to unified temporal length
        A3_prime = self._linear_interp_time_bntc(A3_hat, T_pad)
        D1_prime = self._linear_interp_time_bntc(D1_hat, T_pad)
        D2_prime = self._linear_interp_time_bntc(D2_hat, T_pad)
        D3_prime = self._linear_interp_time_bntc(D3_hat, T_pad)

        # (7) learnable weighted fusion + residual
        x_sum = (
            self.alpha_A3 * A3_prime
            + self.alpha_D1 * D1_prime
            + self.alpha_D2 * D2_prime
            + self.alpha_D3 * D3_prime
        )
        x_wav = x_pad + x_sum

        # crop back to original length
        if pad > 0:
            x_wav = x_wav[:, :, -T:, :]
        return x_wav

class WDGSNet(nn.Module):
    def __init__(
        self,
        batch_size,
        H_a,
        H_b,
        G0,
        G1,
        indices,
        G0_all,
        G1_all,
        H_T_new,
        lwjl,
        num_nodes,
        dropout=0.3,
        supports=None,
        in_dim=4,
        out_dim=1,
        residual_channels=40,
        dilation_channels=40,
        skip_channels=320,
        end_channels=640,
        kernel_size=2,
        blocks=2,
        layers=1,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.H_a = H_a
        self.H_b = H_b
        self.G0 = G0
        self.G1 = G1
        self.H_T_new = H_T_new
        self.lwjl = lwjl
        self.indices = indices
        self.G0_all = G0_all
        self.G1_all = G1_all

        self.dropout = float(dropout)
        self.blocks = int(blocks)
        self.layers = int(layers)

        self.supports = supports if supports is not None else []
        self.num_nodes = int(num_nodes)

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        # MCWP
        self.mcwp = MCWP(residual_channels)

        # SADGC fusion coefficients (Eq. 23–24)
        self.lambda_f = nn.Parameter(torch.tensor(0.5))
        self.lambda_b = nn.Parameter(torch.tensor(0.5))

        # Adaptive graph for baseline supports
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)

        # per-layer modules (DGST-Net stacking)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.dgcn_layers = nn.ModuleList()
        self.atsh_hgcn_layers = nn.ModuleList()
        self.edge_ssm = nn.ModuleList()
        self.bn_g = nn.ModuleList()
        self.bn_hg = nn.ModuleList()

        self.sadgc_layers = nn.ModuleList()
        self.tadg_layers = nn.ModuleList()

        # ATSH incidence weights
        self.alpha_loc = nn.Parameter(torch.tensor(1.0))
        self.alpha_reg = nn.Parameter(torch.tensor(1.0))

        # Graph support length (static fwd/back + adaptive)
        self.graph_support_k = min(3, len(self.supports) + 1)

        receptive_field = 1
        supports_len = len(self.supports) + 1  # + adaptive

        for b in range(self.blocks):
            additional_scope = kernel_size
            new_dilation = 2
            for i in range(self.layers):
                # TCN on nodes (implemented as Conv2d on [B,C,N,T])
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                receptive_field += (additional_scope * 2)

                # DGCN branch output C' = residual/2
                c_half = int(residual_channels / 2)
                self.dgcn_layers.append(
                    DGCN(dilation_channels, c_half, dropout, support_len=self.graph_support_k)
                )
                self.bn_g.append(nn.BatchNorm2d(c_half))

                # Edge temporal SSM + HGCN (ATSH branch)
                self.edge_ssm.append(EdgeTemporalSSM(residual_channels, dilation_channels, dropout=self.dropout))
                self.atsh_hgcn_layers.append(HGCN(dilation_channels, c_half, dropout))
                self.bn_hg.append(nn.BatchNorm2d(c_half))

                # SADGC per layer (but we can compute once and share)
                num_timesteps = max(1, int(13 - (receptive_field - additional_scope * 2) + 1))
                self.sadgc_layers.append(
                    SADGC(in_channels=residual_channels, num_nodes=num_nodes, num_timesteps=num_timesteps, topk=3, dropedge=0.1)
                )

                # TADG fusion (uses an explicit topology context: [B,N,1,2] from lambda_f/lambda_b)
                self.tadg_layers.append(
                    TADG(c_g=c_half, c_h=c_half, c_mid=residual_channels, topo_in=3, dropout=dropout)
                )

        # output head (kept minimal; if you already removed extra heads, this is the core head)
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)

        # paper baseline: linear projection (Conv1x1) on last step input
        self.linear_baseline = nn.Conv2d(in_channels=in_dim, out_channels=end_channels, kernel_size=1, bias=True)
        self.baseline_alpha = nn.Parameter(torch.tensor(0.2))

        # optional station bias calibration (safe to keep; set to 0 init)
        self.node_calib_bias = nn.Parameter(torch.zeros(1, out_dim, num_nodes, 1))

        self.receptive_field = receptive_field

        # TimeXer
        self.timexer = TimeXerBlock(
            patch_size=6,
            in_dim=residual_channels,
            d_model=residual_channels,
            num_heads=4,
            dropout=dropout,
        )

        self.bn_start = nn.BatchNorm2d(in_dim, affine=False)

        # container for shared dynamic supports
        self.new_supports_w = [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]),
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]),
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]),
        ]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # For compatibility, we accept either:
        #   - BNTC: [B, N, T, C_in]
        #   - [B,C,N,T]: [B, C_in, N, T]
        in_dim = self.start_conv.in_channels
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input in paper layout [B,N,T,C], got {input.dim()}D")

        # Strict paper layout: [B, N, T, C_in]
        if input.size(-1) != in_dim:
            raise ValueError(
                f"Expected input in BNTC layout [B,N,T,C_in] with C_in={in_dim}, got shape {tuple(input.shape)}"
            )

        x = input.contiguous()                 # [B,N,T,C_in]

        x = bn2d_bntc(self.bn_start, x)        # [B,N,T,C_in]
        x = conv2d_bntc(self.start_conv, x)    # [B,N,T,C]


        # MCWP
        x = self.mcwp(x)  # [B,N,T,C]

        # SADGC (build once and share across layers)
        A_fwd_dyn, A_bwd_dyn = self.sadgc_layers[0](x, self.indices)  # [B,N,N]

        # static adjacency from supports (predefined static adjacency matrix)
        B_cur = x.size(0)
        S0 = self.supports[0].to(x.device)  # [N,N]
        S1 = self.supports[1].to(x.device)  # [N,N]
        S0B = S0.unsqueeze(0).repeat(B_cur, 1, 1)
        S1B = S1.unsqueeze(0).repeat(B_cur, 1, 1)
        lambda_f = torch.sigmoid(self.lambda_f)
        lambda_b = torch.sigmoid(self.lambda_b)
        A_fwd_shared = (1.0 - lambda_f) * S0B + lambda_f * A_fwd_dyn
        A_bwd_shared = (1.0 - lambda_b) * S1B + lambda_b * A_bwd_dyn

        # adaptive adjacency (node embedding)
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # [N,N]
        adp_new = adp.unsqueeze(0).repeat(B_cur, 1, 1)

        self.new_supports_w[0] = A_fwd_shared
        self.new_supports_w[1] = A_bwd_shared
        self.new_supports_w[2] = adp_new

        # ATSH hypergraph structure (from provided H_a/H_b) — computed once per forward
        H_loc = self.H_a.to(x.device).float()  # [E_loc, N]
        H_reg = self.H_b.to(x.device).float()  # [E_reg, N]
        H_hat_T = torch.cat([self.alpha_loc * H_loc, self.alpha_reg * H_reg], dim=0)  # [E', N]
        H_hat = H_hat_T.t().contiguous()  # [N, E']
        D_e = H_hat_T.sum(dim=1).clamp_min(1.0)  # [E']
        D_v = H_hat.sum(dim=1).clamp_min(1.0)    # [N]
        H_dv = H_hat / D_v.view(-1, 1)  # [N, E']
        G_hg = torch.matmul(H_hat_T / D_e.view(-1, 1), H_dv)  # [E', E']

        # topology context for TADG: use the two fusion coefficients (broadcasted)
        # topology cues for TADG: (deg_f, deg_b, deg_h) as in the paper figure
        deg_f = A_fwd_shared.sum(dim=-1)                  # [B,N]
        deg_b = A_bwd_shared.sum(dim=-1)                  # [B,N]
        deg_h = D_v.view(1, -1).repeat(B_cur, 1)          # [B,N]
        topo_ctx = torch.stack([deg_f, deg_b, deg_h], dim=-1).unsqueeze(2)  # [B,N,1,3]

        # TimeXer on node-wise series (paper: temporal encoding)
        B, N, T_all, C = x.shape
        patch = 6
        win = min(42, T_all)
        win_aligned = (win // patch) * patch
        if win_aligned == 0:
            if T_all >= patch:
                win_aligned = (T_all // patch) * patch
            else:
                pad_len = patch - T_all
                x = pad_time_left_bntc(x, pad_len)
                B, N, T_all, C = x.shape
                win_aligned = patch

        residual = x[:, :, -win_aligned:, :]  # [B,N,T,C]
        B, N, T, C = residual.shape
        x_in = residual.reshape(B * N, T, C)  # [B*N,T,C]
        x_out = self.timexer(x_in)            # [B*N,T,C]
        x = x_out.reshape(B, N, T, C)


        skip = None

        # DGST-Net stacking
        for i in range(self.blocks * self.layers):

            residual_g = x


            filt = torch.tanh(conv2d_bntc(self.filter_convs[i], residual_g))
            gate = torch.sigmoid(conv2d_bntc(self.gate_convs[i], residual_g))
            x_g = filt * gate  # [B,N,T,dilC]


            x_g = self.dgcn_layers[i](x_g, self.new_supports_w[: self.graph_support_k])  # [B,N,T,C']
            x_g = bn2d_bntc(self.bn_g[i], x_g)

            # ATSH branch:

            x_e = torch.einsum('bntc,en->betc', x, H_hat_T)  # [B,E,T,C]
            x_e = x_e / D_e.view(1, -1, 1, 1)

            # edge temporal SSM
            x_e = self.edge_ssm[i](x_e)  # [B,E,T,dilC]

            # hyperedge domain conv (Eq. 31)
            x_u = self.atsh_hgcn_layers[i](x_e, G_hg)  # [B,E,T,C']
            x_u = bn2d_bntc(self.bn_hg[i], x_u)

            # Hyperedge -> Vertex (Eq. 33): Y_h = D_v^{-1} H X_u
            x_h = torch.einsum('betc,ne->bntc', x_u, H_hat)  # [B,N,T,C']
            x_h = x_h / D_v.view(1, -1, 1).unsqueeze(-1)

            # TADG fusion (paper Eq. 34–41)
            x_m = self.tadg_layers[i](x_g, x_h, topo_ctx)  # [B,N,T,C]

            Tm = x_m.size(2)
            x_res = residual_g[:, :, -Tm:, :]
            x_m = x_m + x_res
            x_m = bn2d_bntc(self.bn[i], x_m)

            # skip
            s = conv2d_bntc(self.skip_convs[i], x_m)  # [B,N,T,skipC]
            if skip is None:
                skip = s
            else:
                Ts = s.size(2)
                skip = skip[:, :, -Ts:, :] + s

            x = x_m

# head (paper layout: BNTC throughout)
        y_nl = F.relu(conv2d_bntc(self.end_conv_1, F.relu(skip)))  # [B,N,T,endC]

        # baseline uses the tail frame in paper layout: [B,N,1,C_in]
        x_tail = input[:, :, -1:, :].contiguous()
        baseline = conv2d_bntc(self.linear_baseline, x_tail)  # [B,N,1,endC]

        y0 = conv2d_bntc(self.end_conv_2, y_nl + self.baseline_alpha * baseline)  # [B,N,T,out_dim]

        y0 = y0 + self.node_calib_bias.permute(0, 2, 3, 1)

        return y0
