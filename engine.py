import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from model import WDGSNet



import util
import math


# =========================
# Warmup -> Cosine 调度器（按 epoch 调用 step）
# =========================
class WarmupThenCosine(_LRScheduler):
    """

    """
    def __init__(self, optimizer,
                 warmup_epochs=5,
                 warmup_factor=0.2,
                 T_max=50,
                 eta_min=1e-5,
                 last_epoch=-1):
        assert warmup_epochs >= 0
        assert 0.0 < warmup_factor <= 1.0
        assert T_max > 0


        self.warmup_epochs = int(warmup_epochs)
        self.warmup_factor = float(warmup_factor)
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch + 1


        if self.warmup_epochs > 0 and ep <= self.warmup_epochs:
            scale = self.warmup_factor + (1.0 - self.warmup_factor) * (ep / self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]


        k = ep - self.warmup_epochs
        # eta_t = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(pi * k / T_max))
        lrs = []
        for base_lr in self.base_lrs:
            eta_t = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * (k % self.T_max) / self.T_max))
            lrs.append(eta_t)
        return lrs


# =========================
# EMA
# =========================
class EMA:
    def __init__(self, model, decay=0.999):
        """
        - decay: EMA
        """
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):

        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name in self.shadow:

                new_avg = (1.0 - d) * param.data + d * self.shadow[name]
                self.shadow[name] = new_avg.clone()
            else:

                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def apply_shadow(self, model):

        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class trainer():
    def __init__(
        self,
        batch_size,
        scaler,
        in_dim,
        seq_length,
        num_nodes,
        nhid,
        dropout,
        lrate,
        wdecay,
        supports,
        H_a,
        H_b,
        G0,
        G1,
        indices,
        G0_all,
        G1_all,
        H_T_new,
        lwjl,
        clip=3,
        lr_de_rate=0.97,
        out_dim=1,
        # ===== 训练增强开关 =====
        use_ema=True,
        ema_decay=0.999,   # EMA
        mse_lambda=0.10,
        # ===== Warmup + Cosine =====
        warmup_epochs=5,
        warmup_factor=0.20,
        cosine_Tmax=50,
        eta_min=1e-5
    ):
        self.out_dim     = out_dim
        self.batch_size  = batch_size
        self.scaler      = scaler
        self.in_dim      = in_dim
        self.seq_length  = seq_length
        self.num_nodes   = num_nodes
        self.clip        = clip

        self.model = WDGSNet(
            batch_size=batch_size,
            H_a=H_a,
            H_b=H_b,
            G0=G0,
            G1=G1,
            indices=indices,
            G0_all=G0_all,
            G1_all=G1_all,
            H_T_new=H_T_new,
            lwjl=lwjl,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            in_dim=in_dim,
            out_dim=self.out_dim,
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16
        )
        self.model.cuda()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lrate,
            weight_decay=wdecay
        )

        # ====== LR ：Warmup -> Cosine=====
        self.scheduler = WarmupThenCosine(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            warmup_factor=warmup_factor,
            T_max=cosine_Tmax,
            eta_min=eta_min
        )

        # ====== loss ======
        self.loss_mae   = util.masked_mae
        self.mse_lambda = float(mse_lambda)
        self.base_mse_lambda = float(mse_lambda)  #

        # ====== EMA ======
        self.use_ema = bool(use_ema)
        self.ema = EMA(self.model, decay=ema_decay) if self.use_ema else None


    def _masked_mse_simple(self, pred, true, null_val=0.0):
        mask = (true != null_val).float()
        mask = mask / (mask.mean() + 1e-6)
        loss = (pred - true) ** 2
        loss = loss * mask
        return torch.mean(loss)

    def _r2_score(self, pred, true, null_val=0.0):
        """
        pred/true: [B, 1, N, 1]
        """
        mask = (true != null_val).float()
        m = mask.mean()
        if m <= 0:
            return torch.tensor(0.0, device=pred.device)
        y = true
        y_hat = pred
        y_mean = (y * mask).sum() / (mask.sum() + 1e-6)
        ss_res = ((y - y_hat) ** 2 * mask).sum()
        ss_tot = ((y - y_mean) ** 2 * mask).sum() + 1e-6
        r2 = 1.0 - ss_res / ss_tot
        return r2



    def train(self, input, real_val):
        """One-step training.

        Args:
            input:    [B, N, T, C_in] (BNTC, paper layout)
            real_val: either [B, N, 1] or [B, N, 1, out_dim]
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Pad 1 step on the left along time dimension (paper uses past-window alignment)
        # BNTC => pad on T (dim=2): (pad_C_left, pad_C_right, pad_T_left, pad_T_right)
        input = torch.nn.functional.pad(input, (0, 0, 1, 0))

        output = self.model(input)  # [B, N, 1, out_dim]

        # Align label shape
        if real_val.dim() == 3:
            real = real_val.unsqueeze(-1)  # [B, N, 1, 1]
        elif real_val.dim() == 4:
            real = real_val
        else:
            raise ValueError(f"real_val must be 3D or 4D, got {real_val.dim()}D")

        predict = self.scaler.inverse_transform(output)

        loss = self.loss_mae(predict, real, 0.0)
        if self.mse_lambda > 0.0:
            loss = loss + self.mse_lambda * self._masked_mse_simple(predict, real, 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        if self.use_ema and (self.ema is not None):
            self.ema.update(self.model)

        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    # ====== 单步验证（train.py 负责聚合、打印）======
    @torch.no_grad()
    def eval(self, input, real_val):
        """One-step evaluation (BNTC)."""
        use_shadow = self.use_ema and (self.ema is not None)
        if use_shadow:
            self.ema.apply_shadow(self.model)

        self.model.eval()

        input = torch.nn.functional.pad(input, (0, 0, 1, 0))
        output = self.model(input)  # [B, N, 1, out_dim]

        if real_val.dim() == 3:
            real = real_val.unsqueeze(-1)  # [B, N, 1, 1]
        elif real_val.dim() == 4:
            real = real_val
        else:
            raise ValueError(f"real_val must be 3D or 4D, got {real_val.dim()}D")

        predict = self.scaler.inverse_transform(output)

        loss = self.loss_mae(predict, real, 0.0)
        if self.mse_lambda > 0.0:
            loss = loss + self.mse_lambda * self._masked_mse_simple(predict, real, 0.0)

        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = self._r2_score(predict, real, 0.0).item()
        # print(f"Valid R2: {r2:.4f}")

        if use_shadow:
            self.ema.restore(self.model)

        return loss.item(), mape, rmse
