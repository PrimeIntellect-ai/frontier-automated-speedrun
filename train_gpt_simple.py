"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
import math
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    # Newton-Schulz iteration for the orthogonal factor of the polar decomposition.
    # Classic Muon coefficients, 12 iters (well-converged; the 5-iter optimized set
    # 3.4445/-4.7750/2.0315 screened ~1.5σ worse: under-converges vs 12 classic iters).
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True, update_space=False):
    if update_space:
        # Update-space momentum: NS each gradient, then smooth the orthogonal
        # factors. Standard Muon smooths gradients THEN NS (non-linear op). Here NS
        # is applied to each raw gradient (same NS cost, one per step), then the
        # orthogonal factors are EMA-smoothed on the unit-norm manifold.
        u = zeropower_via_newtonschulz5(grad).to(grad.dtype)
        u *= max(1, grad.size(-2) / grad.size(-1))**0.5
        momentum.lerp_(u, 1 - mu)
        update = u.lerp_(momentum, mu) if nesterov else momentum
        return update
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, lr_mult=None, update_space=False):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, update_space=update_space)
        super().__init__(params, defaults)
        # optional per-param lr multiplier (layer-wise lr); keyed by param object
        self._lr_mult = dict(lr_mult) if lr_mult is not None else {}

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"], update_space=group.get("update_space", False))
                    mult = self._lr_mult.get(p, 1.0)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"] * mult)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


@torch.compile
def lion_update(grad, momentum, beta1=0.9, beta2=0.99):
    # Lion: update = sign(beta1*m + (1-beta1)*g); then m = beta2*m + (1-beta2)*g.
    # Dual-beta sign-based optimizer (different family from AdamW's m/sqrt(v)).
    update = (beta1 * momentum + (1 - beta1) * grad).sign_()
    momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
    return update

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = lion_update(p.grad, state["momentum"], beta1=beta1, beta2=beta2)
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)


@torch.no_grad()
def nadam_step_(p, grad, state, lr, betas, eps, weight_decay):
    # NAdam (Dozat 2016): Adam + Nesterov momentum shift. Same update scale as
    # AdamW (m/sqrt(v)), so AdamW lrs transfer directly. Nesterov often speeds
    # convergence in the few-step regime (Muon already uses Nesterov; this makes
    # the AdamW-path consistent). Bias correction via decayed b-tensors
    # (compile-safe, no python step counter in the graph).
    b1, b2 = betas
    g = grad.float()
    if len(state) == 0:
        state["m"] = torch.zeros_like(g)
        state["v"] = torch.zeros_like(g)
        state["b1"] = torch.tensor(b1, dtype=torch.float32, device=g.device)
        state["b2"] = torch.tensor(b2, dtype=torch.float32, device=g.device)
    m, v = state["m"], state["v"]
    m.lerp_(g, 1.0 - b1)                                # 1st moment of g
    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)           # 2nd moment of g
    state["b1"].mul_(b1); state["b2"].mul_(b2)
    bc1 = 1.0 - state["b1"]; bc2 = 1.0 - state["b2"]
    m_nesterov = (1.0 - b1) * g + b1 * m / bc1          # Nesterov-shifted m_hat
    v_hat = v / bc2
    update = m_nesterov / (v_hat.sqrt() + eps)
    p.mul_(1.0 - lr * weight_decay)
    p.add_(update.to(p.dtype), alpha=-lr)

class NAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]; betas = group["betas"]; eps = group["eps"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                nadam_step_(p, p.grad, self.state[p], lr, betas, eps, wd)


@torch.no_grad()
def adan_step_(p, grad, state, lr, betas, eps, weight_decay):
    # Adan: adaptive Nesterov with 2nd moment of gradient DIFFERENCES.
    # Different family from AdamW (m/sqrt(v)) — designed for fast convergence in
    # few steps (the undertrained regime here). Bias correction via decayed
    # b-tensors (compile-safe, no python step counter in the graph).
    b1, b2, b3 = betas
    g = grad.float()
    if len(state) == 0:
        state["m"] = torch.zeros_like(g)
        state["v"] = torch.zeros_like(g)
        state["n"] = torch.zeros_like(g)
        state["prev"] = torch.zeros_like(g)
        state["b1"] = torch.tensor(b1, dtype=torch.float32, device=g.device)
        state["b2"] = torch.tensor(b2, dtype=torch.float32, device=g.device)
        state["b3"] = torch.tensor(b3, dtype=torch.float32, device=g.device)
    m, v, n, prev = state["m"], state["v"], state["n"], state["prev"]
    diff = g - prev
    m.lerp_(g, 1.0 - b1)                                   # 1st moment of g
    v.lerp_(diff, 1.0 - b2)                                # 1st moment of (g - g_prev)
    n.mul_(b3).addcmul_(diff, diff, value=1.0 - b3)        # 2nd moment of (g - g_prev)
    state["b1"].mul_(b1); state["b2"].mul_(b2); state["b3"].mul_(b3)
    bc1 = 1.0 - state["b1"]
    bc2 = 1.0 - state["b2"]
    bc3 = 1.0 - state["b3"]
    update = (m / bc1 + (1.0 - b1) * v / bc2) / ((n / bc3).sqrt() + eps)
    prev.copy_(g)
    p.mul_(1.0 - lr * weight_decay)
    p.add_(update.to(p.dtype), alpha=-lr)

class Adan(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]; betas = group["betas"]; eps = group["eps"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                adan_step_(p, p.grad, self.state[p], lr, betas, eps, wd)


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


# Optional W&B logging (rank 0, opt-in). Enabled when W&B is configured
# (WANDB_API_KEY set, or WANDB_MODE=offline/online); a silent no-op otherwise, so
# autonomous runs never block on a login prompt. Project is fixed; set WANDB_ENTITY
# to the org. This is fixed logging infrastructure — leave it in place.
_wandb_on = dist.get_rank() == 0 and bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE"))

num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for trial_idx in range(num_trials):

    # Fixed per-trial seed (frozen infra): trial_idx 0..7 are the 8 canonical validation
    # seeds 0xC0FFEE+0..7, and a record is always validated on these same 8. Logged as
    # seed:<n> for oversight. Don't re-seed to pin or cherry-pick a lucky draw.
    seed = 0xC0FFEE + trial_idx
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print0(f"seed:{seed}", console=True)

    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # Minimize this while the 8-trial mean still clears the bar (< 3.27859).
    # Baseline anchor: the stock recipe clears the bar at 3290 (confirm with `bash run.sh 8`).
    train_steps = 3150      # S8-1 SCREEN at 3150 (S8-1 stable_frac_adamw=0.40 stacked). Ref@3160 mean=3.27803.

    # SWA: decay LR to a floor (not 0) then hold a flat tail; evaluate on an EMA of
    # the weights. A flat-tail LR makes params oscillate around the optimum, and
    # averaging that oscillation (Polyak/SWA) finds a wider minimum -> lower val
    # loss. (A plain EMA under a decay-to-0 schedule just lags, as prior runs found.)
    # SWA averaging mode:
    #   "exponential" -> EMA with decay ema_decay throughout (record config).
    #   "uniform"      -> canonical SWA: exp EMA before swa_start, then a uniform
    #                     running average over the flat tail (well-centered, no exp lag,
    #                     uses ALL flat-tail samples equally -> lower variance).
    swa_mode = "exponential"   # "exponential" = record EMA(0.99); "uniform" = canonical SWA (tested, worse)
    ema_decay = 0.99          # record; S5-4: 0.985 worse +0.0013. 0.98/0.985/0.992/0.995 all worse. 0.99 is the clear peak.
    ema_decay_descent = 0.99   # 0.99 = off (use ema_decay throughout = record). SCREEN time-varying EMA was NEUTRAL (+0.0002): flat-tail EMA has only 0.077 weight on swa_start, so descent-phase lag reduction is washed out.
    # Reset EMA to raw weights at swa_start. Rationale: at swa_start the continuous
    # EMA still lags from the descent phase (averaging older worse weights → higher
    # loss than the raw model). Resetting to the raw weights replaces a lagging
    # initial value with a better one. The lagging EMA(2944) has weight 0.99^256≈0.077
    # in the final EMA, so benefit ≈ 0.077 × lag ≈ 0.077 × 0.006 ≈ +0.0005.
    ema_reset_at_swa = False  # S7-3 TESTED neutral (-0.00001): Muon EMA has no descent-phase lag (flat-tail oscillation dominates); only embed (monotonic descent) benefits from reset. Keep False (clean, continuous EMA from step 0).
    ema_reset_embed_at_swa = True  # S6-1: reset ONLY embed EMA at swa_start. 204bcb35 config: seed0@3200=3.27558 (-0.00048 vs clean 3.27606). Embed descends -> EMA lags; reset removes descent-phase lag. Muon oscillates -> keep full-history EMA.
    # Decoupled EMA decay: the embed (AdamW, lr 0.7) descends fast during the flat
    # tail, so a 0.99 EMA on it lags; a faster EMA on AdamW params reduces that lag
    # while keeping 0.99 on Muon params (which oscillate and benefit from averaging).
    decouple_ema = False      # S3-4: faster EMA for AdamW HURT (+0.0012) with adamw_floor=0.15. Embed oscillation needs slow EMA(0.99) to average. KEEP False.
    ema_decay_adamw = 0.95   # faster EMA for AdamW params (embed/proj/1D); window ~20
    ema_decay_muon = 0.99    # record EMA for Muon (2D block) params; window ~100
    # Faster EMA for embed ONLY (isolated from the failed all-AdamW decouple_ema).
    # ema_reset_embed wipes the pre-0.92 embed EMA, so this decay only matters in
    # the flat tail. TESTED 0.97 worse +0.00036: 0.99 (more averaging) > 0.97 (less
    # lag) for embed — ema_reset_embed already handles lag; oscillation averaging
    # is the remaining value. 0.99 peak for embed too (0.992 worse per S3-5).
    # Evaluate with the RAW embed (no EMA) for embed, EMA for the rest. The embed
    # DESCENDS monotonically (undertrained) so its EMA LAGS it (averages older worse
    # embed states); the Muon params OSCILLATE so their EMA is the wide-min center
    # (better than raw). Using raw embed removes the lag while keeping SWA on Muon.
    # Cost: raw embed is one flat-tail oscillation sample (noisier than its EMA).
    eval_raw_embed = False  # TESTED neutral +0.0004: ema_reset_embed already fixes embed lag; raw adds oscillation noise
    swa_start = 0.88          # S7-2 re-confirmed PEAK: 0.90 worse +0.00055 (flat-tail SWA averaging > extra training). S3-6 originally found 0.88 peak; holds with new stack.
    swa_start_adamw = 0.92   # S6-2 LOCKED: per-optimizer flat-tail start for AdamW. Peak: 0.88=3.27558, 0.90=3.27543, 0.92=3.27538(BEST), 0.94=3.27557. More high-LR decay for embed helps; too short flat tail (0.94) hurts.
    swa_floor = 0.10          # LR fraction held during the flat tail (Muon) — record constant value
    swa_floor_start = 0.15   # S4-3 decay mode: Muon floor at start of flat tail (explore: fast descent + oscillation)
    swa_floor_end = 0.05     # S4-3 decay mode: Muon floor at end (converged -> low descent -> low EMA lag, no extrapolation)
    decay_kind = "linear"      # "linear" or "cosine" decay from 1.0 to swa_floor
    decay_power = 1.0        # power for linear decay shape: shape=t^p (1=linear record; 2=quadratic worse+0.0061; 0.5 convex worse+0.0021; cosine neutral)
    stable_frac = 0.30       # record; 0.25 worse(+0.0006) even WITH layerwise (S3-13 worse w/o). 0.20 worse, 0.40/0.50 neutral. LOCKED 0.30.
    # Decoupled stable phase for AdamW (embed/proj/1D). Embed is undertrained — a longer
    # stable phase (more full-LR embed training) may improve convergence. ema_reset_embed
    # wipes pre-0.92 EMA, so this only affects the raw embed quality at swa_start_adamw.
    # stable_frac_adamw = stable_frac → off (record). 0.40 = more full-LR embed training.
    stable_frac_adamw = 0.40   # S8-1 SCREEN: longer stable phase for AdamW (embed benefit)
    # Decoupled schedule: AdamW (embed/proj/1D) uses its own floor, decaying to a
    # lower LR than Muon and effectively freezing during the flat tail. Rationale:
    # the embed (lr 0.7) is still descending fast during the flat tail, so the EMA
    # on it lags; freezing AdamW params there lets the EMA capture converged
    # weights (no lag) while Muon keeps the flat-tail oscillation SWA averages.
    decouple_adamw = True     # S3-1/2/3: separate adamw_floor. Peak=0.15 (-0.00065 vs record); 0.20 same; 0.30 worse. LOCKED 0.15.
    adamw_floor = 0.15        # S3 record: AdamW flat-tail+decay floor (0.10 shared=record; 0.15 decoupled=-0.00065)
    # Zero weight decay during the flat tail. The flat-tail wd causes ~5% weight
    # shrinkage (384 steps * lr*wd) that biases the EMA downward. Zeroing it lets
    # the EMA average the true oscillating weights.
    zero_wd_flat_tail = True  # S6-1: zero Muon wd during flat tail (204bcb35 config). Prevents ~5% weight shrinkage that biases EMA downward.
    zero_adamw_wd_flat_tail = False  # S7-1 TESTED worse +0.00042: embed wd during flat tail provides needed regularization (embed descends, not oscillates). KEEP Muon-only zero_wd.
    # Flat-tail Muon momentum boost. In the low-LR oscillating flat tail, the raw
    # gradient is noisy; a higher mu (longer EMA window) smooths it before the NS
    # orthogonalization -> a cleaner orthogonal factor -> tighter oscillation
    # around the minimum -> the EMA captures a better-centered wide minimum.
    # 0 = off (mu=0.95 throughout = record); only applied to Muon (2D) in flat tail.
    flat_tail_mu = 0  # TESTED both directions: 0.98 worse +0.0031 (over-smooth), 0.90 worse +0.0016 (too noisy). mu=0.95 is the sweet spot throughout. LOCKED off.
    swa_tail = "flat"        # "flat" (record) / "cycle" (worse+0.0013) / "decay" (S4-3 TESTED worse +0.00045: embed dominates flat-tail descent so Muon-floor decay can't cut lag)
    cycle_mid = 0.125         # mean LR of the flat-tail cycle
    cycle_amp = 0.075         # cycle amplitude (range [mid-amp, mid+amp] = [0.05, 0.20])
    cycle_n = 4              # number of cycles across the flat tail
    # Trend-corrected EMA (Holt linear-trend extrapolation). The EMA(0.99) has mean
    # lag ~ema_decay/(1-ema_decay)=99 steps; the flat-tail loss is still descending
    # (~5.5e-5/step), so that lag costs ~0.005 in val loss. We snapshot the EMA late in
    # the flat tail and extrapolate forward: est = ema + alpha*(ema - snap), removing
    # the lag. The trend term (ema - snap) is common-mode: flat-tail oscillation cancels
    # in the difference, so it adds almost no variance (~4e-5). 0 = off (plain EMA).
    trend_alpha = 0.0            # 0 = plain EMA (record); >0 = trend-corrected EMA (TESTED worse +0.0056 @alpha=0.6: EMA is variance-reduction avg, not a lagged tracker; extrapolation overshoots the curved basin -> loss flattens HIGHER)
    trend_snap_frac = 0.95       # snapshot EMA at this fraction of train_steps (unused when trend_alpha=0)

    # initialize model parameters
    init_kind = "normal"  # "normal" (record) or "orthogonal" (worse) or "uniform" (TESTED worse +0.003: bounded tails, same var)
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init (std=1.0; 0.2 much worse +0.0047)
            else:
                if init_kind == "orthogonal":
                    nn.init.orthogonal_(w, gain=0.33**0.5)  # uniform singular values=sqrt(0.33), same element-var as record normal init
                elif init_kind == "uniform":
                    # uniform[-a,a] with same variance as record normal: a=sqrt(3)*std
                    a = (3 * 0.33 / w.size(-1))**0.5
                    w.uniform_(-a, a)
                else:
                    w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # record; 0.25 neutral, 0.30 untested(neutral-likely), 0.5 worse
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # S4-4: layer-wise LR (depth-dependent multiplier). Earlier layers receive
    # attenuated gradients through backprop; a mild boost speeds convergence in the
    # few-step regime. Normalized so the mean multiplier = 1.0 (isolates the depth
    # effect from a global lr change). Applied to BOTH Muon (2D block) and, when
    # layerwise_1d is set, the 1D block norm gains (which also see attenuated grads).
    # 0 = off (record, uniform 1.0).
    layerwise_alpha = 0.30   # S4-4 LOCKED: earlier-layer boost, normalized mean=1.0. Peak: 0.20=3.27561, 0.30=3.27551(BEST), 0.40=3.27596. -0.00064 vs ref 3.27615.
    layerwise_shape = "linear"    # S4-8: "linear"=record(S4-4 BEST), "quadratic" TESTED worse +0.00038 (concentrates boost on earliest, starves middle layers).
    layerwise_1d = True       # S6-1: extend layerwise depth boost to 1D block norm gains (204bcb35 config). Compounds w/ 2D Muon boost.
    _lr_mult = {}
    _depth_mult = [1.0] * len(model.blocks)
    if layerwise_alpha != 0:
        nblk = len(model.blocks)
        if layerwise_shape == "quadratic":
            raw = [1.0 + layerwise_alpha * (1.0 - i / (nblk - 1))**2 for i in range(nblk)]
        else:
            raw = [1.0 + layerwise_alpha * (1.0 - i / (nblk - 1)) for i in range(nblk)]
        avg = sum(raw) / nblk
        _depth_mult = [r / avg for r in raw]
        for i, block in enumerate(model.blocks):
            for p in block.parameters():
                if p.ndim >= 2:
                    _lr_mult[p] = _depth_mult[i]

    # create the optimizer(s). Per-group weight decay: embed wd=0 (it's the
    # undertrained bottleneck + biggest param; wd=0.001*lr=0.7 shrinks it while it's
    # still learning, slowing convergence). proj/1D keep wd=0.001 (record).
    embed_wd = 0.001      # record; S5-2: embed wd=0 WORSE +0.0029 (wd regularizes the 38M-param table, needed). KEEP 0.001.
    _adamw_groups = [dict(params=[model.embed.weight], lr=0.7, weight_decay=embed_wd),  # record; S3-8: 0.8 worse+0.0005. KEEP 0.7.
                     dict(params=[model.proj.weight], lr=0.006)]
    # 1D params: block norm gains get depth-scaled lr (layerwise), top norms + biases stay at base 0.015.
    if layerwise_alpha != 0 and layerwise_1d:
        _block_1d = {i: [] for i in range(len(model.blocks))}
        _other_1d = []
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                continue
            if name.startswith("blocks."):
                _block_1d[int(name.split(".")[1])].append(p)
            else:
                _other_1d.append(p)
        for i, ps in _block_1d.items():
            if ps:
                _adamw_groups.append(dict(params=ps, lr=0.015 * _depth_mult[i]))  # S3-9: base 0.015 (0.020 worse); depth-scaled here
        if _other_1d:
            _adamw_groups.append(dict(params=_other_1d, lr=0.015))
    else:
        _adamw_groups.append(dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015))
    optimizer1 = AdamW(_adamw_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)  # record; eps=1e-8 neutral, beta2=0.99 neutral(+0.0003); beta1=0.7 SCREENED worse +0.0021 (S2-4b, sweet spot=0.8)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.025, weight_decay=0.05, mu=0.95, lr_mult=_lr_mult)   # record; update_space TESTED much worse +0.063 (S4-7): NS on noisy single grads is bad
    optimizers = [optimizer1, optimizer2]
    _adamw_family = [optimizer1]   # follow the AdamW schedule (eta_adamw)
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            group["initial_wd"] = group["weight_decay"]
    # set of AdamW param ids (for decoupled EMA decay)
    _adamw_param_ids = set(id(p) for opt in _adamw_family for g in opt.param_groups for p in g["params"])
    _embed_param_id = id(model.embed.weight)  # for selective EMA reset

    # W&B run for this trial (fixed logging infra; no-op if W&B unconfigured)
    wandb_run = None
    if _wandb_on:
        try:
            import wandb
            wandb_run = wandb.init(project="ee-speedrun-optimizer-wandb",
                                   entity=os.environ.get("WANDB_ENTITY"),
                                   name=f"{Path(logfile).stem}-t{trial_idx}", reinit=True,
                                   settings=wandb.Settings(silent=True),
                                   config={"train_steps": train_steps})
        except Exception as e:
            print0(f"wandb disabled: {e}", console=True)
            wandb_run = None

    muon_warmup = 0         # S5-5: warmup=50 MUCH WORSE +0.0035 (few-step regime needs full LR from step 0). Confirmed: no warmup is optimal.

    # learning rate schedule: stable, then decay to a floor, then flat tail.
    # Muon and AdamW share the stable phase but can have different decay/flat-tail
    # boundaries (swa_start vs swa_start_adamw) and floors (decouple_adamw).
    def _decay_shape(progress, start_frac, end_frac):
        t = (progress - start_frac) / (end_frac - start_frac)
        return (1 - math.cos(math.pi * t)) / 2 if decay_kind == "cosine" else t ** decay_power

    def set_hparams(step):
        progress = step / train_steps
        assert 0 <= progress < 1
        # Muon-only linear warmup (first muon_warmup steps)
        _muon_warmup_factor = min(1.0, step / muon_warmup) if muon_warmup > 0 else 1.0
        # ---- Muon schedule (swa_start) ----
        if progress < stable_frac:
            eta_muon = 1.0 * _muon_warmup_factor
        elif progress < swa_start:
            eta_muon = 1.0 + (swa_floor - 1.0) * _decay_shape(progress, stable_frac, swa_start)
        else:
            if swa_tail == "cycle":
                local = (progress - swa_start) / (1 - swa_start)
                eta_muon = cycle_mid + cycle_amp * math.cos(2 * math.pi * cycle_n * local)
            elif swa_tail == "decay":
                local = (progress - swa_start) / (1 - swa_start)
                eta_muon = swa_floor_start + (swa_floor_end - swa_floor_start) * local
            else:
                eta_muon = swa_floor
        # ---- AdamW schedule (swa_start_adamw when decoupled) ----
        if not decouple_adamw:
            eta_adamw = eta_muon
        elif progress < stable_frac_adamw:
            eta_adamw = 1.0
        elif progress < swa_start_adamw:
            eta_adamw = 1.0 + (adamw_floor - 1.0) * _decay_shape(progress, stable_frac_adamw, swa_start_adamw)
        else:
            eta_adamw = adamw_floor
        for opt in optimizers:
            eta = eta_adamw if opt in _adamw_family else eta_muon
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta
                # Zero weight decay during the flat tail (prevent shrinkage bias on EMA)
                if zero_wd_flat_tail and opt is optimizer2:
                    group["weight_decay"] = 0.0 if progress >= swa_start else group.get("initial_wd", 0.05)
                if zero_adamw_wd_flat_tail and opt is optimizer1:
                    group["weight_decay"] = 0.0 if progress >= swa_start_adamw else group.get("initial_wd", 0.001)
                # Flat-tail Muon momentum boost (smoother NS input in the oscillating tail)
                if opt is optimizer2 and flat_tail_mu != 0:
                    group["mu"] = flat_tail_mu if progress >= swa_start else 0.95


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # parameter EMA (SWA) in float32; params are synced across ranks after
    # broadcast + grad allreduce, so the EMA is identical on every rank.
    ema_params = [p.detach().float().clone() for p in model.parameters()]
    swa_n = 0                       # number of uniform-SWA samples accumulated in the flat tail
    swa_start_step = int(swa_start * train_steps)
    swa_start_adamw_step = int(swa_start_adamw * train_steps)
    trend_snap_step = int(trend_snap_frac * train_steps)
    ema_params_snap = None         # EMA snapshot for trend correction (set at trend_snap_step)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            # evaluate on the EMA (SWA) parameters (trend-corrected in the flat tail)
            _orig = [p.data.clone() for p in model.parameters()]
            use_trend = trend_alpha > 0 and ema_params_snap is not None and step >= trend_snap_step
            for i, (p, e) in enumerate(zip(model.parameters(), ema_params)):
                if eval_raw_embed and id(p) == _embed_param_id:
                    continue  # keep the raw (latest, most-trained) embed; EMA lags the descending embed
                if use_trend:
                    w = e + trend_alpha * (e - ema_params_snap[i])
                    p.data.copy_(w.to(p.dtype))
                else:
                    p.data.copy_(e.to(p.dtype))
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            for p, o in zip(model.parameters(), _orig):
                p.data.copy_(o)
            del _orig
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            if wandb_run is not None:
                wandb_run.log({"val_loss": float(val_loss), "step": step})
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        # update parameter EMA (SWA averaging)
        if ema_reset_at_swa and step == swa_start_step:
            # reset EMA to current raw weights at swa_start (removes descent-phase lag)
            for p, e in zip(model.parameters(), ema_params):
                e.copy_(p.detach().to(e.dtype))
        if ema_reset_embed_at_swa and step == swa_start_adamw_step:
            # S4-9: reset ONLY embed EMA at swa_start (embed descends -> EMA lags;
            # Muon params oscillate -> full-history EMA finds wider min, don't reset)
            for p, e in zip(model.parameters(), ema_params):
                if id(p) == _embed_param_id:
                    e.copy_(p.detach().to(e.dtype))
        if swa_mode == "uniform" and step >= swa_start_step:
            # canonical SWA: uniform running average over the flat-tail weights
            swa_n += 1
            inv_n = 1.0 / swa_n
            for p, e in zip(model.parameters(), ema_params):
                e.lerp_(p.detach().to(e.dtype), inv_n)   # e += (p - e)/n
        elif decouple_ema:
            for p, e in zip(model.parameters(), ema_params):
                d = ema_decay_adamw if id(p) in _adamw_param_ids else ema_decay_muon
                e.lerp_(p.detach().to(e.dtype), 1 - d)
        else:
            d = ema_decay_descent if step < swa_start_step else ema_decay
            for p, e in zip(model.parameters(), ema_params):
                e.lerp_(p.detach().to(e.dtype), 1 - d)
        # snapshot the EMA for trend correction at the configured step
        if trend_alpha > 0 and step == trend_snap_step:
            ema_params_snap = [e.clone() for e in ema_params]
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

    if wandb_run is not None:
        wandb_run.summary["final_val_loss"] = float(val_loss)
        wandb_run.finish()

dist.destroy_process_group()

