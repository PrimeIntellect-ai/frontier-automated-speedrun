"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import math
import os
import sys
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

def _e(name, default):
    """Read a knob from the environment (screening convenience); default == current recipe."""
    v = os.environ.get(name)
    return type(default)(v) if v is not None else default

NS_ITERS   = _e("NS_ITERS", 12)
NS_FP32    = _e("NS_FP32", 0)
CAUTIOUS   = _e("CAUTIOUS", 0)
ADAMUON    = _e("ADAMUON", 0)      # element-wise 2nd moment on the orthogonalized update
ADAMUON_B2 = _e("ADAMUON_B2", 0.95)
NESTEROV   = _e("NESTEROV", 1)
# Weight of the fresh gradient in the Nesterov mix, decoupled from the EMA horizon:
# update = lerp(grad, momentum, MU_NEST). -1 means "use the group's mu" (= baseline).
MU_NEST    = _e("MU_NEST", -1.0)
# Composition order. Muon averages the gradients and then orthogonalizes. ORTH_FIRST
# orthogonalizes each fresh gradient and then averages those orthogonal factors, so the
# resulting spectrum reflects how *consistently* each direction recurs rather than how
# large it is; the result is renormalized to Muon's Frobenius norm so lr transfers.
ORTH_FIRST = _e("ORTH_FIRST", 0)
# Gradient centering: remove each output unit's mean gradient over its inputs before
# orthogonalizing, i.e. strip the rank-1 "uniform" component from the update.
GRAD_CENTER = _e("GRAD_CENTER", 0)
# Trust-ratio Muon (LARS-style). Muon's step has a fixed spectral size, so the *relative*
# step ||update||/||W|| drifts as the weights grow (~3.6x over a run) — and every knob that
# has paid off here (front-loaded wd, the lr tilts) manipulates exactly that ratio
# indirectly. TRUST=1 scales each matrix's update by ||W||_F / ||W_init||_F so lr sets the
# relative step directly. TRUST_MIN floors the ratio for the zero-initialized projections.
# TRUST=1 (full LARS) diverges: the update grows with ||W||, and the front-loaded wd that
# would brake it decays to zero late. But that fixes the sign — front-loaded wd helps by
# keeping `rel` HIGH EARLY, so the wanted direction is rel decaying FASTER with ||W||, i.e.
# a NEGATIVE exponent. TRUST_POW<0 scales the update by (||W||/||W_init||)^TRUST_POW.
# Weight-space heavy ball layered on Muon's gradient momentum (no extra gradients):
#   p_{t+1} = p_t - lr*U_t + WMOM*(p_t - p_{t-1})
# A second, parameter-space momentum stage, distinct from the EMA over gradients.
WMOM       = _e("WMOM", 0.0)
TRUST      = _e("TRUST", 0)
TRUST_POW  = _e("TRUST_POW", 0.0)
TRUST_MAX  = _e("TRUST_MAX", 2.0)
TRUST_MIN  = _e("TRUST_MIN", 0.3)
STD_FAC_G  = _e("STD_FAC", 0.33)
# update scale rule: 0 = baseline max(1, fan_out/fan_in)^0.5, 1 = spectral (fan_out/fan_in)^0.5
SCALE_RULE = _e("SCALE_RULE", 0)
# AdEMAMix-style second, much slower gradient EMA mixed into the Muon direction.
# alpha=0 disables it. The slow EMA starts at zero, so its weight ramps in naturally.
AEM_ALPHA  = _e("AEM_ALPHA", 0.0)
AEM_B3     = _e("AEM_B3", 0.999)

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.float() if NS_FP32 else G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(NS_ITERS):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu, v2=None, split=0, slow=None):
    # `mu` is a 0-dim tensor so a momentum schedule does not retrigger compilation.
    if GRAD_CENTER:
        grad = grad - grad.mean(dim=-1, keepdim=True)
    momentum.lerp_(grad, 1 - mu)
    mix = mu if MU_NEST < 0 else MU_NEST
    update = torch.lerp(grad, momentum, mix) if NESTEROV else momentum.clone()
    if slow is not None:
        slow.lerp_(grad, 1 - AEM_B3)
        update = update + AEM_ALPHA * slow
    if ORTH_FIRST:
        o = zeropower_via_newtonschulz5(grad).to(momentum.dtype)
        momentum.lerp_(o, 1 - mu)
        update = torch.lerp(o, momentum, mix) if NESTEROV else momentum.clone()
        k = min(grad.size(-2), grad.size(-1))**0.5
        update = update * (k / update.norm().clamp(min=1e-12))
        ratio0 = grad.size(-2) / grad.size(-1)
        return update * (ratio0 if SCALE_RULE else max(1, ratio0))**0.5
    if split:
        # Per-head orthogonalization: attention heads span independent subspaces, so
        # treat the matrix as `|split|` separate blocks. split>0 splits the output
        # dim (q/k/v), split<0 the input dim (attn.proj). Total ||U||_F is unchanged,
        # so the lr scale computed from the original shape still applies.
        m, n = update.size(-2), update.size(-1)
        h = abs(split)
        if split > 0:
            update = zeropower_via_newtonschulz5(update.view(h, m // h, n)).view(m, n)
        else:
            update = zeropower_via_newtonschulz5(
                update.mT.reshape(h, n // h, m)).view(n, m).mT.contiguous()
    else:
        update = zeropower_via_newtonschulz5(update)
    if CAUTIOUS:
        mask = (update * grad > 0).to(update.dtype)
        update = update * mask * (mask.numel() / mask.sum().clamp(min=1.0))
    if v2 is not None:
        # AdaMuon: element-wise second moment on the orthogonal factor, then rescale
        # back to the orthogonal factor's RMS so the Muon lr scale is preserved.
        rms0 = update.square().mean().sqrt()
        v2.mul_(ADAMUON_B2).addcmul_(update, update, value=1 - ADAMUON_B2)
        update = update / (v2.sqrt() + 1e-12)
        update = update * (rms0 / update.square().mean().sqrt().clamp(min=1e-12))
    ratio = grad.size(-2) / grad.size(-1)
    update *= (ratio if SCALE_RULE else max(1, ratio))**0.5
    return update

########################################
# Accumulated-Gram Muon ("SOAP-lite").
# Muon's update is the polar factor M (M^T M)^{-1/2}, i.e. it whitens with the
# *instantaneous* Gram of the momentum. PRECOND=1 replaces that Gram with an EMA
# over steps (PRE_BETA), which is the matrix analogue of raising Adam's beta2 --
# and beta2 0.95 -> 0.99 was the one thing that helped here. PRE_BETA=0 reproduces
# plain Muon exactly (via eigh instead of Newton-Schulz), which doubles as a check
# that 12 NS iterations really are converged.
PRECOND  = _e("PRECOND", 0)
PRE_BETA = _e("PRE_BETA", 0.0)
PRE_EPS  = _e("PRE_EPS", 1e-6)   # eigenvalue floor, as a fraction of lambda_max
PRE_FREQ = _e("PRE_FREQ", 1)     # how often to redo the eigendecomposition
# Spectral power. Muon flattens EVERY singular value of the momentum to 1, which
# amplifies the smallest (noisiest) directions to full strength. Applying A^(-PRE_POW/2)
# instead maps sigma -> sigma^(1-PRE_POW), interpolating normalized-SGD (0) <-> Muon (1).
# The update is renormalized to Muon's Frobenius norm either way, so lr transfers.
PRE_POW  = _e("PRE_POW", 1.0)

@torch.compile
def momentum_dir(grad, momentum, mu, slow=None):
    if GRAD_CENTER:
        grad = grad - grad.mean(dim=-1, keepdim=True)
    momentum.lerp_(grad, 1 - mu)
    mix = mu if MU_NEST < 0 else MU_NEST
    update = torch.lerp(grad, momentum, mix) if NESTEROV else momentum.clone()
    if slow is not None:
        slow.lerp_(grad, 1 - AEM_B3)
        update = update + AEM_ALPHA * slow
    return update

def inv_sqrt_psd(A):
    """Symmetric inverse square root, trace-normalized and damped.

    The overall scale is irrelevant (the caller renormalizes the update), so we
    normalize by mean(diag) first; that also makes the damping relative. Early in
    training several Gram matrices are exactly zero (the zero-init projections make
    q/k/v/fc gradients vanish at step 0), hence the clamp on the trace.
    """
    n = A.size(-1)
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    d = A.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
    for jitter in (PRE_EPS, 1e-4, 1e-2):
        try:
            lam, Q = torch.linalg.eigh(A / d + jitter * I)
            w = lam.clamp_min(jitter).pow(-0.5 * PRE_POW)
            return (Q * w.unsqueeze(-2)) @ Q.mT
        except Exception:
            continue
    return I

@torch.compile
def apply_precond(M, B, rows_small, scale):
    U = (B @ M) if rows_small else (M @ B)
    # renormalize to the Frobenius norm of a Muon update (all singular values 1)
    k = min(M.size(-2), M.size(-1))**0.5
    return U * (scale * k / U.norm().clamp(min=1e-12))

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, split=0):
        assert isinstance(params, list) and len(params) >= 1
        by_size = lambda ps: sorted(ps, key=lambda x: x.size(), reverse=True)
        if isinstance(params[0], dict):     # explicit param groups (e.g. per-layer lr)
            arg = [dict(g, params=by_size(g["params"])) for g in params]
            dev = arg[0]["params"][0].device
        else:
            assert isinstance(params[0], torch.nn.Parameter)
            arg = by_size(params)
            dev = arg[0].device
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, split=split)
        super().__init__(arg, defaults)
        for group in self.param_groups:
            group["mu_t"] = torch.tensor(float(group["mu"]), device=dev)

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
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        if ADAMUON:
                            state["v2"] = torch.zeros_like(p)
                        if AEM_ALPHA > 0:
                            state["slow"] = torch.zeros_like(p)
                        if PRECOND:
                            k = min(p.size(-2), p.size(-1))
                            state["gram"] = torch.zeros(k, k, device=p.device, dtype=torch.float32)
                            state["k"] = 0
                    if PRECOND:
                        M = momentum_dir(p.grad, state["momentum"], group["mu_t"],
                                         slow=state.get("slow"))
                        Mf = M.float()
                        rows_small = p.size(-2) <= p.size(-1)
                        G = (Mf @ Mf.mT) if rows_small else (Mf.mT @ Mf)
                        state["gram"].mul_(PRE_BETA).add_(G, alpha=1 - PRE_BETA)
                        if state["k"] % PRE_FREQ == 0:
                            state["B"] = inv_sqrt_psd(state["gram"])
                        state["k"] += 1
                        ratio = p.size(-2) / p.size(-1)
                        update = apply_precond(M, state["B"], rows_small,
                                               max(1, ratio)**0.5)
                    else:
                        update = muon_update(p.grad, state["momentum"], group["mu_t"],
                                             v2=state.get("v2"), split=group["split"],
                                             slow=state.get("slow"))
                    if TRUST or TRUST_POW:
                        ref = (STD_FAC_G * p.size(-2))**0.5   # ||W||_F of the standard init
                        r = (p.norm() / ref).clamp(min=1e-4)
                        r = r if TRUST else r.pow(TRUST_POW)
                        update = update * r.clamp(TRUST_MIN, TRUST_MAX)
                    if WMOM > 0:
                        cur = p.detach().clone()
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    if WMOM > 0:
                        if "wprev" in state:
                            p.add_(cur - state["wprev"], alpha=WMOM)
                        state["wprev"] = cur
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


LAZY_EMBED = _e("LAZY_EMBED", 0)

@torch.compile
def embed_adam_step(master, p, grad, m, v, lr_t, wd_t, bc1_t, bc2_t, b1, b2, eps):
    g = grad.float()
    m.lerp_(g, 1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    upd = (m / bc1_t) / ((v / bc2_t).sqrt() + eps)
    if LAZY_EMBED:
        # Lazy/sparse Adam: ~20-40% of vocabulary rows get no gradient in a given batch,
        # yet dense Adam still moves them on stale momentum and decays them. Restrict the
        # step to rows that actually occurred.
        live = (grad.abs().sum(-1, keepdim=True) > 0).to(master.dtype)
        master.mul_(1 - lr_t * wd_t * live)
        master.sub_(lr_t * upd * live)
    else:
        master.mul_(1 - lr_t * wd_t)
        master.sub_(lr_t * upd)
    p.copy_(master)

class EmbedAdamW(torch.optim.Optimizer):
    """AdamW for the bf16 embedding, with fp32 master weights and fp32 moments.

    The architecture makes nn.Embedding bf16, so the stock fused AdamW keeps its moments
    and applies its updates in bf16. Once the rows grow to RMS ~15 the bf16 ulp there is
    0.0625 while a typical update is ~0.2, so ~15% of every update is quantization error;
    worse, as the schedule decays the updates drop below the ulp and the table silently
    stops moving. A fp32 master copy (written back to the bf16 parameter each step)
    removes both effects and changes nothing else about the algorithm.
    """
    def __init__(self, params, lr, betas, eps, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        for group in self.param_groups:
            dev = group["params"][0].device
            group["_t"] = [torch.zeros((), device=dev) for _ in range(4)]
            group["step"] = 0

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            group["step"] += 1
            k = group["step"]
            lr_t, wd_t, bc1_t, bc2_t = group["_t"]
            lr_t.fill_(group["lr"]); wd_t.fill_(group["weight_decay"])
            bc1_t.fill_(1 - b1 ** k); bc2_t.fill_(1 - b2 ** k)
            for p in group["params"]:
                st = self.state[p]
                if len(st) == 0:
                    st["master"] = p.detach().float().clone()
                    st["m"] = torch.zeros_like(st["master"])
                    st["v"] = torch.zeros_like(st["master"])
                embed_adam_step(st["master"], p.detach(), p.grad, st["m"], st["v"],
                                lr_t, wd_t, bc1_t, bc2_t, b1, b2, group["eps"])


@torch.compile
def lion_update(p, grad, m, lr_t, wd_t, b1, b2):
    # Lion: sign of an interpolation between the fresh gradient and the state, with the
    # state kept on a *different* beta. Decoupled wd, as in the original.
    u = torch.lerp(grad, m, b1).sign()
    p.mul_(1 - lr_t * wd_t)
    p.sub_(u * lr_t)
    m.lerp_(grad, 1 - b2)

class Lion(torch.optim.Optimizer):
    """Lion, for whichever group we point it at (self-contained, no external import)."""
    def __init__(self, params, lr, betas, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        for group in self.param_groups:
            dev = group["params"][0].device
            group["_t"] = [torch.zeros((), device=dev) for _ in range(2)]

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr_t, wd_t = group["_t"]
            lr_t.fill_(group["lr"]); wd_t.fill_(group["weight_decay"])
            for p in group["params"]:
                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p)
                lion_update(p.detach(), p.grad, st["m"], lr_t, wd_t, b1, b2)


class Lookahead:
    """Lookahead slow weights, as an optimizer wrapper (no extra forward/backward).

    Placed last in `optimizers`, so it runs after the real updates. Every `k` steps it
    pulls the parameters partway back toward a slow exponential average of themselves:
    slow += alpha*(fast - slow); fast <- slow. Distinct from TailAverage, which only
    materializes an average at the very end and never feeds it back into training.
    """
    def __init__(self, params, k, alpha):
        self.params = list(params)
        self.k, self.alpha = k, alpha
        self.param_groups = [dict(params=[], lr=0.0)]
        self.t = 0
        self.slow = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        if self.slow is None:
            self.slow = [p.detach().float().clone() for p in self.params]
        if self.t % self.k:
            return
        for sl, p in zip(self.slow, self.params):
            sl.lerp_(p.detach().float(), self.alpha)
            p.detach().copy_(sl)


class TailAverage:
    """Polyak/EMA tail averaging of the iterate — an optimizer wrapper, not extra work.

    Placed last in `optimizers` so it runs after the real updates. Over the final
    `window` steps it keeps an EMA of the parameters, and on the last step it writes
    that average back into the parameters. Adds no forward/backward passes.
    """
    def __init__(self, params, total_steps, window, decay):
        self.params = [p for p in params]
        self.total, self.decay = total_steps, decay
        self.start = total_steps - window
        self.param_groups = [dict(params=[], lr=0.0)]
        self.t = 0
        self.buf = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        if self.t <= self.start:
            return
        if self.buf is None:
            self.buf = [p.detach().float().clone() for p in self.params]
        else:
            for b, p in zip(self.buf, self.params):
                b.lerp_(p.detach().float(), 1 - self.decay)
        if self.t >= self.total:
            for b, p in zip(self.buf, self.params):
                p.detach().copy_(b)


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
    train_steps = _e("STEPS", 2920)

    LR_MUON   = _e("LR_MUON", 0.025)
    WD_MUON   = _e("WD_MUON", 0.05)
    MU        = _e("MU", 0.95)
    LR_EMBED  = _e("LR_EMBED", 0.7)
    LR_HEAD   = _e("LR_HEAD", 0.004)
    LR_SCALAR = _e("LR_SCALAR", 0.015)
    ADAM_B1   = _e("ADAM_B1", 0.8)
    ADAM_B2   = _e("ADAM_B2", 0.99)
    ADAM_EPS  = _e("ADAM_EPS", 1e-10)
    ADAM_WD   = _e("ADAM_WD", 0.001)
    COOLDOWN  = _e("COOLDOWN", 0.7)
    WARMUP    = _e("WARMUP", 0.0)     # fraction of run spent ramping eta 0->1
    # A small lr FLOOR is worth ~0.0014. Decaying linearly to exactly zero wastes the
    # tail of the cooldown: the last few percent of steps run at a near-zero lr and do
    # almost nothing. A ~1.5% floor gives a soft landing instead. Flat over 0.01-0.035;
    # 0 and 0.06 are both clearly worse, and DECAY_POW/COOLDOWN stay optimal alongside it.
    FINAL     = _e("FINAL", 0.015)    # eta floor at the very end
    # FLOOR_MODE 0: the floor is folded into eta and then scaled by each group's tilt, so
    # groups end at (1 - tilt)*FINAL — 0.3x for block 0, 1.4x for the lm head. Mode 1
    # applies the floor uniformly *after* the tilt so every group lands on the same floor.
    FLOOR_MODE = _e("FLOOR_MODE", 0)
    # The lr decay is driven by step/train_steps, but the schedule horizon need not equal
    # the run length. SCHED_MULT<1 completes the decay early and then HOLDS at the floor for
    # the rest of the run; SCHED_MULT>1 ends the run before the anneal finishes. Neither
    # shape is reachable via COOLDOWN or FINAL. Applies to the lr decay only — the tilts,
    # the wd profile and the momentum warmup keep using true progress.
    SCHED_MULT = _e("SCHED_MULT", 1.0)
    # Decay shape over the cooldown: 0 = power (dec**DECAY_POW, linear at POW=1),
    # 1 = cosine, which flattens both ends of the cooldown rather than just the tail.
    DECAY_SHAPE = _e("DECAY_SHAPE", 0)
    # Mode 0 gives each group a terminal lr of (1-tilt)*FINAL, so the head already lands on
    # 1.4x the floor and the uniform variant (mode 1) was worse => the head wants MORE floor
    # than the rest. FINAL_HEAD gives it its own pre-tilt floor.
    FINAL_HEAD = _e("FINAL_HEAD", 0.0)   # >0 overrides FINAL for the lm head group
    # Per-group decay curvature. The head's schedule shape is currently only adjustable via
    # its linear tilt; its curvature is locked to the global DECAY_POW.
    DPOW_HEAD  = _e("DPOW_HEAD", 0.0)    # >0 overrides DECAY_POW for the lm head group
    # Mode 1 both LOWERED the head's terminal lr (1.4*FINAL -> FINAL) and RAISED the shallow
    # blocks' (0.3*FINAL -> FINAL), and was worse; this hybrid keeps mode 0 but adds an
    # absolute minimum, so only the low-floor (high-tilt, shallow) groups are lifted.
    FLOOR_MIN  = _e("FLOOR_MIN", 0.0)
    # q/k gauge fixing. Scaling (q.weight, q.bias) by any c is EXACTLY function-preserving,
    # because each head's q is RMS-normalized before the dot product. Their norms drift up
    # ~3.6x over a run, which silently decays their effective lr by that much on top of the
    # schedule. Pinning the norm removes that drift and nothing else.
    QK_GAUGE   = _e("QK_GAUGE", 0.0)   # >0: pin ||q||_F, ||k||_F to this multiple of init
    # Curve diagnostic: this recipe's advantage over stock peaks at ~79% of the run and
    # then erodes by 30%. In that window the positive tilts have driven the body's lr
    # multiplier down toward 0.3x. TILT_MIN floors the tilt multiplier so the body keeps
    # learning late, without touching the early phase (or the head, whose multiplier >1).
    TILT_MIN   = _e("TILT_MIN", 0.0)
    DECAY_POW = _e("DECAY_POW", 1.0)  # decay shape: eta ~ t**DECAY_POW, t = (1-progress)/cooldown
    STD_FAC   = _e("STD_FAC", 0.33)
    # Depth profile for the hidden init std: every block currently gets the same std
    # regardless of depth. std_l = std * (1 + INIT_DEPTH*(l/(nlayers-1) - 0.5)).
    INIT_DEPTH = _e("INIT_DEPTH", 0.0)
    # Structural init priors (I have only ever varied init *scales*). Both leave the initial
    # function untouched because attn.proj is zero-init; they only shape early gradients.
    #   QK_TIE_INIT: k.weight <- q.weight, so attention starts as a token-similarity match
    #                (an induction-head-like prior) instead of a random projection.
    #   V_EYE_INIT:  v.weight <- identity, so attention starts as a copy of the residual.
    QK_TIE_INIT = _e("QK_TIE_INIT", 0)
    V_EYE_INIT  = _e("V_EYE_INIT", 0)
    INIT_ORTHO = _e("INIT_ORTHO", 0)
    # Muon momentum warmup: early gradients rotate fast, so a 20-step EMA (mu=0.95) is
    # stale at the start. Ramping mu 0.80 -> 0.95 over the first 10% of the run is worth
    # ~0.0035 val loss. Both a lower MU0 (0.75) and a longer ramp (0.20) are worse.
    MU0       = _e("MU0", 0.80)       # momentum at step 0 (ramps linearly to MU)
    MU_WARM   = _e("MU_WARM", 0.10)   # fraction of the run over which mu ramps MU0 -> MU
    MU_LATE   = _e("MU_LATE", MU)     # mu at the very end (ramps MU -> MU_LATE after warmup)
    # Shape of the momentum warmup, which is the single most valuable component (dropping
    # it costs 0.0059). 0 = linear in mu (as tuned); 1 = linear in the averaging horizon
    # h = 1/(1-mu), i.e. 5 -> 20 steps, which raises mu faster at the start.
    MU_SHAPE  = _e("MU_SHAPE", 0)
    # Per-group schedule tilt: multiplies a group's eta by (1 + TILT*(1-2*progress)),
    # i.e. TILT>0 makes that group relatively hotter early and colder late.
    # The groups cool at very different rates (relative motion between step 300 and 2400
    # falls ~7x for the Muon matrices but ~15x for the lm head), and rebalancing that is
    # worth ~0.002: body hotter early, readout hotter late. EMBED/SCALAR are best left 0.
    TILT_EMBED  = _e("TILT_EMBED", 0.0)
    TILT_HEAD   = _e("TILT_HEAD", -0.40)
    TILT_SCALAR = _e("TILT_SCALAR", 0.0)   # RMSNorm gains
    # Block biases get their own tilt: with the rest of the recipe in place they are the
    # last group left out of balance -- by step 2400 their relative motion is 5-10x that
    # of the weight matrices (attn.v.bias worst at ~0.010 vs ~0.0013).
    TILT_BIAS   = _e("TILT_BIAS", 0.45)
    # Late-window diagnostic: by the end, the biases' relative motion is 2-4x that of the
    # Muon matrices (they decay only ~4x over the last third while the matrices decay 8-16x).
    # For the *redundant* biases (attn.v, both output projections) that late motion should be
    # pure noise, so give just those groups their own tilt to cool them late.
    TILT_RBIAS  = _e("TILT_RBIAS", 0.0)
    # Depth profile for the RMSNorm gains (they are per-block, and their targets run from
    # ~0.4 at block 0 to ~2.0 at block 11): one Adam group per block when nonzero.
    # The two block gains have opposite trajectories (norm1 rises with depth to ~1.7 at
    # block 11, norm2 falls to ~0.84 at block 0), so try separate lrs for the attention-
    # branch and MLP-branch input scales.
    LR_GAIN1  = _e("LR_GAIN1", 0.0)   # >0 overrides LR_GAIN for blocks.*.norm1.gains
    LR_GAIN2  = _e("LR_GAIN2", 0.0)   # >0 overrides LR_GAIN for blocks.*.norm2.gains
    GAIN_DEPTH_TILT = _e("GAIN_DEPTH_TILT", 0.0)
    GAIN_DEPTH_LR   = _e("GAIN_DEPTH_LR", 0.0)
    # The two top-level gains are special: model.norm1.gains sets the residual-stream
    # scale (diagnostics: it wants ~2.2 but is the slowest-moving tensor in the model,
    # rel ~0.0019) and model.norm2.gains scales the head's input. Give them their own lr.
    # model.norm1.gains sets the residual-stream scale for the whole network and is the
    # slowest-moving tensor in the model at the stock lr (rel ~0.0019 vs ~0.015 for
    # everything else). Giving just that one tensor a ~17x lr is worth ~0.0027. The
    # pre-head gain does NOT want it (0.015 and 0.006 tie; 0.06+ is worse).
    LR_TOPGAIN  = _e("LR_TOPGAIN", 0.25)   # lr for model.norm1.gains (post-embed)
    LR_HEADNORM = _e("LR_HEADNORM", 0.015) # lr for model.norm2.gains (pre-head)
    # With the boosted lr, model.norm1.gains climbs to ~5 by step 1200 and settles ~3.8,
    # so it spends a third of the run travelling. Start it partway there.
    TOPGAIN_INIT = _e("TOPGAIN_INIT", 1.0)
    # mlp.fc uses squared ReLU, so fc.bias sets the activation sparsity. It starts at 0
    # (~50% of units active) and the model learns it to wrms ~0.3 against a pre-activation
    # RMS of ~0.34, so it clearly wants a shift; try starting it negative (sparser).
    FC_BIAS_INIT = _e("FC_BIAS_INIT", 0.0)
    # The block output projections are zero-initialized, so their residual branches start
    # silent and their gradients unlock only via the layers above. With the much hotter
    # early schedule this recipe now uses, a small non-zero start may suit better.
    PROJ_STD  = _e("PROJ_STD", 0.0)   # >0: init blocks' attn.proj/mlp.proj at this
                                      #     fraction of the standard normal init
    # AdamW decay is p *= (1 - lr*wd), so boosting this tensor's lr 17x also boosted its
    # decay 17x — which fights the growth to ~4 that it is trying to achieve.
    WD_TOPGAIN = _e("WD_TOPGAIN", -1.0)   # <0 means "use WD_GAIN"
    # The post-embed gain has to travel 1 -> ~5 in the first ~1200 steps and then merely
    # track; a positive tilt gives it the lr where it needs it and cuts the end-of-run
    # jitter that its large lr introduced (across-seed sd doubled when it was added).
    TILT_TOPGAIN = _e("TILT_TOPGAIN", 0.0)
    # attn.v.bias is the hottest tensor in the model (rel ~0.042 vs ~0.025 for the
    # matrices) and is nearly redundant with attn.proj.bias -- proj(v_bias) and proj.bias
    # add the same kind of constant to the residual stream. Give it its own lr.
    LR_VBIAS   = _e("LR_VBIAS", 0.0015)   # >0 overrides LR_SCALAR for the attn v biases
    # attn.proj.bias and mlp.proj.bias both just add a constant vector to the residual
    # stream, so they largely duplicate each other across all 12 blocks -- same story.
    # In the attention logits (q+b_q).(k+b_k), the q.b_k and b_q.b_k terms are constant
    # across keys and cancel in the softmax, so b_k only acts through the QK-RMSNorm
    # nonlinearity — nearly functionless, i.e. the same situation as attn.v.bias.
    LR_KBIAS   = _e("LR_KBIAS", 0.0)      # >0 overrides LR_SCALAR for the attn k biases
    LR_PBIAS   = _e("LR_PBIAS", 0.004)    # >0 overrides LR_SCALAR for the output-proj biases
    # Shape of every tilt: multiplier = 1 + TILT*(2/(k+1) - 2*progress**k). k=1 is the
    # linear ramp crossing over at progress 0.5; k<1 moves the crossover earlier. The
    # 2/(k+1) offset keeps the mean multiplier at 1 so this is a pure shape knob.
    TILT_POW    = _e("TILT_POW", 1.0)
    # TILT_PHASE=1: hold each group's tilt at its maximum through the plateau and ramp it
    # only across the cooldown, instead of ramping linearly over the whole run. Mean-centred
    # so the average multiplier (and hence each group's average lr) is unchanged.
    TILT_PHASE  = _e("TILT_PHASE", 0)
    # Per-group cooldown fraction. The linear tilt can only reweight a group's lr across
    # the run; it cannot move where that group's decay *starts*. These can.
    COOL_HEAD  = _e("COOL_HEAD", COOLDOWN)
    COOL_MUON  = _e("COOL_MUON", COOLDOWN)
    COOL_EMBED = _e("COOL_EMBED", COOLDOWN)
    TILT_HBIAS  = _e("TILT_HBIAS", 0.0)
    # Depth profile for the Muon matrices: features form bottom-up, so per-layer lr and
    # schedule shape may want to vary with depth. f = l/(nlayers-1) - 0.5.
    DEPTH_LR   = _e("DEPTH_LR", 0.0)
    # Features form bottom-up: shallow blocks want to be relatively hotter early and deep
    # blocks hotter late (worth another ~0.0014). A *static* depth lr slope does not help
    # (both signs are worse) -- only this temporal one.
    DEPTH_TILT = _e("DEPTH_TILT", -0.80)
    # Extra tilt applied to the MLP matrices only, on top of TILT_MUON: attention (mixing)
    # and MLP (features) may want to form at different points in the run.
    TILT_MLP   = _e("TILT_MLP", 0.0)
    # q and k are RMS-normalized per head before the dot product, so their weight *norms*
    # are pure gauge: weight decay on them carries no regularization cost and acts only as
    # an effective-lr knob (smaller norm -> larger relative update). Give them their own.
    WD_QK      = _e("WD_QK", 0.0)      # >0: separate wd for q,k
    TILT_QK    = _e("TILT_QK", 0.0)    # extra lr tilt for q,k
    # The lm head wanted to be relatively hotter LATE (readout fits after features form).
    # attn.proj and mlp.proj are the per-block readouts, so try the same for them.
    TILT_OUT   = _e("TILT_OUT", 0.0)
    # Static lr multiplier for the two *output* projections (attn.proj, mlp.proj). They are
    # zero-initialized and grow from nothing, unlike q/k/v/fc which start at the default
    # init, so the shape-based `max(1, fan_out/fan_in)^0.5` rule may not be right for them.
    OUT_MULT   = _e("OUT_MULT", 1.0)
    # mlp.fc is the only matrix whose shape gives it a 2x update scale. MLP_MULT moved fc
    # and mlp.proj together and OUT_MULT moves the two output projections; neither isolates
    # fc, so its shape-derived factor has never actually been tuned.
    FC_MULT    = _e("FC_MULT", 1.0)
    # v is the only attention matrix whose output magnitude matters directly: q and k are
    # RMS-normalized per head before the dot product, and attn.proj is the block's readout.
    # It has always shared q/k's lr.
    V_MULT     = _e("V_MULT", 1.0)
    # Depth profile for the momentum warmup start: a deep block's inputs are still moving
    # (every block below it is changing), so its gradient may rotate faster and want an
    # even shorter memory early.  MU0_l = MU0 + DEPTH_MU * f.
    DEPTH_MU   = _e("DEPTH_MU", 0.0)
    # cos(fresh grad, momentum) diagnostic: deep blocks oscillate 2-3x more than shallow
    # ones throughout training (mlp.fc -0.42 at block 11 vs -0.15 at block 0), i.e. their
    # momentum is comparatively stale. DEPTH_MU only shifted the warmup START; this shifts
    # the plateau value that every block holds for 90% of the run.
    DEPTH_MUP  = _e("DEPTH_MUP", 0.0)   # mu_l = MU + DEPTH_MUP*(l/(nlayers-1) - 0.5)
    # Quadratic correction to the depth tilt (the linear profile may over/under-shoot at
    # the ends): adds DEPTH_TILT2 * (f*f - 0.087), mean-zero over the 12 blocks.
    DEPTH_TILT2 = _e("DEPTH_TILT2", 0.0)
    TILT_MUON   = _e("TILT_MUON", 0.30)
    # Weight decay is applied as p *= (1 - lr*wd), so it currently vanishes along with
    # the lr schedule. COUPLE=1 keeps that; COUPLE=0 divides wd by eta so the *absolute*
    # shrinkage per step stays constant, which keeps update/weight ratios up late.
    WD_COUPLE  = _e("WD_COUPLE", 1.0)   # Muon groups
    AWD_COUPLE = _e("AWD_COUPLE", 1.0)  # AdamW groups
    # Temporal profile for wd itself (independent of the lr tilt, since lr sets the update
    # size while lr*wd sets the shrinkage): wd *= (1 + WD_TILT*(1-2*progress)).
    # Analogue of the FINAL win for wd: with WD_TILT+WD_DEPTH the wd multiplier for the
    # deep blocks reaches exactly 0 at ~72% progress and stays there. WD_FLOOR floors it.
    WD_FLOOR   = _e("WD_FLOOR", 0.0)
    WD_TILT    = _e("WD_TILT", 1.5)   # Muon groups: front-loading wd is worth ~0.0025
    # Deeper blocks want MORE wd front-loading than shallow ones (worth ~0.0010).
    # Static depth slope on the wd BASE (WD_DEPTH only slopes the front-loading tilt, so
    # every block still shares the same average wd). wd_l = WD_MUON*(1 + WDB_DEPTH*f).
    WDB_DEPTH  = _e("WDB_DEPTH", 0.0)
    WD_DEPTH   = _e("WD_DEPTH", 1.5)  # depth slope of WD_TILT: wdtilt_l = WD_TILT + WD_DEPTH*f
    AWD_TILT   = _e("AWD_TILT", 0.0)  # same for the AdamW groups
    # Same mechanism for Adam's first moment: memory should be shorter while the
    # gradient direction is still rotating fast.
    B1_0      = _e("B1_0", ADAM_B1)
    B1_WARM   = _e("B1_WARM", 0.0)
    # Profile for Adam's *second* moment: late in the run the gradient scale is stable,
    # so a longer second-moment horizon may denoise. Ramps every Adam group's beta2 by
    # (B2_LATE - ADAM_B2) * progress.
    B2_LATE   = _e("B2_LATE", ADAM_B2)
    EMA_WIN   = _e("EMA_WIN", 0)      # tail-average window in steps (0 = off)
    EMA_DECAY = _e("EMA_DECAY", 0.98)
    HEAD_MUON = _e("HEAD_MUON", 0.0)  # >0: lm head under Muon at this lr instead of AdamW
    MLP_MULT  = _e("MLP_MULT", 1.0)   # Muon lr multiplier for the MLP matrices vs attn
    SPLIT_QKV   = _e("SPLIT_QKV", 0)   # orthogonalize q/k/v per attention head
    SPLIT_APROJ = _e("SPLIT_APROJ", 0) # orthogonalize attn.proj per attention head
    DIAG      = _e("DIAG", 0)         # >0: log per-tensor scale diagnostics every DIAG steps
    LR_GAIN   = _e("LR_GAIN", LR_SCALAR)    # separate lr for the RMSNorm gains
    LR_HBIAS  = _e("LR_HBIAS", LR_SCALAR)   # separate lr for the lm-head bias
    # Depth-profiled gain init. A diagnostic run showed the gains all migrate to a
    # strong depth-dependent profile but only get there ~40% into the run (their
    # relative update rate is ~5x smaller than every other tensor's). GAIN_INIT=1
    # starts them there:  blocks[L].norm{1,2}.gains = A + B*L/(nlayers-1).
    GAIN_INIT = _e("GAIN_INIT", 0)
    # Embedding row norms are pure gauge (RMSNorm follows the embedding), so the init
    # scale here is a knob on Adam's *effective* embedding lr, not on the function.
    EMBED_STD = _e("EMBED_STD", 1.0)
    G1_A, G1_B = _e("G1_A", 0.55), _e("G1_B", 1.35)
    G2_A, G2_B = _e("G2_A", 0.41), _e("G2_B", 0.53)
    G_EMB, G_HEAD = _e("G_EMB", 2.0), _e("G_HEAD", 0.85)

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                if PROJ_STD > 0 and name.startswith("blocks."):
                    w.normal_(std=PROJ_STD * STD_FAC**0.5 / w.size(-1)**0.5)
                else:
                    w.zero_()
            elif "embed" in name:
                w.normal_(std=EMBED_STD)  # default torch init
            elif INIT_ORTHO:
                torch.nn.init.orthogonal_(w, gain=(STD_FAC * max(1, w.size(-2) / w.size(-1)))**0.5)
            else:
                sd = STD_FAC**0.5 / w.size(-1)**0.5  # default torch init
                if INIT_DEPTH and name.startswith("blocks."):
                    li = int(name.split(".")[1])
                    sd *= 1 + INIT_DEPTH * (li / (len(model.blocks) - 1) - 0.5)
                w.normal_(std=sd)
        elif name.endswith("bias"):
            w.fill_(FC_BIAS_INIT if (FC_BIAS_INIT and name.endswith("mlp.fc.bias")) else 0.0)
        elif name.endswith("gains"):
            g = 1.0
            if GAIN_INIT:
                nl = len(model.blocks)
                if name.startswith("blocks."):
                    L = int(name.split(".")[1]) / (nl - 1)
                    g = (G1_A + G1_B * L) if "norm1" in name else (G2_A + G2_B * L)
                else:
                    g = G_EMB if "norm1" in name else G_HEAD
            elif name == "norm1.gains":
                g = TOPGAIN_INIT
            w.normal_(mean=g, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    if QK_TIE_INIT or V_EYE_INIT:
        with torch.no_grad():
            for blk in model.blocks:
                if QK_TIE_INIT:
                    blk.attn.k.weight.copy_(blk.attn.q.weight)
                if V_EYE_INIT:
                    e = torch.eye(blk.attn.v.weight.size(0), blk.attn.v.weight.size(1),
                                  device=blk.attn.v.weight.device)
                    blk.attn.v.weight.copy_(e * (STD_FAC * blk.attn.v.weight.size(-2))**0.5
                                            / max(1.0, e.norm().item()))

    # create the optimizer(s)
    gains = [p for n, p in model.named_parameters() if n.endswith("gains")]
    g1 = [blk.norm1.gains for blk in model.blocks]
    g2 = [blk.norm2.gains for blk in model.blocks]
    if LR_GAIN1 > 0 or LR_GAIN2 > 0:
        drop = ([id(x) for x in g1] if LR_GAIN1 > 0 else []) + \
               ([id(x) for x in g2] if LR_GAIN2 > 0 else [])
        gains = [p for p in gains if id(p) not in drop]
    topgains = [model.norm1.gains, model.norm2.gains]
    if LR_TOPGAIN > 0:
        gains = [p for p in gains if all(p is not t for t in topgains)]
    vbiases = [blk.attn.v.bias for blk in model.blocks]
    pbiases = [b for blk in model.blocks for b in (blk.attn.proj.bias, blk.mlp.proj.bias)]
    kbiases = [blk.attn.k.bias for blk in model.blocks]
    special = ([id(v) for v in vbiases] if LR_VBIAS > 0 else []) + \
              ([id(b) for b in pbiases] if LR_PBIAS > 0 else []) + \
              ([id(b) for b in kbiases] if LR_KBIAS > 0 else [])
    biases = [p for n, p in model.named_parameters()
              if p.ndim < 2 and not n.endswith("gains") and p is not model.proj.bias
              and id(p) not in special]
    # beta2 per group: the one thing that helped was raising beta2, and the embedding's
    # gradients are the sparse ones, so allow each group its own second-moment horizon.
    # Per-group eps. With eps=1e-10 Adam is pure sign descent, which for the *embedding*
    # means rare-token rows (near-zero summed gradient) still take full-size steps — pure
    # noise. A per-group eps on the order of the gradient RMS suppresses exactly those.
    # Weight decay per group. The stock recipe decays the RMSNorm gains and the biases
    # along with everything else; standard practice excludes normalization/bias params,
    # and here it biases the gains toward 0 while they actually want 0.4..2.2.
    # The embedding's row norms are pure gauge, so its weight decay does not regularize —
    # it only sets how fast its effective lr decays (norms grow 1 -> ~15 over the run).
    # That is exactly the kind of temporal profile that has paid off elsewhere.
    WD_EMBED  = _e("WD_EMBED", ADAM_WD)
    # Muon on the embedding. I ruled this out by argument all session (the embedding's
    # gradient is row-sparse, so an orthogonalized update touches every vocabulary row
    # including tokens absent from the batch) — but FINAL taught me to measure instead.
    # lr is chosen to reproduce the current update RMS; wd is set to preserve lr*wd.
    EMBED_MUON = _e("EMBED_MUON", 0.0)
    EMBED_FP32 = _e("EMBED_FP32", 0)  # 1: fp32 master weights + moments for the embedding
    # The lm head is effectively unregularized: lr*wd = 0.004*0.001 = 4e-6 per step, so it
    # shrinks by 1% over the whole run. Front-loaded wd was a big win on the Muon side, and
    # wd here acts the same way — as an effective-lr control, not as regularization.
    WD_HEAD   = _e("WD_HEAD_ADAM", ADAM_WD)
    AWD_TILT_HEAD = _e("AWD_TILT_HEAD", 0.0)
    WD_GAIN   = _e("WD_GAIN", ADAM_WD)
    WD_BIAS   = _e("WD_BIAS", ADAM_WD)
    EPS_EMBED = _e("EPS_EMBED", ADAM_EPS)
    EPS_HEAD  = _e("EPS_HEAD", ADAM_EPS)
    B2_EMBED  = _e("B2_EMBED", ADAM_B2)
    B2_HEAD   = _e("B2_HEAD", ADAM_B2)
    B2_SCALAR = _e("B2_SCALAR", ADAM_B2)
    embed_group = dict(params=[model.embed.weight], lr=LR_EMBED, betas=(ADAM_B1, B2_EMBED),
                       tilt=TILT_EMBED, eps=EPS_EMBED, weight_decay=WD_EMBED,
                       cool=COOL_EMBED)
    adam_groups = ([] if (EMBED_FP32 or EMBED_MUON > 0) else [embed_group]) + [
                   *( [dict(params=gains, lr=LR_GAIN, betas=(ADAM_B1, B2_SCALAR),
                             tilt=TILT_SCALAR, weight_decay=WD_GAIN)]
                      if not (GAIN_DEPTH_TILT or GAIN_DEPTH_LR) else
                      [dict(params=[q for n, q in blk.named_parameters() if n.endswith("gains")],
                            lr=LR_GAIN * (1 + GAIN_DEPTH_LR * (li / (len(model.blocks) - 1) - 0.5)),
                            betas=(ADAM_B1, B2_SCALAR), weight_decay=WD_GAIN,
                            tilt=TILT_SCALAR + GAIN_DEPTH_TILT * (li / (len(model.blocks) - 1) - 0.5))
                       for li, blk in enumerate(model.blocks)]
                      + ([] if LR_TOPGAIN > 0 else
                         [dict(params=[model.norm1.gains, model.norm2.gains], lr=LR_GAIN,
                               betas=(ADAM_B1, B2_SCALAR), tilt=TILT_SCALAR,
                               weight_decay=WD_GAIN)]) ),
                   dict(params=[model.proj.bias], lr=LR_HBIAS, betas=(ADAM_B1, B2_SCALAR),
                        tilt=TILT_HBIAS, weight_decay=WD_BIAS),
                   dict(params=biases, lr=LR_SCALAR, betas=(ADAM_B1, B2_SCALAR),
                        tilt=TILT_BIAS, weight_decay=WD_BIAS)]
    for lrg, ps in ((LR_GAIN1, g1), (LR_GAIN2, g2)):
        if lrg > 0:
            adam_groups.append(dict(params=ps, lr=lrg, betas=(ADAM_B1, B2_SCALAR),
                                    tilt=TILT_SCALAR, weight_decay=WD_GAIN))
    if LR_VBIAS > 0:
        adam_groups.append(dict(params=vbiases, lr=LR_VBIAS, betas=(ADAM_B1, B2_SCALAR),
                                tilt=TILT_BIAS + TILT_RBIAS, weight_decay=WD_BIAS))
    if LR_PBIAS > 0:
        adam_groups.append(dict(params=pbiases, lr=LR_PBIAS, betas=(ADAM_B1, B2_SCALAR),
                                tilt=TILT_BIAS + TILT_RBIAS, weight_decay=WD_BIAS))
    if LR_KBIAS > 0:
        adam_groups.append(dict(params=kbiases, lr=LR_KBIAS, betas=(ADAM_B1, B2_SCALAR),
                                tilt=TILT_BIAS, weight_decay=WD_BIAS))
    if LR_TOPGAIN > 0:
        adam_groups.append(dict(params=[model.norm1.gains], lr=LR_TOPGAIN,
                                betas=(_e("B1_TOPGAIN", ADAM_B1),
                                       _e("B2_TOPGAIN", B2_SCALAR)),
                                tilt=TILT_SCALAR + TILT_TOPGAIN,
                                weight_decay=WD_GAIN if WD_TOPGAIN < 0 else WD_TOPGAIN))
        adam_groups.append(dict(params=[model.norm2.gains],
                                lr=LR_HEADNORM if LR_HEADNORM > 0 else LR_TOPGAIN,
                                betas=(ADAM_B1, B2_SCALAR), tilt=TILT_SCALAR,
                                weight_decay=WD_GAIN))
    LION_HEAD = _e("LION_HEAD", 0.0)   # >0: Lion (at this lr) for the lm head
    if HEAD_MUON <= 0 and LION_HEAD <= 0:
        adam_groups.insert(1, dict(params=[model.proj.weight], lr=LR_HEAD,
                                   betas=(ADAM_B1, B2_HEAD), tilt=TILT_HEAD,
                                   eps=EPS_HEAD, weight_decay=WD_HEAD,
                                   wdtilt=AWD_TILT_HEAD, cool=COOL_HEAD,
                                   **({"final": FINAL_HEAD} if FINAL_HEAD > 0 else {}),
                                   **({"dpow": DPOW_HEAD} if DPOW_HEAD > 0 else {})))
    # ADAM_KIND: 0 = AdamW (stock), 1 = NAdam (Nesterov-Adam, decoupled wd). NAdam also
    # carries a built-in beta1 ramp via momentum_decay, which is its own mechanism.
    if _e("ADAM_KIND", 0):
        optimizer1 = torch.optim.NAdam(adam_groups, betas=(ADAM_B1, ADAM_B2),
                                       eps=ADAM_EPS, weight_decay=ADAM_WD,
                                       decoupled_weight_decay=True)
    else:
        optimizer1 = AdamW(adam_groups, betas=(ADAM_B1, ADAM_B2), eps=ADAM_EPS,
                           weight_decay=ADAM_WD, fused=True,
                           amsgrad=bool(_e("AMSGRAD", 0)))
    embed_opt = (EmbedAdamW([embed_group], lr=LR_EMBED, betas=(ADAM_B1, B2_EMBED),
                            eps=EPS_EMBED, weight_decay=WD_EMBED) if EMBED_FP32 else None)
    nh = model.blocks[0].attn.num_heads
    qkv  = [p for n, p in model.blocks.named_parameters()
            if p.ndim >= 2 and any(f".attn.{t}." in n for t in "qkv")]
    aproj = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and ".attn.proj." in n]
    mlp_mats = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and ".mlp." in n]
    assert len(qkv) + len(aproj) + len(mlp_mats) == \
        len([p for p in model.blocks.parameters() if p.ndim >= 2])
    if DEPTH_LR or DEPTH_TILT:
        # one Muon param group per block, so lr and schedule shape can vary with depth
        nl = len(model.blocks)
        dgroups = []
        for li, blk in enumerate(model.blocks):
            f = li / (nl - 1) - 0.5
            parts = [(blk.attn, 0.0, WD_MUON), (blk.mlp, TILT_MLP, WD_MUON)]
            if OUT_MULT != 1.0 or FC_MULT != 1.0 or V_MULT != 1.0:
                parts = [([blk.attn.q.weight, blk.attn.k.weight], 0.0, WD_MUON, 1.0),
                         ([blk.attn.v.weight], 0.0, WD_MUON, V_MULT),
                         ([blk.mlp.fc.weight], TILT_MLP, WD_MUON, FC_MULT),
                         ([blk.attn.proj.weight], TILT_OUT, WD_MUON, OUT_MULT),
                         ([blk.mlp.proj.weight], TILT_MLP + TILT_OUT, WD_MUON, OUT_MULT)]
            elif TILT_OUT:
                parts = [([blk.attn.q.weight, blk.attn.k.weight, blk.attn.v.weight],
                          0.0, WD_MUON),
                         ([blk.mlp.fc.weight], TILT_MLP, WD_MUON),
                         ([blk.attn.proj.weight], TILT_OUT, WD_MUON),
                         ([blk.mlp.proj.weight], TILT_MLP + TILT_OUT, WD_MUON)]
            elif WD_QK > 0 or TILT_QK:
                qkw = [blk.attn.q.weight, blk.attn.k.weight]
                vpw = [blk.attn.v.weight, blk.attn.proj.weight]
                parts = [(qkw, TILT_QK, WD_QK or WD_MUON), (vpw, 0.0, WD_MUON),
                         (blk.mlp, TILT_MLP, WD_MUON)]
            for part in parts:
                sub, extra, wdg = part[0], part[1], part[2]
                mult = part[3] if len(part) > 3 else 1.0
                dgroups.append(dict(params=list(sub) if isinstance(sub, list)
                                    else [q for q in sub.parameters() if q.ndim >= 2],
                                    weight_decay=wdg * (1 + WDB_DEPTH * f),
                                    lr=LR_MUON * mult * (1 + DEPTH_LR * f),
                                    tilt=TILT_MUON + DEPTH_TILT * f + extra
                                         + DEPTH_TILT2 * (f * f - 0.087),
                                    wdtilt=WD_TILT + WD_DEPTH * f,
                                    mu0=min(0.99, max(0.3, MU0 + DEPTH_MU * f)),
                                    mu_end=min(0.99, max(0.3, MU + DEPTH_MUP * f))))
        optimizers = [optimizer1, Muon(dgroups, lr=LR_MUON, weight_decay=WD_MUON, mu=MU)]
    elif SPLIT_QKV or SPLIT_APROJ or MLP_MULT != 1.0:
        optimizers = [optimizer1,
                      Muon(qkv, lr=LR_MUON, weight_decay=WD_MUON, mu=MU,
                           split=nh if SPLIT_QKV else 0),
                      Muon(aproj, lr=LR_MUON, weight_decay=WD_MUON, mu=MU,
                           split=-nh if SPLIT_APROJ else 0),
                      Muon(mlp_mats, lr=LR_MUON * MLP_MULT, weight_decay=WD_MUON, mu=MU)]
    else:
        optimizers = [optimizer1, Muon(qkv + aproj + mlp_mats, lr=LR_MUON,
                                       weight_decay=WD_MUON, mu=MU)]
    if HEAD_MUON > 0:
        optimizers.append(Muon([model.proj.weight], lr=HEAD_MUON,
                               weight_decay=_e("WD_HEAD", 0.0), mu=MU))
    if EMBED_MUON > 0:
        optimizers.append(Muon([model.embed.weight], lr=EMBED_MUON, mu=MU,
                               weight_decay=ADAM_WD * LR_EMBED / EMBED_MUON))
    if LION_HEAD > 0:
        optimizers.append(Lion([dict(params=[model.proj.weight], lr=LION_HEAD,
                                     tilt=TILT_HEAD, cool=COOL_HEAD)],
                               lr=LION_HEAD, betas=(0.9, ADAM_B2), weight_decay=WD_HEAD))
    if embed_opt is not None:
        optimizers.append(embed_opt)
    LA_K = _e("LA_K", 0)
    if LA_K > 0:
        optimizers.append(Lookahead(list(model.parameters()), LA_K, _e("LA_ALPHA", 0.5)))
    if EMA_WIN > 0:
        optimizers.append(TailAverage(list(model.parameters()), train_steps, EMA_WIN, EMA_DECAY))
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            group.setdefault("tilt", TILT_MUON if "mu_t" in group else 0.0)
            group.setdefault("cool", COOL_MUON if "mu_t" in group else COOLDOWN)
            if "weight_decay" in group:
                group["initial_wd"] = group["weight_decay"]
            if "betas" in group:
                group["init_b2"] = group["betas"][1]

    # record the resolved knob values in the log (env overrides are screening-only)
    print0("CFG " + " ".join(f"{k}={v}" for k, v in dict(
        STEPS=train_steps, LR_MUON=LR_MUON, WD_MUON=WD_MUON, MU=MU, LR_EMBED=LR_EMBED,
        LR_HEAD=LR_HEAD, LION_HEAD=LION_HEAD, LR_SCALAR=LR_SCALAR, ADAM_B1=ADAM_B1, ADAM_B2=ADAM_B2,
        ADAM_EPS=ADAM_EPS, ADAM_WD=ADAM_WD, COOLDOWN=COOLDOWN, WARMUP=WARMUP,
        FINAL=FINAL, FLOOR_MODE=FLOOR_MODE, SCHED_MULT=SCHED_MULT, DECAY_SHAPE=DECAY_SHAPE, FINAL_HEAD=FINAL_HEAD, DPOW_HEAD=DPOW_HEAD, FLOOR_MIN=FLOOR_MIN, QK_GAUGE=QK_GAUGE, TILT_MIN=TILT_MIN, DECAY_POW=DECAY_POW, STD_FAC=STD_FAC, INIT_DEPTH=INIT_DEPTH, QK_TIE_INIT=QK_TIE_INIT, V_EYE_INIT=V_EYE_INIT, INIT_ORTHO=INIT_ORTHO,
        NS_ITERS=NS_ITERS, NS_FP32=NS_FP32, CAUTIOUS=CAUTIOUS, ADAMUON=ADAMUON,
        ADAMUON_B2=ADAMUON_B2, NESTEROV=NESTEROV, MU0=MU0, MU_WARM=MU_WARM,
        EMA_WIN=EMA_WIN, EMA_DECAY=EMA_DECAY, HEAD_MUON=HEAD_MUON,
        MLP_MULT=MLP_MULT, SCALE_RULE=SCALE_RULE, LR_GAIN=LR_GAIN, LR_HBIAS=LR_HBIAS,
        GAIN_INIT=GAIN_INIT, G1_A=G1_A, G1_B=G1_B, G2_A=G2_A, G2_B=G2_B,
        G_EMB=G_EMB, G_HEAD=G_HEAD, EMBED_STD=EMBED_STD, SPLIT_QKV=SPLIT_QKV,
        SPLIT_APROJ=SPLIT_APROJ, AEM_ALPHA=AEM_ALPHA, AEM_B3=AEM_B3, PRECOND=PRECOND,
        PRE_BETA=PRE_BETA, PRE_EPS=PRE_EPS, PRE_FREQ=PRE_FREQ, PRE_POW=PRE_POW,
        MU_NEST=MU_NEST, ORTH_FIRST=ORTH_FIRST, B2_EMBED=B2_EMBED, EPS_EMBED=EPS_EMBED, WD_EMBED=WD_EMBED, EMBED_FP32=EMBED_FP32, EMBED_MUON=EMBED_MUON, WD_HEAD=WD_HEAD, AWD_TILT_HEAD=AWD_TILT_HEAD, WD_GAIN=WD_GAIN, WD_BIAS=WD_BIAS, EPS_HEAD=EPS_HEAD, B2_HEAD=B2_HEAD,
        B2_SCALAR=B2_SCALAR, B1_0=B1_0, B1_WARM=B1_WARM, B2_LATE=B2_LATE, MU_LATE=MU_LATE, MU_SHAPE=MU_SHAPE,
        TILT_EMBED=TILT_EMBED, TILT_HEAD=TILT_HEAD, TILT_SCALAR=TILT_SCALAR,
        TILT_MUON=TILT_MUON, WD_COUPLE=WD_COUPLE, AWD_COUPLE=AWD_COUPLE, WD_TILT=WD_TILT, WD_FLOOR=WD_FLOOR, WD_DEPTH=WD_DEPTH, WDB_DEPTH=WDB_DEPTH, AWD_TILT=AWD_TILT,
        TILT_HBIAS=TILT_HBIAS, TILT_BIAS=TILT_BIAS, TILT_RBIAS=TILT_RBIAS, GAIN_DEPTH_TILT=GAIN_DEPTH_TILT, LR_GAIN1=LR_GAIN1, LR_GAIN2=LR_GAIN2, GAIN_DEPTH_LR=GAIN_DEPTH_LR, LR_TOPGAIN=LR_TOPGAIN, LR_HEADNORM=LR_HEADNORM, TOPGAIN_INIT=TOPGAIN_INIT, FC_BIAS_INIT=FC_BIAS_INIT, PROJ_STD=PROJ_STD, WD_TOPGAIN=WD_TOPGAIN, TILT_TOPGAIN=TILT_TOPGAIN, LR_VBIAS=LR_VBIAS, LR_PBIAS=LR_PBIAS, LR_KBIAS=LR_KBIAS, TILT_POW=TILT_POW, TILT_PHASE=TILT_PHASE, COOL_HEAD=COOL_HEAD, COOL_MUON=COOL_MUON, COOL_EMBED=COOL_EMBED, DEPTH_LR=DEPTH_LR, DEPTH_TILT=DEPTH_TILT, TILT_MLP=TILT_MLP, WD_QK=WD_QK, TILT_QK=TILT_QK, TILT_OUT=TILT_OUT, OUT_MULT=OUT_MULT, FC_MULT=FC_MULT, V_MULT=V_MULT, DEPTH_MU=DEPTH_MU, DEPTH_MUP=DEPTH_MUP, DEPTH_TILT2=DEPTH_TILT2).items()),
        console=True)

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

    # learning rate schedule: (optional warmup) then stable then decay
    def set_hparams(step, cooldown_frac=COOLDOWN):
        progress = step / train_steps
        assert 0 <= progress < 1
        dprog = progress if SCHED_MULT == 1.0 else min(1.0, progress / SCHED_MULT)
        if dprog < 1 - cooldown_frac:
            eta = 1.0
        else:
            dec = (1 - dprog) / cooldown_frac           # 1 -> 0 over the cooldown
            shp = (0.5 * (1 - math.cos(math.pi * dec)) if DECAY_SHAPE
                   else dec**DECAY_POW)
            eta = shp if FLOOR_MODE else FINAL + (1 - FINAL) * shp
        if WARMUP > 0 and progress < WARMUP:
            eta = min(eta, (step + 1) / (WARMUP * train_steps))
        if TILT_PHASE:
            tshape = (1.0 if progress < 1 - cooldown_frac
                      else 1 - 2 * (progress - (1 - cooldown_frac)) / cooldown_frac)
            tshape -= 1 - cooldown_frac          # mean-centre
        else:
            tshape = 2 / (TILT_POW + 1) - 2 * progress**TILT_POW
        for opt in optimizers:
            for group in opt.param_groups:
                cg = group["cool"]
                fg = group.get("final", FINAL)
                dp = group.get("dpow", DECAY_POW)
                if cg == cooldown_frac and fg == FINAL and dp == DECAY_POW:
                    eta_g = eta
                elif progress < 1 - cg:
                    eta_g = 1.0
                else:
                    d = (1 - progress) / cg
                    sh = (0.5 * (1 - math.cos(math.pi * d)) if DECAY_SHAPE
                          else d**dp)
                    eta_g = sh if FLOOR_MODE else fg + (1 - fg) * sh
                t = group["tilt"]
                m = max(TILT_MIN, 1 + t * tshape) if t else 1.0
                group["lr"] = group["initial_lr"] * (max(eta_g * m, FINAL) if FLOOR_MODE
                                                     else max(eta_g * m, FLOOR_MIN))
                muon = "mu_t" in group
                c = WD_COUPLE if muon else AWD_COUPLE
                wt = group.get("wdtilt", WD_TILT if muon else AWD_TILT)
                if "initial_wd" in group and (c != 1.0 or wt):
                    group["weight_decay"] = (group["initial_wd"]
                                             * max(eta, 1e-8)**(c - 1)
                                             * max(WD_FLOOR, 1 + wt * (1 - 2 * progress)))
        if MU_WARM > 0 or MU_LATE != MU:
            if progress < MU_WARM:
                if MU_SHAPE:
                    h0, h1 = 1 / (1 - MU0), 1 / (1 - MU)
                    mu_now = 1 - 1 / (h0 + (h1 - h0) * progress / MU_WARM)
                else:
                    mu_now = MU0 + (MU - MU0) * progress / MU_WARM
            else:
                mu_now = MU + (MU_LATE - MU) * (progress - MU_WARM) / (1 - MU_WARM)
            for opt in optimizers:
                for group in opt.param_groups:
                    if "mu_t" in group:
                        m0 = group.get("mu0", MU0)
                        me = group.get("mu_end", MU)
                        if me != MU:
                            v = (m0 + (me - m0) * progress / MU_WARM
                                 if progress < MU_WARM else
                                 me + (MU_LATE - MU) * (progress - MU_WARM) / (1 - MU_WARM))
                        else:
                            v = (mu_now if (MU_SHAPE or progress >= MU_WARM)
                                 else m0 + (MU - m0) * progress / MU_WARM)
                        group["mu_t"].fill_(v)
        if B2_LATE != ADAM_B2:
            shift = (B2_LATE - ADAM_B2) * progress
            for group in optimizer1.param_groups:
                group["betas"] = (group["betas"][0],
                                  min(0.9999, group["init_b2"] + shift))
        if B1_WARM > 0:
            b1 = B1_0 + (ADAM_B1 - B1_0) * min(1.0, progress / B1_WARM)
            for group in optimizer1.param_groups:
                group["betas"] = (b1, group["betas"][1])
        if QK_GAUGE > 0:
            for blk in model.blocks:
                for lin in (blk.attn.q, blk.attn.k):
                    ref = QK_GAUGE * (STD_FAC * lin.weight.size(-2))**0.5
                    c = ref / lin.weight.detach().norm().clamp(min=1e-12)
                    lin.weight.detach().mul_(c)
                    lin.bias.detach().mul_(c)
        if DIAG and step % DIAG == 0:
            # per-tensor weight RMS and relative update size, to spot mis-scaled groups
            muon_p = {id(p) for opt in optimizers for g in opt.param_groups
                      for p in g["params"] if "mu_t" in g}
            # cos(fresh gradient, accumulated first moment) per tensor: a direct read on
            # whether each group's momentum is still useful or has gone stale.
            mom_of = {}
            for opt in optimizers:
                for pp, stt in getattr(opt, "state", {}).items():
                    mv = stt.get("momentum", stt.get("exp_avg"))
                    if mv is not None:
                        mom_of[id(pp)] = mv
            lines = []
            for name, p in model.named_parameters():
                w = p.detach().float()
                wr = w.square().mean().sqrt().item()
                gr = p.grad.detach().float().square().mean().sqrt().item() if p.grad is not None else 0.0
                if id(p) in muon_p:
                    grp = next(g for opt in optimizers for g in opt.param_groups
                               if any(q is p for q in g["params"]))
                    m, n = p.size(-2), p.size(-1)
                    ur = grp["lr"] * max(1, m / n)**0.5 * (min(m, n) / (m * n))**0.5
                else:
                    st = optimizer1.state.get(p, {})
                    if "exp_avg" in st:
                        grp = next(g for g in optimizer1.param_groups if any(q is p for q in g["params"]))
                        ur = grp["lr"] * (st["exp_avg"].float() /
                                          (st["exp_avg_sq"].float().sqrt() + ADAM_EPS)
                                          ).square().mean().sqrt().item()
                    else:
                        ur = 0.0
                cs = float("nan")
                mv = mom_of.get(id(p))
                if mv is not None and p.grad is not None:
                    gf = p.grad.detach().float().flatten()
                    mf = mv.detach().float().flatten()
                    den = (gf.norm() * mf.norm()).clamp(min=1e-20)
                    cs = (gf @ mf / den).item()
                lines.append(f"D {step} {name} wrms={wr:.4g} grms={gr:.4g} urms={ur:.4g} "
                             f"rel={ur / max(wr, 1e-12):.4g} cos={cs:.4g}")
            print0("\n".join(lines))


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
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
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
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
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

    if wandb_run is not None:
        wandb_run.summary["final_val_loss"] = float(val_loss)
        wandb_run.finish()

dist.destroy_process_group()

