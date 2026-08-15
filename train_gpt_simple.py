"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

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

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True, prenorm=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    if prenorm:
        update = update / (update.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8)  # pre-NS row norm
    update = zeropower_via_newtonschulz5(update).float()
    update *= max(1, max(grad.size(-2), grad.size(-1)) / min(grad.size(-2), grad.size(-1)))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, lr_mults=None, qkv_groups=(), mu_overrides=None):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.lr_mults = lr_mults or {}
        self.mu_overrides = mu_overrides or {}
        self.prenorm_skips = set()
        self.cooldown_overrides = {}
        self.eta_mults = {}
        self.perhead = set()  # params orthogonalized per attention head
        self.wd_mults = {}
        # items = independent update units: either a single param or a list of params
        # whose grads are stacked (rows dim) and orthogonalized jointly (e.g. q,k,v)
        grouped = {p for g in qkv_groups for p in g}
        items = [[p] for p in params if p not in grouped] + [list(g) for g in qkv_groups]
        items.sort(key=lambda ps: -sum(p.numel() for p in ps))
        self.items = items

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        group = self.param_groups[0]
        items = self.items
        assert len(items) % world_size == 0, f"{len(items)} items not divisible by world_size {world_size}"
        for base_i in range(0, len(items), world_size):
            round_items = items[base_i:base_i + world_size]
            max_numel = max(sum(p.numel() for p in it) for it in round_items)
            mine = round_items[rank]
            if mine:
                key = mine[0]
                state = self.state[key]
                rows = sum(p.size(-2) for p in mine)
                cols = mine[0].size(-1)
                perhead = key in self.perhead and len(mine) == 1
                nheads = 6
                if len(state) == 0:
                    mom_shape = (nheads, rows // nheads, cols) if perhead else (rows, cols)
                    state["momentum"] = torch.zeros(mom_shape, device=key.device, dtype=torch.float32)

                g = torch.cat([p.grad.float() for p in mine], dim=-2) if len(mine) > 1 else mine[0].grad
                if perhead:
                    g = g.view(nheads, rows // nheads, cols)
                update = muon_update(g, state["momentum"],
                                     mu=self.mu_overrides.get(key, group["mu"]),
                                     prenorm=key not in self.prenorm_skips)
                if perhead:
                    update = update.view(rows, cols)
                mult = self.lr_mults.get(key, 1.0) * self.eta_mults.get(key, 1.0)
                offset = 0
                for p in mine:
                    n = p.size(-2)
                    p.mul_(1 - group["lr"] * mult * group["weight_decay"])
                    p.add_(update[offset:offset + n], alpha=-group["lr"] * mult)
                    offset += n
                send = torch.cat([p.detach().flatten() for p in mine])
                send = torch.cat([send, send.new_zeros(max_numel - send.numel())])
            else:
                send = torch.zeros(max_numel, device="cuda")
            recv = [torch.empty(max_numel, device=send.device, dtype=send.dtype) for _ in range(world_size)]
            dist.all_gather(recv, send)
            for j, it in enumerate(round_items):
                if j == rank or not it:
                    continue
                offset = 0
                for p in it:
                    n = p.numel()
                    p.data.copy_(recv[j][offset:offset + n].view_as(p))
                    offset += n


@torch.compile
def fp32_adam_update(g, m, v, bc1, bc2, beta1, beta2, eps):
    # classic AdamW direction, all state/compute in fp32 (fused keeps bf16 for bf16 params)
    m.lerp_(g, 1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    return (m / bc1) / ((v / bc2).sqrt() + eps)

class Fp32AdamW(torch.optim.Optimizer):
    """AdamW with fp32 states (for the bf16 embedding table, whose fused states are bf16).
    Params are replicated across ranks — every rank updates identically, no gather."""

    def __init__(self, params, lr=0.002, betas=(0.8, 0.99), eps=1e-8, weight_decay=0.001):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]
                bc1 = torch.tensor(1 - beta1 ** t, device=p.device)
                bc2 = torch.tensor(1 - beta2 ** t, device=p.device)
                update = fp32_adam_update(g.float(), state["m"], state["v"], bc1, bc2, beta1, beta2, eps)
                master = getattr(self, "master_weight", None)
                if master is not None:
                    # mixed-precision master weights: accumulate in fp32, cast to bf16 for the model
                    master.mul_(1 - group["lr"] * group["weight_decay"])
                    master.add_(update, alpha=-group["lr"])
                    p.data.copy_(master.to(p.dtype))
                else:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.to(p.dtype), alpha=-group["lr"])


@torch.compile
def soap_update(g, m, v, QL, QR, bc1, bc2, beta1, beta2, eps):
    # project the gradient into the Shampoo eigenbasis
    g_proj = QR.mT @ g @ QL
    # Adam update in the eigenbasis
    m.lerp_(g_proj, 1 - beta1)
    v.mul_(beta2).addcmul_(g_proj, g_proj, value=1 - beta2)
    update_proj = (m / bc1) / ((v / bc2).sqrt() + eps)
    # project back to the original basis
    return QR @ update_proj @ QL.mT

class SOAP(torch.optim.Optimizer):
    """SOAP: Shampoo Kronecker preconditioner with Adam run in the eigenbasis.

    Distributed across ranks exactly like Muon: each rank owns a shard of the
    params, maintains that shard's preconditioner state, and updated params are
    all-gathered. The eigenbasis is refreshed every `precond_freq` steps.
    """
    def __init__(self, params, lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 precond_freq=10, shampoo_beta=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precond_freq=precond_freq, shampoo_beta=shampoo_beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            beta1, beta2 = group["betas"]
            sb = group["shampoo_beta"]
            freq = group["precond_freq"]
            eps = group["eps"]
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    g = p.grad.float()
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        rows, cols = g.shape
                        state["LL"] = torch.zeros(cols, cols, device=g.device, dtype=torch.float32)
                        state["RR"] = torch.zeros(rows, rows, device=g.device, dtype=torch.float32)
                        state["QL"] = torch.eye(cols, device=g.device, dtype=torch.float32)
                        state["QR"] = torch.eye(rows, device=g.device, dtype=torch.float32)
                        state["m"] = torch.zeros_like(g)
                        state["v"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    # EMA update of the Shampoo Kronecker factors
                    state["LL"].mul_(sb).add_(g.mT @ g, alpha=1 - sb)   # G^T G  (in x in)
                    state["RR"].mul_(sb).add_(g @ g.mT, alpha=1 - sb)   # G G^T  (out x out)
                    # periodic refresh of the eigenbasis
                    if t % freq == 0:
                        state["QL"] = torch.linalg.eigh(state["LL"])[1]
                        state["QR"] = torch.linalg.eigh(state["RR"])[1]
                    bc1 = torch.tensor(1 - beta1 ** t, device=g.device)
                    bc2 = torch.tensor(1 - beta2 ** t, device=g.device)
                    update = soap_update(g, state["m"], state["v"], state["QL"], state["QR"],
                                         bc1, bc2, beta1, beta2, eps)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.to(p.dtype), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


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
    train_steps = 2968  # CONFIRM stack24 @2968 (8-trial record attempt)

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 * 1.4 / w.size(-1)**0.5)  # 1.4x init scale (best)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer0 = Fp32AdamW([dict(params=[model.embed.weight], lr=0.85, betas=(0.7, 0.99))],
                           betas=(0.8, 0.99), eps=1e-8, weight_decay=0.001)
    optimizer0.master_weight = model.embed.weight.detach().float().clone()
    optimizer1 = AdamW([dict(params=[model.proj.weight], lr=0.0045, cooldown=0.65),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                       betas=(0.8, 0.99), eps=1e-8, weight_decay=0.001, fused=True)
    _muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    optimizer2 = Muon(_muon_params, lr=0.025, weight_decay=0.05,
                      lr_mults={block.mlp.fc.weight: 1.5 for block in model.blocks}
                      | {block.mlp.proj.weight: 1.25 for block in model.blocks}
                      | {p: 0.75 for block in model.blocks
                         for p in [block.attn.q.weight, block.attn.k.weight, block.attn.v.weight]}
                      | {block.attn.proj.weight: 0.55 for block in model.blocks},
                      mu_overrides={})
    optimizer2.prenorm_skips = set()
    optimizer2.perhead = set()
    optimizers = [optimizer0, optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

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

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        # Muon momentum warmup: 0.85 -> 0.96 over the first 300 steps
        # (discrete values so the compiled muon_update is not re-traced every step)
        mu = 0.85 if step < 75 else 0.875 if step < 150 else 0.9 if step < 225 else 0.925 if step < 300 else 0.96
        for opt in optimizers:
            for group in opt.param_groups:
                cf = group.get("cooldown", cooldown_frac)  # per-group cooldown override
                fl = group.get("floor", 0.035)  # per-group lr floor override
                eta = 1.0 if progress < 1 - cf else max((1 - progress) / cf, fl)
                group["lr"] = group["initial_lr"] * eta
                if "mu" in group:
                    group["mu"] = mu
                # per-param cooldown overrides: store relative eta multiplier
                if getattr(opt, "cooldown_overrides", None):
                    fl_over = getattr(opt, "floor_overrides", {})
                    for p, cf_p in opt.cooldown_overrides.items():
                        fl_p = fl_over.get(p, fl)
                        eta_p = 1.0 if progress < 1 - cf_p else max((1 - progress) / cf_p, fl_p)
                        opt.eta_mults[p] = eta_p / eta


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