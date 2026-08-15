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
def muon_update(grad, momentum, row_variance_ema, col_variance_ema,
                mu=0.95, beta2=0.80, col_beta2=0.80, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    row_variance = update.float().square().mean(dim=-1, keepdim=True)
    row_variance_ema.lerp_(row_variance, 1 - beta2)
    update *= (row_variance_ema.rsqrt() / grad.size(-1)**0.5).to(update.dtype)
    if grad.size(-2) > grad.size(-1):
        col_variance = update.float().square().mean(dim=-2, keepdim=True)
        col_variance_ema.lerp_(col_variance, 1 - col_beta2)
        update *= (col_variance_ema.rsqrt() / grad.size(-1)**0.5).to(update.dtype)
    return update

@torch.compile
def scheduled_adamw_update(param, grad, exp_avg, exp_avg_sq, step, beta2_prod,
                           lr, beta2, beta1=0.8, eps=1e-10, weight_decay=0.001):
    param.mul_(1 - lr * weight_decay)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).add_(grad.square() * (1 - beta2))
    step.add_(1)
    beta2_prod.mul_(beta2)
    bias1 = 1 - beta1**step
    bias2 = 1 - beta2_prod
    denom = exp_avg_sq.sqrt() / bias2.sqrt() + eps
    param.add_(exp_avg / denom * (-lr / bias1))

class ScheduledAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.004, beta1=0.8, beta2=0.99,
                 eps=1e-10, weight_decay=0.001):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            device = group["params"][0].device
            group["lr"] = torch.tensor(lr, device=device)
            group["beta1"] = torch.tensor(beta1, device=device)
            group["beta2"] = torch.tensor(beta2, device=device)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = torch.zeros((), device=p.device)
                    state["beta2_prod"] = torch.ones((), device=p.device)
                scheduled_adamw_update(
                    p, p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"],
                    state["beta2_prod"], group["lr"], group["beta2"],
                    beta1=group["beta1"], eps=group["eps"],
                    weight_decay=group["weight_decay"])

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

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
                        state["row_variance"] = torch.full(
                            (p.size(0), 1), 1 / max(p.size()), device=p.device, dtype=torch.float32)
                        state["col_variance"] = torch.full(
                            (1, p.size(1)), 1 / p.size(1), device=p.device, dtype=torch.float32)
                    mu = (group["tall_mu"] if p.size(0) > p.size(1) else
                          group["wide_mu"] if p.size(0) < p.size(1) else group["mu"])
                    update = muon_update(p.grad, state["momentum"], state["row_variance"],
                                         state["col_variance"], mu=mu)
                    lr = group["tall_lr"] if p.size(0) > p.size(1) else group["lr"]
                    p.mul_(1 - lr * group["weight_decay"])
                    p.add_(update, alpha=-lr)
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
    train_steps = 3058

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if name == "proj.weight":
                torch.nn.init.orthogonal_(
                    w, gain=0.20 * (0.33 * w.size(0) / w.size(1))**0.5)
            elif "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.70),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer_proj = ScheduledAdamW([model.proj.weight])
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.026, weight_decay=0.05)
    for group in optimizer2.param_groups:
        group["mu"] = torch.tensor(0.85, device=device)
        group["tall_mu"] = torch.tensor(0.85, device=device)
        group["wide_mu"] = torch.tensor(0.85, device=device)
    optimizers = [optimizer1, optimizer_proj, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = float(group["lr"])

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
    def set_hparams(step, cooldown_frac=0.70, schedule_steps=3135):
        progress = step / schedule_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group_idx, group in enumerate(opt.param_groups):
                if opt is optimizer2:
                    opt_eta = eta**1.25
                else:
                    opt_eta = eta
                if opt is optimizer_proj:
                    group["lr"].fill_(group["initial_lr"] * opt_eta)
                else:
                    group["lr"] = group["initial_lr"] * opt_eta
                if opt is optimizer2:
                    group["tall_lr"] = 0.028 * eta**1.25
        for group in optimizer_proj.param_groups:
            group["beta1"].fill_(0.80)
            group["beta2"].fill_(0.95 + 0.04 * min(step / 400, 1))
        for group in optimizer2.param_groups:
            group["mu"].fill_(0.85 + 0.115 * min(step / 600, 1))
            group["tall_mu"].fill_(0.85 + 0.120 * min(step / 600, 1))
            group["wide_mu"].fill_(0.85 + 0.115 * min(step / 600, 1))


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

