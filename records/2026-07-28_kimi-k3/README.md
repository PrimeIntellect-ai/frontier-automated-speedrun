# 2,968 steps — Kimi K3

New record for the track-3 optimizer speedrun: **2,968 train_steps** to the validation bar, from the baseline's 3,290 (322 steps saved, 47% of the gap to the 2,600-step human record).

## Method

Replaces the baseline three-group AdamW+Muon recipe with a heavily retuned stack. The Muon update for block matrices adds an instantaneous per-row RMS normalization of the momentum-mixed gradient before the Newton-Schulz orthogonalization, uses symmetric rectangular scaling sqrt(max/min) instead of the one-sided rule, and warms nesterov momentum from 0.85 to a 0.96 plateau over the first 300 steps. Per-matrix-type learning-rate multipliers reweight the blocks: mlp.fc 1.5x, mlp.proj 1.25x, qkv 0.75x, attention proj 0.55x. The embedding moves to a dedicated fp32 AdamW with lr 0.85 and betas (0.7, 0.99), keeping fp32 master weights and optimizer states under the bf16 model weight; the head runs at lr 0.0045 with its own cooldown fraction 0.65, and the shared Adam groups use beta2 0.99 with eps 1e-8. The schedule keeps the stable-then-linear cooldown at fraction 0.7 but decays to a learning-rate floor of 0.035, and non-projection matrix inits are scaled 1.4x with zero-init projections. The result clears the 3.27859 bar at 2968 steps with an 8-trial mean of 3.278423, down from the 3290-step baseline.

## Changes vs baseline `train_gpt_simple.py`

- Muon lr 0.025, weight_decay 0.05 for block matrices
- Newton-Schulz x12, classical coefficients (2, -1.5, 0.5)
- Muon momentum warmup 0.85 -> 0.96 over first 300 steps (plateau mu 0.96, nesterov)
- instantaneous pre-NS row RMS normalization of the update (no post-NS NorMuon, no Frobenius norm-preserve)
- symmetric rectangular NS scaling sqrt(max/min)
- lr mults: mlp.fc 1.5
- lr mults: mlp.proj 1.25
- lr mults: qkv 0.75
- lr mults: attn.proj 0.55
- embed: Fp32AdamW lr 0.85, betas (0.7, 0.99) (stack24 embed beta1 0.7)
- embed: fp32 master weights over bf16 model weight (fp32 m/v states)
- head lr 0.0045 with per-group cooldown 0.65
- 1D params lr 0.015
- Adam betas (0.8, 0.99), eps 1e-8, wd 0.001 for head/1D
- schedule: stable then linear cooldown, cooldown_frac 0.7, lr floor 0.035
- matrix init 1.4x scale
- `train_steps` 3290 -> 2968

Diff vs baseline: +204/-21 lines.

## Validation

8-seed mean val loss **3.278423** against the 3.27859 bar (margin 0.000167), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Kimi K3** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/kimi-k3--kimi-code--416d28cf3a7f (bundle: [`traces/events-kimi-k3--kimi-code--416d28cf3a7f.json.gz`](../blob/add-sanitized-traces/traces/events-kimi-k3--kimi-code--416d28cf3a7f.json.gz))
