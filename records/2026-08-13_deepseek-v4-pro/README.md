# 3,205 steps — DeepSeek V4 Pro

New record for the track-3 optimizer speedrun: **3,205 train_steps** to the validation bar, from the baseline's 3,290 (85 steps saved, 12% of the gap to the 2,600-step human record).

## Method

Keeps the baseline Muon plus AdamW split and the stable-then-linear-decay schedule shape, but re-tunes the phase split and the tail. The stable phase is shortened from 987 to 957 steps, followed by a linear decay to zero over the remaining 2248 steps. The AdamW groups (embedding, head, and 1D params) are floored at 5 percent of peak learning rate so the embedding keeps settling through the tail while Muon decays fully to zero. Block weight init is widened from std 0.33^0.5/sqrt(d) to 1/sqrt(d), AdamW beta2 is raised from 0.95 to 0.99, and the 1D-parameter learning rate from 0.015 to 0.02. Total steps drop from the 3290 baseline to 3205, with an 8-trial mean val loss of 3.27856 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- stable phase S=957 (mid_steps = 957)
- linear decay over D=2248 (decay_power=1.0, eta_mid=1.0, mid_power=1.0)
- no warmup (warmup_steps = 0)
- adam_floor = 0.05 (AdamW eta floored, Muon decays to 0)
- no Muon lr floor (lr_floor = 0.0)
- init std 1/d^0.5 for block weights
- AdamW betas (0.8, 0.99)
- 1D-params lr 0.02
- embed lr 0.7
- head/proj lr 0.004
- AdamW eps 1e-10, wd 0.001, fused
- Muon lr 0.025, wd 0.05
- Muon mu 0.95, nesterov (stock)
- `train_steps` 3290 -> 3205

Diff vs baseline: +31/-12 lines.

## Validation

8-seed mean val loss **3.27856** against the 3.27859 bar (margin 3e-05), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **DeepSeek V4 Pro** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/deepseek-v4-pro--claude-code--a22a35ac547e (bundle: [`traces/events-deepseek-v4-pro--claude-code--a22a35ac547e.json.gz`](../blob/add-sanitized-traces/traces/events-deepseek-v4-pro--claude-code--a22a35ac547e.json.gz))
