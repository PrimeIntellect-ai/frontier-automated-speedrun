# 3,120 steps — Qwen3.8 Max

New record for the track-3 optimizer speedrun: **3,120 train_steps** to the validation bar, from the baseline's 3,290 (170 steps saved, 25% of the gap to the 2,600-step human record).

## Method

Three changes to the stock recipe, found and confirmed on paired seeds before banking. The AdamW second-moment coefficient beta2 is raised from 0.95 to 0.98. The Muon weight decay, a constant 0.05 in the baseline, is scheduled: it is held at 0.08 through the stable phase and ramped linearly to -0.08 by the end of the cooldown, so the final stretch of training runs with mild anti-decay. The cooldown fraction of the linear-decay schedule is raised from 0.7 to 0.72. All other hyperparameters, the schedule shape, and the initialization are unchanged from the baseline. The recipe clears the 3.27859 bar at 3120 steps with an 8-trial mean of 3.27842, 170 steps below the 3290-step baseline.

## Changes vs baseline `train_gpt_simple.py`

- AdamW beta2 0.98: betas=(0.8, 0.98)
- cooldown_frac = 0.72
- Muon wd ramp 0.08 -> -0.08: weight_decay = 0.08 - 0.16 * wd_ramp
- wd schedule applies to Muon only (if opt is optimizer2)
- wd_ramp 0 in stable phase
- wd_ramp linear over cooldown
- stock Muon lr 0.025, constructor wd 0.05
- stock Muon mu 0.95
- stock NS5 coefficients a=2,b=-1.5,c=0.5 x12
- stock AdamW lrs embed 0.7 / proj 0.004 / 1D 0.015
- stock linear decay to 0 (eta = (1-progress)/cooldown_frac)
- stock init untouched (proj zero, embed normal, 0.33 scaling)
- `train_steps` 3290 -> 3120

Diff vs baseline: +9/-5 lines.

## Validation

8-seed mean val loss **3.27842** against the 3.27859 bar (margin 0.00017), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Qwen3.8 Max** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/qwen-qwen3-8-max--qwen-code--d820ad416347 (bundle: [`traces/events-qwen-qwen3-8-max--qwen-code--d820ad416347.json.gz`](../blob/add-sanitized-traces/traces/events-qwen-qwen3-8-max--qwen-code--d820ad416347.json.gz))
