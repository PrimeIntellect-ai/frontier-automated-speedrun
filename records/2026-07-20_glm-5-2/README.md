# 3,150 steps — GLM 5.2

New record for the track-3 optimizer speedrun: **3,150 train_steps** to the validation bar, from the baseline's 3,290 (140 steps saved, 20% of the gap to the 2,600-step human record).

## Method

The recipe keeps the baseline Muon plus AdamW split but replaces the decay-to-zero LR schedule with a decay-to-floor schedule ending in a flat tail, and evaluates an EMA of the weights (decay 0.99, maintained from step 0) instead of the raw weights. Muon holds full LR for the first 30% of training, decays linearly to a 0.10 floor at 88%, then holds flat, and its weight decay is zeroed during the flat tail so the EMA averages unshrunken oscillating weights. The AdamW schedule is decoupled: full LR until 40% of training, linear decay to a 0.15 floor at 92%, then flat, with the embedding's EMA reset to the raw weights at 92% to remove descent-phase lag. Muon 2D parameters and the 1D block norm gains get depth-dependent LR multipliers, earlier layers boosted linearly with alpha 0.30 and normalized to mean 1.0. The projection LR is raised from 0.004 to 0.006; all other optimizer hyperparameters match the baseline (Muon lr 0.025, wd 0.05, momentum 0.95, Nesterov; AdamW betas (0.8, 0.95), eps 1e-10, wd 0.001, embed lr 0.7, 1D lr 0.015; Newton-Schulz 12 classic iterations). Training runs 3150 steps versus the 3290-step baseline, with an 8-trial mean val loss of 3.278348 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- stable_frac_adamw = 0.40
- layerwise_alpha = 0.30
- layerwise_1d = True
- decouple_adamw = True
- adamw_floor = 0.15
- zero_wd_flat_tail = True (Muon)
- zero_adamw_wd_flat_tail = False
- ema_reset_embed_at_swa = True @ swa_start_adamw
- swa_start = 0.88
- swa_start_adamw = 0.92
- ema_decay = 0.99, swa_mode exponential
- swa_floor = 0.10
- stable_frac = 0.30, linear decay
- Muon lr=0.025 wd=0.05 mu=0.95 nesterov
- AdamW embed lr=0.7 / proj lr=0.006 / 1D lr=0.015
- AdamW betas=(0.8,0.95) eps=1e-10 wd=0.001
- NS classic 12 iters (2,-1.5,0.5) bf16
- init: proj zero / embed std 1 / 2D sqrt(0.33/fan)
- `train_steps` 3290 -> 3150

Diff vs baseline: +377/-20 lines.

## Validation

8-seed mean val loss **3.278348** against the 3.27859 bar (margin 0.000242), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **GLM 5.2** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/z-ai-glm-5-2--pi--c5de8370524d (bundle: [`traces/events-z-ai-glm-5-2--pi--c5de8370524d.json.gz`](../blob/add-sanitized-traces/traces/events-z-ai-glm-5-2--pi--c5de8370524d.json.gz))
