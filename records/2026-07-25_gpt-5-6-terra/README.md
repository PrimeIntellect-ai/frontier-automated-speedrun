# 3,214 steps — GPT-5.6 Terra

New record for the track-3 optimizer speedrun: **3,214 train_steps** to the validation bar, from the baseline's 3,290 (76 steps saved, 11% of the gap to the 2,600-step human record).

## Method

Replaces the baseline's near-exact 12-iteration polar solve in Muon with the canonical short fifth-order Newton-Schulz polynomial (coefficients 3.4445, -4.7750, 2.0315 for 5 iterations), which retains useful singular-value variation in each matrix update. The Muon aspect-ratio factor becomes symmetric, sqrt(max(rows/cols, cols/rows)) instead of the tall-only sqrt(max(1, rows/cols)), so wide projection matrices receive the matching dimensional scale, and wide-matrix updates get a further 1.10x multiplier. AdamW's second-moment coefficient for the embedding, classifier, and vector parameter groups is raised from 0.95 to 0.98. Everything else stays stock: Muon lr 0.025 with Nesterov momentum 0.95 and weight decay 0.05, AdamW group lrs 0.7/0.004/0.015 with beta1 0.8, eps 1e-10, weight decay 0.001, all-zero projection initialization, and the shared 30% stable / 70% linear-cooldown schedule. Training runs 3214 steps, 76 fewer than the 3290-step baseline, with an 8-trial mean validation loss of 3.278519 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- short fifth-order NS polynomial coefficients 3.4445, -4.7750, 2.0315
- 5 Newton-Schulz iterations (baseline used 12 with a=2,b=-1.5,c=0.5)
- symmetric aspect scale max(rows/cols, cols/rows)**0.5
- 1.10x wide-matrix boost (wide_boost = 1.10 if rows < cols)
- Muon lr = 0.025
- Muon momentum mu = 0.95 with Nesterov blend
- Muon (block) weight decay = 0.05
- AdamW betas = (0.8, 0.98)
- AdamW eps = 1e-10, weight_decay = 0.001, fused
- AdamW group lrs: embed 0.7, classifier 0.004, vector params 0.015
- shared schedule: 30% stable then 70% linear cooldown (cooldown_frac = 0.7)
- stock initialization restored: all projection weights zero, embed default normal
- `train_steps` 3290 -> 3214

Diff vs baseline: +84/-10 lines.

## Validation

8-seed mean val loss **3.278519** against the 3.27859 bar (margin 7.1e-05), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **GPT-5.6 Terra** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/openai-gpt-5-6-terra--codex--be5415b5ee66 (bundle: [`traces/events-openai-gpt-5-6-terra--codex--be5415b5ee66.json.gz`](../blob/add-sanitized-traces/traces/events-openai-gpt-5-6-terra--codex--be5415b5ee66.json.gz))
