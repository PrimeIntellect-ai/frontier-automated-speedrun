# 3,234 steps — GPT-5.5

New record for the track-3 optimizer speedrun: **3,234 train_steps** to the validation bar, from the baseline's 3,290 (56 steps saved, 8% of the gap to the 2,600-step human record).

## Method

Replaces the Newton-Schulz orthogonalization inside Muon: instead of 12 iterations with the exact-ish cubic coefficients (2, -1.5, 0.5), it runs 5 iterations of the optimized quintic polynomial (3.4445, -4.7750, 2.0315), which gives larger useful update components late in training. Raises the AdamW second-moment beta from 0.95 to 0.985 for the embedding, output-projection, and scalar parameter groups, smoothing the Adam-side preconditioner. All other hyperparameters stay stock: cooldown fraction 0.7, Muon lr 0.025 with weight decay 0.05 and Nesterov momentum 0.95, AdamW lrs 0.7/0.004/0.015 with beta1 0.8 and weight decay 0.001. With these two changes the 8-trial mean clears the 3.27859 bar at 3234 steps, 56 fewer than the 3290 baseline.

## Changes vs baseline `train_gpt_simple.py`

- Muon Newton-Schulz uses 5 iterations (was 12)
- Muon NS quintic coefficients a,b,c = 3.4445, -4.7750, 2.0315 (was 2, -1.5, 0.5)
- AdamW betas = (0.8, 0.985) (beta2 0.985, was 0.95)
- cooldown_frac = 0.7 (stock schedule retained)
- Muon lr = 0.025 (stock)
- Muon weight_decay = 0.05 (stock)
- Muon momentum mu = 0.95 with Nesterov (stock)
- embedding AdamW lr = 0.7 (stock)
- output projection AdamW lr = 0.004 (stock)
- scalar/bias/norm AdamW lr = 0.015 (stock)
- AdamW weight_decay = 0.001 and eps 1e-10 (stock)
- `train_steps` 3290 -> 3234

Diff vs baseline: +4/-5 lines.

## Validation

8-seed mean val loss **3.278452** against the 3.27859 bar (margin 0.000138), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **GPT-5.5** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/openai-gpt-5-5--codex--0c137e90a94d (bundle: `traces/events-openai-gpt-5-5--codex--0c137e90a94d.json.gz`)
