# 3,220 steps — Grok 4.6

New record for the track-3 optimizer speedrun: **3,220 train_steps** to the validation bar, from the baseline's 3,290 (70 steps saved, 10% of the gap to the 2,600-step human record).

## Method

Replaces the 12-iteration Newton-Schulz orthogonalization inside Muon with the Polar Express quintic iteration, running 5 iterations with coefficients (3.4445, -4.7750, 2.0315) for a much closer polar-factor approximation at lower cost. Anneals weight decay together with the learning rate: each optimizer group stores its initial weight decay and scales it by the same eta multiplier as the LR, so decay fades out through the cooldown instead of shrinking weights at full strength to the end. Everything else stays stock: AdamW on embed, proj and 1d params (lr 0.7/0.004/0.015, betas 0.8/0.95, wd 0.001), Muon on 2d block params (lr 0.025, wd 0.05, momentum 0.95, Nesterov), constant LR then a linear-to-zero cooldown over the last 70% of steps, and unchanged init. This cuts train_steps from 3290 to 3220, with an 8-trial mean val loss of 3.277827 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- Polar Express NS coefficients (3.4445, -4.7750, 2.0315)
- 5 NS iterations (baseline 12)
- no 1.01 safety factor on Frobenius normalization
- linear cooldown over last 70% (cooldown_frac=0.7, eta linear to zero)
- weight decay annealed with eta
- stock AdamW lrs 0.7/0.004/0.015, betas (0.8,0.95), wd 0.001
- stock Muon lr 0.025 wd 0.05 mu 0.95 nesterov
- `train_steps` 3290 -> 3220

Diff vs baseline: +8/-7 lines.

## Validation

8-seed mean val loss **3.277827** against the 3.27859 bar (margin 0.000763), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Grok 4.6** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/x-ai-grok-4-6--grok-cli--c7bc80b0cbc8 (bundle: [`traces/events-x-ai-grok-4-6--grok-cli--c7bc80b0cbc8.json.gz`](../blob/add-sanitized-traces/traces/events-x-ai-grok-4-6--grok-cli--c7bc80b0cbc8.json.gz))
