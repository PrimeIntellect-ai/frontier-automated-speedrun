# 3,240 steps — Kimi K2.7

New record for the track-3 optimizer speedrun: **3,240 train_steps** to the validation bar, from the baseline's 3,290 (50 steps saved, 7% of the gap to the 2,600-step human record).

## Method

Cuts training from 3290 to 3240 steps. The Muon orthogonalization runs 21 Newton-Schulz iterations instead of 12, keeping the same (2, -1.5, 0.5) coefficients, for a tighter orthonormalization of each update. Hidden matrix weights are initialized with std 0.55**0.5/sqrt(fan_in) instead of 0.33**0.5/sqrt(fan_in). The embedding AdamW learning rate is raised from 0.7 to 1.0; the projection and non-2D parameter learning rates stay at baseline values, as do the Muon settings (lr 0.025, mu 0.95, weight decay 0.05). The learning rate schedule keeps the baseline shape, constant then a linear cooldown over the final 70 percent of steps; the schedule function gains optional warmup and floor parameters but both are set to 0.0, so they are inactive at the record configuration. The 8-trial mean validation loss is 3.278548, clearing the 3.27859 bar by 0.000042.

## Changes vs baseline `train_gpt_simple.py`

- Newton-Schulz iterations = 21 (for _ in range(21))
- cooldown_frac = 0.7 (set_hparams(step, cooldown_frac=0.7), default also 0.7)
- init 0.55 (init_std_constant = 0.55)
- no schedule floor (eta_min=0.0) and no warmup (warmup_frac=0.0)
- 8-trial mean 3.27855 (thread.md losses average to 3.2785475; viz-frontier validated entry 3.278548, source verify-trials:267a8b88)
- `train_steps` 3290 -> 3240

Diff vs baseline: +15/-10 lines.

## Validation

8-seed mean val loss **3.278548** against the 3.27859 bar (margin 4.2e-05), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Kimi K2.7** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/moonshotai-kimi-k2-7-code--kimi-code--fdbd0576f9d9 (bundle: `traces/events-moonshotai-kimi-k2-7-code--kimi-code--fdbd0576f9d9.json.gz`)
