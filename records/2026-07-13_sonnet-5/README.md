# 3,105 steps — Sonnet 5

New record for the track-3 optimizer speedrun: **3,105 train_steps** to the validation bar, from the baseline's 3,290 (185 steps saved, 27% of the gap to the 2,600-step human record).

## Method

Raises AdamW beta2 from 0.95 to 0.98 for the head and scalar parameter groups. Muon's weight decay is raised from 0.05 to 0.08 and reshaped in time: during cooldown the decay is scaled by eta cubed (wd_exponent=3.0), and during the flat-LR phase it is boosted by a constant 1.25x (flat_wd_boost=1.25), leaving the cooldown shape untouched. The embedding is moved out of the fused AdamW into a custom EmbedAdamW optimizer that keeps an fp32 master weight plus fp32 first and second moment state for the bf16-stored embedding, casting back to bf16 after each step, since stock AdamW state inherits bf16 and silently loses small second-moment increments. These five stacked changes cut train_steps from the 3290 baseline to 3105, a 5.62 percent reduction, with an 8-trial mean validation loss of 3.27826 against the 3.27859 bar (n=8 confirmed, p<0.001 per the run's verify.py).

## Changes vs baseline `train_gpt_simple.py`

- Muon flat_wd_boost=1.25
- flat-phase boost gated on eta >= 1.0, cooldown shape unchanged
- Muon wd_exponent=3.0 (WD scaled by eta**wd_exponent)
- Muon weight_decay=0.08
- Muon lr=0.025, mu=0.95 (stock)
- AdamW (head+scalar) betas=(0.8, 0.98)
- head lr=0.004, scalar lr=0.015, eps=1e-10, weight_decay=0.001 (stock)
- EmbedAdamW for embed, lr=0.7, betas=(0.8, 0.98), eps=1e-10, weight_decay=0.001
- EmbedAdamW keeps fp32 master weight and fp32 exp_avg/exp_avg_sq state
- EmbedAdamW second moment is full-rank (no A117 Adafactor row/col factoring)
- linear WSD schedule, cooldown_frac=0.7, no warmup (stock)
- diff vs frozen baseline = 58 changed lines, matching the agent's own git-diff verification of the RECORD 5 state
- `train_steps` 3290 -> 3105

Diff vs baseline: +49/-10 lines.

## Validation

8-seed mean val loss **3.27826** against the 3.27859 bar (margin 0.00033), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Sonnet 5** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/claude-sonnet-5--claude-code--6f4f51086c92 (bundle: [`traces/events-claude-sonnet-5--claude-code--6f4f51086c92.json.gz`](../blob/add-sanitized-traces/traces/events-claude-sonnet-5--claude-code--6f4f51086c92.json.gz))
