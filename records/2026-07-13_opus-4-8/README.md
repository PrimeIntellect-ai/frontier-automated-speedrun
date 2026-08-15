# 3,018 steps — Opus 4.8

New record for the track-3 optimizer speedrun: **3,018 train_steps** to the validation bar, from the baseline's 3,290 (272 steps saved, 39% of the gap to the 2,600-step human record).

## Method

Splits the Muon learning rate by matrix role inside a single optimizer group: square attention projections step at 0.8x the base 0.025 (0.02) and rectangular MLP matrices at 1.4x (0.035), replacing the uniform rate. Muon weight decay rises from 0.05 to 0.06 and its momentum follows a schedule, warming from 0.85 to a 0.97 peak over the first 10% of steps, holding until 50%, then decaying to 0.90. AdamW beta2 is rescheduled from a constant 0.95 to a rising 0.97 to 0.99 ramp, and the Adam-side groups (embed, head, scalars) keep a floor of 0.1 on the learning-rate decay while the Muon blocks anneal to zero. Block initialization is decoupled by nonlinearity: attention value projections init at variance factor 1.0 and all other block matrices at 0.66, up from the uniform 0.33. The cooldown fraction extends from 0.7 to 0.8 with the linear decay shape retained. These changes cut train_steps from 3290 to 3018, with an 8-seed mean val loss of 3.278425 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- decoupled Muon lr: attn (square) x0.8 = 0.02 via ATTN_LR_SCALE
- decoupled Muon lr: mlp (rectangular) x1.4 = 0.035 via MLP_LR_SCALE
- Muon base lr 0.025, weight_decay 0.06
- init decoupled by nonlinearity: attn.v factor 1.0, others 0.66 (q/k reverted to 0.66)
- cooldown_frac 0.8
- linear LR decay (cosine/quadratic rejected)
- Adam LR floor 0.1, Muon blocks anneal to 0
- mu schedule: warmup 0.85->0.97 over first 10%, hold to 0.5, decay to 0.90
- beta2 rising schedule 0.97->0.99
- Adam beta1 0.8
- embed lr 0.7, head lr 0.004, scalars lr 0.015
- no leftover test markers
- `train_steps` 3290 -> 3018

Diff vs baseline: +35/-11 lines.

## Validation

8-seed mean val loss **3.278425** against the 3.27859 bar (margin 0.000165), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Opus 4.8** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/claude-opus-4-8--claude-code--42139b7480a2 (bundle: [`traces/events-claude-opus-4-8--claude-code--42139b7480a2.json.gz`](../blob/add-sanitized-traces/traces/events-claude-opus-4-8--claude-code--42139b7480a2.json.gz))
