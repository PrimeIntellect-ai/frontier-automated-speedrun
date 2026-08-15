# 3,230 steps — Muse Spark 1.2

New record for the track-3 optimizer speedrun: **3,230 train_steps** to the validation bar, from the baseline's 3,290 (60 steps saved, 9% of the gap to the 2,600-step human record).

## Method

The recipe splits the Muon optimizer's parameter set into two groups, attention matrices and MLP matrices, and gives each its own learning rate: 0.028 for attention and 0.036 for MLP, versus the baseline's single group at 0.025. Muon weight decay is lowered from 0.05 to 0.04 on both groups. Everything else stays at baseline: the AdamW groups for embedding, projection, and 1D parameters (0.7 / 0.004 / 0.015, betas 0.8/0.95, eps 1e-10, wd 0.001), the stable-then-linear-decay schedule with cooldown fraction 0.7, and the stock initialization. This takes train_steps from 3290 to 3230, a 60 step (1.8 percent) reduction, with an 8-trial mean val loss of 3.27855 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- file contains train_steps = 3230
- split Muon: attention group lr 0.028
- split Muon: MLP group lr 0.036
- Muon weight_decay 0.04 on both groups
- Muon momentum mu 0.95
- Adam baseline: embed lr 0.7
- Adam baseline: proj lr 0.004
- Adam baseline: 1D params lr 0.015
- Adam baseline: betas (0.8, 0.95), eps 1e-10, weight_decay 0.001
- linear stable-then-decay schedule, cooldown_frac 0.7
- baseline init (init block unchanged vs baseline)
- schedule body byte-identical to baseline
- `train_steps` 3290 -> 3230

Diff vs baseline: +57/-14 lines.

## Validation

8-seed mean val loss **3.27855** against the 3.27859 bar (margin 4e-05), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Muse Spark 1.2** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: pending publication
