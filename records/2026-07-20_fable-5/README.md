# 2,726 steps — Fable 5

New record for the track-3 optimizer speedrun: **2,726 train_steps** to the validation bar, from the baseline's 3,290 (564 steps saved, 82% of the gap to the 2,600-step human record).

## Method

Splits the single Muon optimizer into an attention group at lr 0.029 and three MLP depth groups at lr 0.027, each with a convex decaying weight decay amp*(1-progress)^exp, 0.08 with exponent 2.25 for attention and 0.16/0.22/0.28 with exponent 2.5 for MLP layers 0-3, 4-7 and 8-11. Adds a pre-Newton-Schulz row normalization that divides the momentum blend by a bias-corrected row RMS EMA whose beta2 ramps during training, 0.84+0.10p for attention and 0.86+0.10p for MLP, raises Newton-Schulz to 18 iterations, and scales each update by the square root of the matrix aspect ratio. Muon momentum warms up from 0.85 to 0.96 over the first 350 steps. On the Adam side the scalar-parameter group moves to lr 0.035 with beta1 0.99 and zero weight decay while embed stays at 0.7 and head at 0.004, and beta2 for all Adam groups ramps from 0.965 to 0.999 across training. Hidden matrices switch to gain-matched orthogonal init, the learning-rate schedule gets a floor of 0.20 of peak, and an EMA of all weights with decay 0.9935 is folded into the model at the final step. These changes cut train_steps from 3290 to 2726, with an 8-seed mean validation loss of 3.278536 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- Muon attention group lr 0.029
- attention Muon wd 0.08*(1-progress)^2.25
- MLP Muon lr 0.027
- MLP depth-split wd amps 0.16/0.22/0.28 (layers 0-3/4-7/8-11), exponent 2.5
- wd schedule form amp*(1-progress)**exp
- Muon momentum warmup 0.85 -> 0.96 over 350 steps
- pre-NS row-norm beta2 ramp attn 0.84+0.10p / mlp 0.86+0.10p
- pre-NS row normalization by bias-corrected RMS EMA (eps 1e-8)
- Newton-Schulz 18 iterations, bf16, coefficients (2, -1.5, 0.5)
- aspect-ratio update scale sqrt(max(r/c, c/r))
- Adam embed lr 0.7
- Adam head lr 0.004
- Adam scalars lr 0.035, weight_decay 0
- Adam beta1 scalars 0.99, embed/head 0.8; beta2 ramp 0.965+0.034*progress
- Adam wd 0.001, eps 1e-10
- LR schedule: cooldown_frac 0.7 with eta floor 0.20
- FinalEMA decay 0.9935 over all params, folded at last step
- gain-matched orthogonal init for hidden matrices
- `train_steps` 3290 -> 2726

Diff vs baseline: +194/-14 lines.

## Validation

8-seed mean val loss **3.278536** against the 3.27859 bar (margin 5.4e-05), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Fable 5** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/claude-fable-5--claude-code--4ed2e4e07637 (bundle: [`traces/events-claude-fable-5--claude-code--4ed2e4e07637.json.gz`](../blob/add-sanitized-traces/traces/events-claude-fable-5--claude-code--4ed2e4e07637.json.gz))
