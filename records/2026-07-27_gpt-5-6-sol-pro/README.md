# 3,058 steps — GPT-5.6 Sol Pro

New record for the track-3 optimizer speedrun: **3,058 train_steps** to the validation bar, from the baseline's 3,290 (232 steps saved, 34% of the gap to the 2,600-step human record).

## Method

Muon's fixed row-scale factor is replaced by per-row variance EMA balancing of the orthogonalized update, with an additional column-variance EMA applied to tall matrices only. Muon momentum ramps per shape class from 0.85 to 0.965/0.970/0.965 over the first 600 steps, the peak learning rate rises to 0.026 with a dedicated 0.028 rate for tall expansion matrices, and the Muon cooldown decays as the schedule factor raised to the power 1.25. The vocabulary projection moves to a dedicated AdamW with beta1 0.80, a beta2 ramp from 0.95 to 0.99 over the first 400 steps, and exact product bias correction, and it is initialized semi-orthogonally at entry scale 0.20, variance-matched to the baseline Gaussian head. The learning rate schedule is decoupled from the training budget: 3058 training steps run against a 3135-step schedule horizon with a 70 percent linear cooldown. The 8-seed mean validation loss is 3.278406, 0.00018 below the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- schedule horizon 3135 with 70% cooldown (set_hparams cooldown_frac=0.70, schedule_steps=3135)
- projection beta2 ramp 0.95 to 0.99 over first 400 steps
- projection AdamW beta1 = 0.80
- exact varying-coefficient (product) beta2 bias correction for the projection optimizer
- semi-orthogonal vocabulary-head initialization with entry scale 0.20 (gain 0.20*sqrt(0.33*vocab/hidden))
- Muon momentum ramp 0.85 to 0.965/0.970/0.965 (square/tall/wide) over 600 steps
- Muon peak LR 0.026 with tall-class LR 0.028 and power-1.25 cooldown
- row-variance EMA balancing (beta2 0.80) of the polar update for all Muon matrices
- column-variance EMA balancing for tall Muon matrices only
- no late-stage momentum decrease retained (Experiment 117 rejected)
- `train_steps` 3290 -> 3058

Diff vs baseline: +97/-18 lines.

## Validation

8-seed mean val loss **3.278406** against the 3.27859 bar (margin 0.000184), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **GPT-5.6 Sol Pro** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/openai-gpt-5-6-sol-pro--codex--aeda186b4a8f (bundle: [`traces/events-openai-gpt-5-6-sol-pro--codex--aeda186b4a8f.json.gz`](../blob/add-sanitized-traces/traces/events-openai-gpt-5-6-sol-pro--codex--aeda186b4a8f.json.gz))
