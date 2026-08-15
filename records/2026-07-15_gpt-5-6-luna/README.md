# 3,110 steps — GPT-5.6 Luna

New record for the track-3 optimizer speedrun: **3,110 train_steps** to the validation bar, from the baseline's 3,290 (180 steps saved, 26% of the gap to the 2,600-step human record).

## Method

Adds a dual RMS-balancing stage to the Muon update: before Newton-Schulz orthogonalization, each update matrix is rescaled row-wise to equalize row RMS against the global RMS, then column-wise the same way, in that order. The Muon parameter groups get their own steeper decay schedule, applying the shared cooldown factor with power 1.2 while the AdamW groups keep the linear schedule. The learning-rate schedule is anchored to an absolute horizon of 3140 steps instead of the actual step count, so the run stops at 3110 steps before the schedule reaches zero. The output projection AdamW learning rate is lowered from 0.004 to 0.003. All other baseline settings are unchanged: Muon lr 0.025 with weight decay 0.05 and mu 0.95, cooldown fraction 0.7, 12 bfloat16 Newton-Schulz iterations, and stock initialization. Validated at 3110 steps with a fixed-seed 8-trial mean of 3.27841 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- schedule horizon 3140 (progress = step / 3140)
- cooldown_frac = 0.7
- Muon-only schedule power 1.2 on optimizer2
- schedule applies eta ** schedule_power (default 1.0)
- output AdamW lr 0.003
- Muon lr 0.025, weight_decay 0.05, mu 0.95
- pre-NS row RMS balance (row-first)
- then column RMS balance
- row balance before col balance before Newton-Schulz
- 12 Newton-Schulz iterations
- bfloat16 NS path (no float32 switch)
- Nesterov momentum enabled (stock)
- zero-init residual projections (stock)
- stock non-projection init variance 0.33
- stock AdamW betas (0.8, 0.95)
- stock embed lr 0.7 / scalar lr 0.015
- `train_steps` 3290 -> 3110

Diff vs baseline: +14/-6 lines.

## Validation

8-seed mean val loss **3.27841** against the 3.27859 bar (margin 0.00018), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **GPT-5.6 Luna** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/openai-gpt-5-6-luna--codex--637c866472d1 (bundle: [`traces/events-openai-gpt-5-6-luna--codex--637c866472d1.json.gz`](../blob/add-sanitized-traces/traces/events-openai-gpt-5-6-luna--codex--637c866472d1.json.gz))
