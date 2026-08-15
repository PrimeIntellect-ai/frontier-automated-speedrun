# 3,120 steps — Grok 4.5

New record for the track-3 optimizer speedrun: **3,120 train_steps** to the validation bar, from the baseline's 3,290 (170 steps saved, 25% of the gap to the 2,600-step human record).

## Method

Swaps the baseline Newton-Schulz iteration (coefficients 2, -1.5, 0.5 for 12 steps) for the steeper quintic 3.5, -4.9, 2.1 run for 5 steps. Splits the Muon-optimized block matrices into attention and MLP groups with learning rates 0.021 and 0.029, and raises Muon weight decay to a 0.062 peak that anneals with the square of the cooldown eta down to a 0.15 floor. The linear cooldown keeps fraction 0.7 but floors the learning-rate multiplier at 0.02, and Adam beta2 for the embedding, projection, and scalar parameters rises from 0.95 to 0.99. With this stack the 8-trial mean is 3.27817 at train_steps = 3120, clearing the 3.27859 bar with 170 fewer steps than the 3290 baseline.

## Changes vs baseline `train_gpt_simple.py`

- Muon attn lr=0.021
- Muon mlp lr=0.029
- Muon weight decay peak 0.062 on both groups
- Newton-Schulz coeffs 3.5/-4.9/2.1
- ns_steps=5
- WD anneals with eta**2
- WD anneal floor 0.15
- LR floor 0.02
- cooldown_frac=0.7
- Adam betas (0.8, 0.99)
- Adam scalar lr 0.015
- embed lr 0.7 / proj lr 0.004 (stock)
- Muon mu=0.95 nesterov=True
- attn/mlp split covers all block matrices (assert present)
- `train_steps` 3290 -> 3120

Diff vs baseline: +39/-20 lines.

## Validation

8-seed mean val loss **3.27817** against the 3.27859 bar (margin 0.00042), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **Grok 4.5** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/x-ai-grok-4-5--grok-cli--fe260fe66649 (bundle: [`traces/events-x-ai-grok-4-5--grok-cli--fe260fe66649.json.gz`](../blob/add-sanitized-traces/traces/events-x-ai-grok-4-5--grok-cli--fe260fe66649.json.gz))
