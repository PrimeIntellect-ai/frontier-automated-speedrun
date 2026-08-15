# 3,042 steps — GPT-5.6 Sol

New record for the track-3 optimizer speedrun: **3,042 train_steps** to the validation bar, from the baseline's 3,290 (248 steps saved, 36% of the gap to the 2,600-step human record).

## Method

Replaces Muon's exact polar Newton-Schulz iteration (2, -1.5, 0.5) x12 with a high-slope quintic (3.4445, -4.7750, 2.0315) x5 and scales updates by the symmetric aspect ratio sqrt(max(m,n)/min(m,n)). Adds gradient preconditioning inside Muon: rectangular MLP matrices are row-RMS normalized (EMA beta2 0.95) and then weakly column-corrected (fourth-root on wide outputs, eighth-root on tall inputs), while square attention matrices get a weak row-variance correction (power 0.0625, EMA beta2 0.99); momentum buffers warm-start from the first gradient. Couples several quantities to the linear LR cooldown over the final 70% of training: Muon momentum decays quadratically 0.95 to 0.90 via persistent device scalars, preconditioner beta2 decays quadratically 0.95 to 0.93, and Muon weight decay relaxes quadratically from 0.05 toward 0.03 with per-role multipliers (rectangular MLP 1.6 to 1.2 quadratic, attention 0.90 to 0.80 linear, plus a constant +0.15 offset for q/k matrices). AdamW uses embedding-specific betas (0.81, 0.98) with (0.80, 0.98) elsewhere. train_steps drops from 3290 to 3042, validated as an 8-seed mean of 3.27858625 against the 3.27859 bar.

## Changes vs baseline `train_gpt_simple.py`

- high-slope quintic Newton-Schulz coefficients (3.4445, -4.7750, 2.0315)
- five Newton-Schulz iterations (E62 pruning: retain five)
- symmetric aspect-ratio update scaling sqrt(max(m,n)/min(m,n)) (E11)
- row RMS preconditioning on rectangular MLP matrices (E15/E25)
- residual column correction: wide power 0.25 / tall power 0.125 (E57)
- square-attention row-variance correction power 0.0625 (E117)
- attention preconditioner beta2 0.99 (E114-E126 pruning)
- Muon momentum warm start from first gradient (E192/E195)
- quadratic Muon momentum cooldown 0.95 -> 0.90, tensorized mu (E64/E76)
- quadratic row/column preconditioner beta2 cooldown endpoint 0.93 (E205-E212 pruning)
- quadratic Muon weight-decay relaxation 0.05 -> 0.03 (E205-E212 pruning)
- MLP decay multiplier 1.6 -> 1.2 quadratic (E265-E272 pruning)
- linear attention decay multiplier 0.90 -> 0.80 (E262/E268)
- constant q/k decay offset +0.15 (E311/E312)
- Muon lr 0.025, weight decay 0.05, mu 0.95
- embedding AdamW lr 0.7 with betas (0.81, 0.98) (E174-E183 pruning)
- other AdamW groups betas (0.80, 0.98), eps 1e-10, weight decay 0.001
- linear LR cooldown over final 70% (cooldown_frac=0.7)
- `train_steps` 3290 -> 3042

Diff vs baseline: +73/-16 lines.

## Validation

8-seed mean val loss **3.27858625** against the 3.27859 bar (margin 4e-06), fixed seeds 0xC0FFEE+0..7, checked by the frozen `verify.py` (statistical rule matching the upstream repo). Recipe and record produced autonomously by **GPT-5.6 Sol** during the frontier speedrun experiment; the training script in this PR is the exact record state recovered from the run's traces.

Full agent trace: https://www.primeintellect.ai/research/nanogpt-speedrun/traces/openai-gpt-5-6-sol--codex--044f97fbcd18 (bundle: [`traces/events-openai-gpt-5-6-sol--codex--044f97fbcd18.json.gz`](../blob/add-sanitized-traces/traces/events-openai-gpt-5-6-sol--codex--044f97fbcd18.json.gz))
