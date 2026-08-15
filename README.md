# Frontier automated speedrun

Artifacts from an experiment measuring how well frontier models do research. We ran 18
frontier models autonomously on the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
track 3 optimizer speedrun: each model gets a GPU node (8xH200), the training repository,
a rulebook, and one goal message, then runs unattended for days. The full write-up is on the
[Prime Intellect blog](https://www.primeintellect.ai/blog/comparing-frontier-models-nanogpt-speedrun)
and the interactive results live at
[primeintellect.ai/research/nanogpt-speedrun](https://www.primeintellect.ai/research/nanogpt-speedrun).

## The task

The speedrun trains a 124M parameter GPT and counts how many training steps it takes to
reach validation loss 3.28. The baseline recipe in `train_gpt_simple.py` (Muon plus an
auxiliary AdamW, tuned) passes at **3,290 steps**. A record claim means training the
recipe eight times on fixed seeds the agent cannot touch and beating a mean val loss of
**3.27859**, a margin that makes passing on luck alone roughly one in a thousand. A frozen
`verify.py` checks every claim. The human record for reference sits at 2,600 steps.

## program.md

`program.md` is the rulebook every agent runs under: what can be edited, what counts as a
record, and how to use the node. The standard version tells agents to run each experiment
in a subagent and act on its report. `program-serial.md` is a variant used between July 20
and August 13 that made agents wait for each run serially instead; runs under it are
labeled in the results and are being rerun under the standard rulebook.

## traces/

Complete, sanitized agent trajectories for a curated set of runs, one triplet per run:

- `events-<id>.json.gz` — the full transcript: text, thinking, tool calls, tool results
- `subagents-<id>.json.gz` — child agent transcripts, where the harness has them
- `scratch-<id>.json.gz` — the run's scratchpad (decision logs, saved variants)
- `manifest.json.gz` — per-run metadata (model, harness, records, tooling stats)

Run ids are `<agent>--<harness>--<key>`. Node names, paths, and infrastructure
identifiers are redacted; the content of the model's work is untouched. The same bundles
power the trace viewer on the results site.

## Record PRs

Each open pull request is one model's best validated record, presented the way records
are submitted to the upstream nanogpt speedrun: the PR modifies `train_gpt_simple.py`
from the baseline to the exact record state recovered from the run's traces, adds an
entry under `records/`, and documents the method, the diff against the baseline, and the
8-seed validation. One PR per model, last record only.

## Leaderboard

| # | Record | Description | Date | Log |
|---|--------|-------------|------|-----|
| 1 | 3,290 steps | Baseline (Muon + aux AdamW, tuned) | 2026-07-08 | — |
| . | 3,205 steps | DeepSeek V4 Pro | 2026-08-13 | [record](records/2026-08-13_deepseek-v4-pro/README.md) |
