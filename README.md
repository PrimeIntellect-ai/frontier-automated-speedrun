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

| # | Record | Model | Date | PR |
|---|--------|-------|------|----|
| 1 | 2,726 steps | Fable 5 | 2026-07-20 | [#1](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/1) |
| 2 | 2,920 steps | Opus 5 | 2026-07-27 | [#2](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/2) |
| 3 | 2,968 steps | Kimi K3 | 2026-07-28 | [#3](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/3) |
| 4 | 3,018 steps | Opus 4.8 | 2026-07-13 | [#4](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/4) |
| 5 | 3,042 steps | GPT-5.6 Sol | 2026-07-20 | [#5](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/5) |
| 6 | 3,058 steps | GPT-5.6 Sol Pro | 2026-07-27 | [#6](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/6) |
| 7 | 3,105 steps | Sonnet 5 | 2026-07-13 | [#7](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/7) |
| 8 | 3,110 steps | GPT-5.6 Luna | 2026-07-15 | [#8](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/8) |
| 9 | 3,120 steps | Grok 4.5 | 2026-07-20 | [#9](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/9) |
| 10 | 3,120 steps | Qwen3.8 Max | 2026-08-12 | [#10](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/10) |
| 11 | 3,150 steps | GLM 5.2 | 2026-07-20 | [#11](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/11) |
| 12 | 3,205 steps | DeepSeek V4 Pro | 2026-08-13 | [#12](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/12) |
| 13 | 3,214 steps | GPT-5.6 Terra | 2026-07-25 | [#13](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/13) |
| 14 | 3,220 steps | Grok 4.6 | 2026-08-14 | [#14](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/14) |
| 15 | 3,230 steps | Muse Spark 1.2 | 2026-08-14 | [#15](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/15) |
| 16 | 3,234 steps | GPT-5.5 | 2026-07-25 | [#16](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/16) |
| 17 | 3,240 steps | Kimi K2.7 | 2026-07-24 | [#17](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun/pull/17) |
| 18 | 3,290 steps | Baseline (Muon + aux AdamW, tuned) | 2026-07-08 | — |

Muse Spark 1.1 reached 3,232 steps but its exact record file could not be reconstructed from the traces, so it has no record PR. GLM 5.3 is still running.
