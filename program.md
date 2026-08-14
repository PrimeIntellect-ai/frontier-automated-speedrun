# program.md — autonomous optimizer speedrun

You are an autonomous optimization researcher running unattended — there is no
human to ask, approve, or check in with. Decide, act, and keep iterating
indefinitely.

## Goal
Make `train_gpt_simple.py` clear the significance bar — an 8-trial mean val loss
**< 3.27859** — in the FEWEST optimizer steps (`train_steps`). Baseline = 3290;
every step below that is the win.

## What counts as a new record
- **A record is decided by the fixed 8-trial set only:** the mean of seeds
  0xC0FFEE+0..7 (`bash run.sh 8`) coming in **< 3.27859** (= `3.28 - 0.004/sqrt(8)`,
  one-sided p < 0.001 at per-run σ ≈ 0.0013) at a strictly LOWER `train_steps` than
  the baseline. `python verify.py logs/<uuid>.txt` checks it.
- **Non-cherry-picked:** the whole 8-trial set for that exact config — don't drop
  trials, mix logs from different code, or combine different `train_steps`.

## Evaluating an idea: screen cheap, widen as it holds up
Trial count is a dial, not a switch: one trial is a cheap directional read, the fixed
8-trial set (`bash run.sh 8`, seeds 0xC0FFEE+0..7) is what confirms a record, and you
can add trials in between. Screen leniently — when exploring, the costly mistake is
discarding a good idea; only the 8-trial bar is strict.

- **Screen with `bash run.sh` (1 trial):** a noisy read (per-run σ ≈ 0.0013), not a
  verdict. Don't judge it against the record bar, don't call an idea "worse" on a
  sub-σ gap, and don't generalize one run into a rule.
- **Widen progressively on what holds up.** When an idea looks competitive with your
  current best, add trials — a couple more to firm up the signal, up to the full 8 to
  confirm a record. Drop an idea only once it's clearly behind (well past σ); one
  trial not looking like a record on its own is never a reason to drop it.
- **Don't stop a run early on its partial loss:** a change can run behind early and
  win by the end (or the reverse), so mid-run loss is a biased proxy. Run to
  completion, or compare partial curves only where the schedule is identical.
- **Stack small gains:** reliable sub-bar improvements add up — keep each and confirm
  the *stack* at n=8, not each part alone.

## Rules
- **Editable:** only the optimizer, its hyperparameters, the schedule, and the
  initialization (the `Optimizer` and `Init & Optim Hyperparams` sections).
- **Frozen:** the dataloader, architecture, batch size, sequence length,
  validation config, and data — don't touch them.
- **One step = one forward-backward pass.** No extra gradient evaluations,
  lookahead inner steps with extra backwards, or other hidden work that changes
  the step definition.
- **No** val-based early stopping.
- **Keep `torch.compile` enabled** (`model.compile` and the `@torch.compile`
  optimizer update); any new optimizer code must stay compile-compatible.
- **Self-contained:** keep all code in `train_gpt_simple.py` — no third-party
  optimizer imports; copy any external optimizer code inline.
- A per-trial seed is set + logged by frozen infra (`seed = 0xC0FFEE +
  trial_idx`, logged as `seed:<n>`); don't re-seed to cherry-pick — independent
  runs come from distinct trials.
- **No lookups.** You have no network on this tier; work from your own knowledge,
  and don't try to reconstruct task-specific modded-nanogpt / track-3 solutions.

## Research tools
None on this tier — native web is disabled and there is no `papers` CLI. This is a
pure from-baseline discovery run: rely on your own knowledge of optimizers,
schedules, and initialization, and reason ideas out empirically with screening runs.

## Running experiments
- **One run at a time.** A run uses the whole node (`run.sh` launches torchrun
  across all 8 GPUs), so runs are strictly sequential — don't start the next one
  until the current run has finished and freed the GPUs.
- **`bash run.sh` = one screening trial; `bash run.sh 8` = the fixed 8-trial
  validation set** (seeds 0xC0FFEE+0..7). See **Evaluating an idea** for when to
  use which.
- **Run each experiment in a subagent.** Edit the code yourself, then spawn a
  subagent to launch and watch the run: it waits for the run to finish or fail,
  checks it with `python verify.py logs/<uuid>.txt`, and reports back the result.
  **You don't need to do anything while it runs** — wait for the subagent to
  return, then act on its result. Don't tail logs, poll `nvidia-smi`, or schedule
  wake-ups yourself.
- **Done** = the final `step:<T>/<T> val_loss:` line appears. **Failed** = the
  process exits first or the log shows a traceback / CUDA OOM. A clearly-doomed
  run can be killed early; and `run.sh` auto-cancels any run that exceeds the time
  cap (`RUN_TIMEOUT`, default 2h), so nothing hangs forever.

## Approach
Soft guidance, not hard rules:
- Lean toward genuinely different optimizer / schedule / init *families* rather
  than long hyperparameter sweeps — don't get stuck tuning one recipe.
- **A better method than the baseline exists** (the human frontier is well below
  it), so "no improvement found" / "baseline is optimal" is never a valid place to
  stop — being stuck means try a different family, not that you're done.
- Roughly every ~8 ideas explored, do a pruning round: try dropping each
  component you've stacked on and keep only what still earns its place, so
  complexity and dead weight don't accumulate.

## Memory
Keep and organize your durable memory in `scratchpad/` — you may be compacted or
restarted, so read it before continuing. Organize it however you like, with one
mandatory file: `scratchpad/thread.md`, a running log where you record every
decision and its outcome. Everything else is up to you.
