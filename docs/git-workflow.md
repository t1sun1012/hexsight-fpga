# Milestone branches

Publish each completed, verified project step on its own GitHub branch. Use names of the form `step-NN-description`, without a tool or agent prefix.

## Current milestones

- `step-01-data-setup`: repository foundation, labeled dataset audit, and preprocessing.
- `step-02-cnn-baseline`: CNN training, comparison, verification, and associated reports (Step 2).
- `step-03-fixed-point`: integer CNN reference, FPGA parameter export, range proofs, and verification vectors (Step 3).
- `step-04-rtl-inference`: synthesizable CNN, stage-by-stage simulation, and standalone core resource/internal timing checks (Step 4).

Compare `step-01-data-setup` against main, `step-02-cnn-baseline` against `step-01-data-setup`, `step-03-fixed-point` against `step-02-cnn-baseline`, and `step-04-rtl-inference` against `step-03-fixed-point`.

The earlier MLP branch has been retired, and the CNN baseline branch has been renamed from Step 3 to Step 2. Its commits and comparison artifacts are preserved in the current CNN branch; it is no longer a branch to use as a review base.

## For subsequent steps

1. Create a descriptive branch such as `step-05-board-integration`.
2. Base it on main if the preceding milestone has been merged; otherwise base it on the preceding milestone branch.
3. Implement the step, run its relevant checks, and update its walkthrough with results and limitations.
4. Commit source code and small reproducible evidence. Exclude local environments, full datasets, machine settings, caches, and course handouts.
5. Push the branch and report its GitHub link, checks, and intended comparison base.
6. Review and merge milestones in dependency order when ready.

Do not force-push published milestone history as part of this workflow.
