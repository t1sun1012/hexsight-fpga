# Milestone branches

Publish each completed, verified project step on its own GitHub branch. Use names of the form `step-NN-description`, without a tool or agent prefix.

## Current milestones

- `step-01-data-setup`: repository foundation, labeled dataset audit, and preprocessing.
- `step-02-training-baseline`: CPU MLP training, saved checkpoint, validation reports, and verification.
- `step-03-cnn-baseline`: CNN training, validation comparison, and updated architecture.

Step 2 is based on Step 1. Compare Step 1 against main, and Step 2 against the Step 1 branch to review only the new training work. Step 3 builds on Step 2; compare against the Step 2 branch to review the CNN changes.

## For subsequent steps

1. Create a descriptive branch such as `step-04-fixed-point`.
2. Base it on main if the preceding milestone has been merged; otherwise base it on the preceding milestone branch.
3. Implement the step, run its relevant checks, and update its walkthrough with results and limitations.
4. Commit source code and small reproducible evidence. Exclude local environments, full datasets, machine settings, caches, and course handouts.
5. Push the branch and report its GitHub link, checks, and intended comparison base.
6. Review and merge milestones in dependency order when ready.

Do not force-push published milestone history as part of this workflow.
