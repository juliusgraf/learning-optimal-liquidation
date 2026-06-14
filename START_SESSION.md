This project is complete: all eight revision phases are implemented. We are now
in maintenance/extension mode. Before doing any work, load context from disk and
orient yourself — do NOT edit anything yet.

1. Confirm you have CLAUDE.md loaded; treat its binding conventions as
   non-negotiable for everything that follows. In particular, any change that
   touches the market model or RL must preserve: the corrected Eqs. (1)-(2) with
   the -zeta*nu sign and N^+ = buys; the H_cl timing (auction estimate computed
   at end of t-1 and cached, D1; Algorithm 1 update at end of CLOB step t feeding
   t+1, D2); the predictable theta recursion; the time-free admissibility set
   Adm(x) with C(x) = max_i (1 - x^{9,(i)}) * 1{x^{17,(i)} > 0}; the paper's
   reward forms (f_c with no clamp, f_a, d_t * c_t); no terminal inventory
   clipping; and the Richard et al. rough-Heston scheme.

2. Read these orientation files (they encode the project's invariants and
   decisions): audit/AUDIT.md, audit/PARAMS_FROM_CODE.md, audit/BEHAVIOR_CHANGES.md,
   docs/rl_design.md, docs/continuous_action_extension.md, docs/metrics_schema.md,
   and REPRODUCING.md.

3. Build a structural map WITHOUT reading every file: view the src/lmm/ tree and
   the tests/ tree, and skim public interfaces (class/function signatures and
   docstrings) in market/, env/, agents/, and experiments/. Use grep/targeted
   reads, not full-file reads, to keep context lean.

4. Establish the baseline: run `pytest -m "not slow"` and report green/red; run
   `git log -5 --oneline` and `git status` so we know the current commit and
   whether the tree is clean.

5. Give me a <= 20-line orientation summary: the package layout, where the
   convention tests live (timing, sign, rewards, clearing, admissibility),
   how to launch a training run and regenerate figures/tables, and any failing
   tests or uncommitted changes.

Then STOP and ask me what to work on. When I give you a task: if it touches the
math or RL conventions, the existing convention tests must stay green AND you add
a test covering the new behavior; keep experiment parameters in configs (never
hard-coded); never edit anything under paper/; and preserve the seeding
discipline (one Generator per component from the master seed; no global RNG).