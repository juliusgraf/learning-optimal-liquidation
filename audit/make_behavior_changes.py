"""Generate audit/BEHAVIOR_CHANGES.md: legacy vs new-env summary statistics
under the Phase-1 characterization policy, each difference attributed to a
numbered ruling (Phase 3 deliverable).

Runs, for seeds 101/202/303 and both settings:
- the LEGACY emulators (legacy/main.py exec-truncated, legacy/data.py
  imported; same loading technique as tests/audit/test_legacy_characterization.py)
  under the fixed policy pinned there (CLOB (v=5, delta=2); auction (K=2,
  S = S_mid + 2*alpha); cancel variant at t=135);
- the NEW MarketMakingEnv under the mirrored policy (volume capped at
  inventory — the new env rejects rather than silently clamps, AUDIT N12;
  auction quote as the tick offset +2, AUDIT N4).

Usage:  python3 audit/make_behavior_changes.py   (writes BEHAVIOR_CHANGES.md
next to this script; deterministic given the seeds).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / "legacy"
OUT_PATH = Path(__file__).resolve().parent / "BEHAVIOR_CHANGES.md"

sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

# torch.set_num_interop_threads may only be called once per process (both
# legacy modules call it at import); see tests/audit for the same guard.
import torch  # noqa: E402

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_num_interop_threads = lambda n: None
torch.set_num_threads(1)

import numpy as np  # noqa: E402

from lmm.config import load_config  # noqa: E402
from lmm.env.action_spaces import AuctionAction, ClobAction  # noqa: E402
from lmm.env.mdp import make_env  # noqa: E402

SEEDS = (101, 202, 303)
CANCEL_AT = 135


# --------------------------------------------------------------------------
# Legacy emulators (loading per tests/audit/test_legacy_characterization.py)
# --------------------------------------------------------------------------

def load_legacy_main():
    src = (LEGACY_DIR / "main.py").read_text()
    cut = src.index("\nSEED = 42")
    mod = types.ModuleType("legacy_main_truncated")
    mod.__file__ = str(LEGACY_DIR / "main.py")
    sys.modules[mod.__name__] = mod
    code = compile(src[:cut], "main.py[truncated]", "exec", flags=0, dont_inherit=True)
    exec(code, mod.__dict__)
    return mod.__dict__


def load_legacy_data():
    spec = importlib.util.spec_from_file_location("legacy_data", LEGACY_DIR / "data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cat_mid_path():
    import pandas as pd

    df = (
        pd.read_csv(LEGACY_DIR / "data.csv", parse_dates=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    return df["CAT"].to_numpy(dtype=float)[0:120]


def make_legacy_synthetic(ns):
    return ns["MarketEmulator"](
        tau_op=120, tau_cl=150, T=150, I=100, V=30, L=12, Lc=12, La=12,
        L_max_auction=100, lambda_param=0.5, kappa=0.1, q=1.0, d=0.1,
        gamma=0.95, v_m=2.0, pareto_gamma=2.5, poisson_rate=1.0, alpha=0.01,
        seed=None, V_top_max=15.0, beta_a=2.0, beta_b=5.0, depth_decay=0.5,
        price_band_ticks=10.0, mid_rh_H=0.1, mid_rh_v0=0.02, mid_rh_theta=0.04,
        mid_rh_kappa=0.3, mid_rh_xi=0.3, mid_rh_rho=-0.7, sigma_mid=0.1,
    )


def make_legacy_historical(mod, mid_path):
    return mod.MarketEmulator(
        tau_op=120, tau_cl=150, T=150, I=100, V=30, L=12, Lc=12, La=12,
        L_max_auction=100, lambda_param=0.5, kappa=0.1, q=1.0, d=0.1,
        gamma=0.99,  # legacy bug: RL discount in the smoothing slot (D15/N2)
        v_m=2.0, pareto_gamma=2.5, poisson_rate=60.0, alpha=0.01, seed=None,
        V_top_max=15.0, beta_a=2.0, beta_b=5.0, depth_decay=0.5,
        price_band_ticks=10.0, mid_price_path=mid_path,
    )


def run_legacy_episode(env, seed, cancel_at=None):
    s = env.reset(seed=seed)
    done, total_r, n_steps = False, 0.0, 0
    h_traj = [s["X3"]]
    while not done:
        if env.phase == "continuous":
            s, r, done = env.step((5.0, 2.0))
        else:
            t = int(s["time"])
            c_vec = [0] * (env.tau_cl + 1)
            if cancel_at is not None and t == cancel_at:
                c_vec[env.tau_op:t] = [1] * (t - env.tau_op)
            s, r, done = env.step((2.0, s["X10"] + 2 * env.alpha, c_vec))
        total_r += r
        n_steps += 1
        h_traj.append(s["X3"])
    return {
        "decisions": n_steps,
        "total_reward": float(total_r),
        "final_inventory": float(env.inventory),
        "h_final": float(h_traj[-1]),
        "h_mean": float(np.mean(h_traj)),
    }


# --------------------------------------------------------------------------
# New env, mirrored policy
# --------------------------------------------------------------------------

def run_new_episode(cfg, seed, cancel_at=None, symbol=None):
    env = make_env(cfg, symbol=symbol, repo_root=REPO_ROOT)
    env.reset(seed=seed)
    total_r, n_steps = 0.0, 0
    h_traj = [env.h_cl]
    info = {}
    while True:
        if env.phase == "clob":
            a = ClobAction(min(5.0, env.inventory), 2)
        else:
            cancel = int(
                cancel_at is not None
                and int(env.t) == cancel_at
                and env._ledger.cancel_admissible()
            )
            a = AuctionAction(2.0, 2, cancel)
        _, r, term, _, info = env.step(a)
        total_r += r
        n_steps += 1
        h_traj.append(info["H_next"])
        if term:
            break
    return {
        "decisions": n_steps,
        "total_reward": float(total_r),
        "final_inventory": float(env.inventory),
        "h_final": float(info["S_cl"]),  # = j = m+1 estimate (Eq. (1))
        "s_cl": float(info["S_cl"]),
        "h_mean": float(np.mean(h_traj)),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

COLUMNS = ("decisions", "total_reward", "final_inventory", "h_final", "h_mean")


def fmt_row(label, stats):
    return (
        f"| {label} | {stats['decisions']} | {stats['total_reward']:.4f} "
        f"| {stats['final_inventory']:.4f} | {stats['h_final']:.4f} "
        f"| {stats['h_mean']:.4f} |"
    )


def table(rows):
    head = (
        "| run | decisions | total reward | final inventory | final H (=S_cl new) | H mean |\n"
        "|---|---|---|---|---|---|"
    )
    return "\n".join([head, *rows])


def main() -> None:
    legacy_main = load_legacy_main()
    legacy_data = load_legacy_data()
    cat = cat_mid_path()

    cfg_syn = load_config(REPO_ROOT / "configs/base.yaml",
                          REPO_ROOT / "configs/synthetic_rough_heston.yaml")
    cfg_hist = load_config(REPO_ROOT / "configs/base.yaml",
                           REPO_ROOT / "configs/historical_sp500.yaml")

    syn_rows, hist_rows = [], []
    for seed in SEEDS:
        for cancel_at, tag in ((None, ""), (CANCEL_AT, " +cancel@135")):
            if cancel_at and seed != SEEDS[0]:
                continue  # cancel variant for the first seed only (as Phase 1)
            leg = run_legacy_episode(make_legacy_synthetic(legacy_main), seed, cancel_at)
            new = run_new_episode(cfg_syn, seed, cancel_at)
            syn_rows.append(fmt_row(f"legacy seed {seed}{tag}", leg))
            syn_rows.append(fmt_row(f"new    seed {seed}{tag}", new))
            leg = run_legacy_episode(
                make_legacy_historical(legacy_data, cat), seed, cancel_at
            )
            new = run_new_episode(cfg_hist, seed, cancel_at, symbol="CAT")
            hist_rows.append(fmt_row(f"legacy seed {seed}{tag}", leg))
            hist_rows.append(fmt_row(f"new    seed {seed}{tag}", new))

    body = f"""# Behavior changes: legacy emulators vs the Phase-3 `MarketMakingEnv`

Generated by `python3 audit/make_behavior_changes.py` (deterministic; seeds
{SEEDS}, fixed characterization policy of Phase 1 — CLOB (v=5, delta=2),
auction (K=2, quote two ticks above the frozen mid), cancel variant at
t={CANCEL_AT}). Legacy goldens are pinned bit-exactly in
`tests/audit/test_legacy_characterization.py`; the new env is pinned by
`tests/test_determinism.py`.

**Read this first:** the two systems use different RNG architectures by
design (ruling D10: per-component `numpy` Generators from a SeedSequence vs
legacy's globally re-seeded MT19937), and ruling D7 changes the number of
draws per auction step, so for a given seed the realized exogenous paths are
DIFFERENT. Per-seed values are therefore **not pointwise comparable**;
compare structural statistics (decision counts, magnitudes, signs) and
attribute systematic differences with the table at the bottom.

## Synthetic (rough Heston), per-seed summary

{table(syn_rows)}

## Historical (CAT path), per-seed summary

{table(hist_rows)}

## Attribution of differences (each maps to a binding ruling)

| # | Observable difference | Cause / ruling |
|---|---|---|
| 1 | Historical decision count 149 -> **150** (120 CLOB + 30 auction); synthetic counts shift similarly (the grid now always reaches t_n) | **AUDIT N1 fix**: legacy never acted at t_n = tau_op - 1; the new grid caps hat_t at n and takes the final CLOB decision there |
| 2 | Reward at each step uses the END-of-previous-step H_cl; legacy's auction estimate included same-step flow and the same-step cancel | **D1** (auction Eq. (2) cache) and **D2** (Algorithm 1 end-of-step output is H_{{t+1}}); pinned by tests/test_timing_conventions.py |
| 3 | CLOB rewards can exceed legacy's at quotes above H_cl (multiplier > 1 no longer clamped) | **D5**: paper-exact f_c; the legacy clamp at 1 (main.py:545) was wrong |
| 4 | Cancel cost is d_t = (t - n - 1)d once per cancel-all; legacy charged d per c-vector entry (numerically equal under its own expansion, but vacuous cancels are now inadmissible) | **D4** + the C(x) admissibility constraint; cancel variants differ when a cancel-all would have been vacuous |
| 5 | Different per-step draw counts in the auction (one Bernoulli(0.05) per side instead of legacy's two-draw construction); buy side drawn before sell | **D7** (same law, fewer draws) and **D3** (paper side-naming; pure stream relabeling) |
| 6 | Quotes (agent and exogenous MM) snap to the tick grid alpha*N anchored at floor(S_mid/alpha); legacy added offsets to the raw float mid | **AUDIT N4** resolution (paper: S^a in alpha*N); sub-tick differences in S^a and S^i |
| 7 | Historical H_cl smoothing uses tau = 0.95; legacy historical accidentally used 0.99 (the RL discount, passed into the smoothing slot at data.py:1391) | **D15** (Q1 ruling): smoothing 0.95 in BOTH settings; chi = 0.99 reserved for the RL objective |
| 8 | Degenerate clearing (zero aggregate slope) falls back to the FROZEN mid S^mid_{{tau_op}}; legacy kept the previous H (estimate) or used other fallbacks (terminal) | **D17** (Q3 ruling); logged + counted (`Eq2Cache.n_degenerate_fallbacks`) |
| 9 | Terminal inventory I_{{tau_cl}} = I_{{tau_op}} - Z is NEVER clipped; legacy clipped (main.py:694) | **D8**; optional `numerical_guard` (default OFF) asserted never to bind on seeded runs |
| 10 | Identical seeds give different paths between legacy and new (and legacy main.py reseeded GLOBAL numpy state on reset) | **D10**: fresh seeding architecture; the env touches no global RNG (asserted in tests/test_determinism.py) |
| 11 | N^+ / X^7 now COUNT BUY market orders (paper convention); feature order swapped accordingly | **D3** rename (legacy economics already correct; naming only) |

Legacy emulators remain bit-pinned (10 characterization tests) and are
unaffected by Phase 3; every new-env property above is enforced by the
Phase-3 test files named in the table.
"""
    OUT_PATH.write_text(body)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
