"""Per-step episode traces for the anatomy figures (Phase 7).

``make_figures.py`` must regenerate everything from saved outputs only (no
env-stepping inside plotting code; engineering conventions). The episode- and
benchmark-anatomy figures need a full per-step trajectory, so ``evaluate.py``
replays a handful of eval episodes with an :class:`EpisodeTraceRecorder`
attached to ``rl.loops.run_episode(on_step=...)`` and writes one CSV per
(policy, episode) under ``eval/traces/``. The recorder only READS values
already computed for the step (the ``info`` dict + a few env accessors that the
:class:`~lmm.env.action_spaces.ContinuousActionAdapter` delegates), so the
traced trajectory is identical to an untraced one (asserted in tests).

Schema: one row per decision step, the UNION of CLOB-only and auction-only
columns; phase-inapplicable cells are empty. The final auction row carries the
terminal clearing fields (``is_terminal = 1``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["TRACE_COLUMNS", "EpisodeTraceRecorder"]

TRACE_COLUMNS = [
    # -- common (every step) --
    "step",
    "t",
    "phase",
    "reward",
    "cum_reward",
    "s_mid",
    "inventory",
    "h_cl",
    "h_next",
    # -- CLOB-only --
    "E_t",
    "S_bullet",
    "top_ask",
    "top_bid",
    "depth_ask",
    "depth_bid",
    "n_buy_step",
    "n_sell_step",
    "act_volume",
    "act_delta",
    # -- auction-only --
    "S_a",
    "d_t",
    "n_mm",
    "n_buy_auc",
    "n_sell_auc",
    "ev_new_mm",
    "ev_mm_cancelled",
    "ev_new_buy_taker",
    "ev_new_sell_taker",
    "ev_buy_taker_cancelled",
    "ev_sell_taker_cancelled",
    "act_Ka",
    "act_ell",
    "act_b",
    "auction_anchor",
    "auction_anchor_b",
    "auction_anchor_price",
    "act_cancel",
    "degenerate_fallback",
    # -- terminal row only --
    "is_terminal",
    "S_cl",
    "Z",
    "I_final",
    "terminal_reward",
]


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else format(float(v), ".17g")
    return str(v)


class EpisodeTraceRecorder:
    """Accumulate per-step rows; pass an instance as ``run_episode(on_step=...)``.

    Top-of-book volumes / depths (CLOB) and the running auction counts are read
    from the simulator's private generator; everything
    else comes from the step's ``info``. CLOB rows record the standing book as
    it is right AFTER the step (the book the next decision sees; on the final
    CLOB step it is the post-flow book, since no refresh happens at the auction
    open) — a documented vintage, fine for the qualitative anatomy panels.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(
        self,
        step_idx: int,
        phase: str,
        t_decision: float,
        reward: float,
        cum_reward: float,
        info: dict[str, Any],
        env: Any,
    ) -> None:
        row: dict[str, Any] = dict.fromkeys(TRACE_COLUMNS, "")
        row.update(
            step=step_idx,
            t=info["t"],
            phase=phase,
            reward=reward,
            cum_reward=cum_reward,
            s_mid=env.s_mid,
            inventory=env.inventory,
            h_cl=info["H_used"],
            h_next=info["H_next"],
            is_terminal=0,
        )
        if phase == "clob":
            base_env = env.env if hasattr(env, "env") else env
            book = base_env._generator.book  # noqa: SLF001
            a = info["action"]
            row.update(
                E_t=info["E_t"],
                S_bullet=info["S_bullet"],
                top_ask=float(book.ask_volumes[0]),
                top_bid=float(book.bid_volumes[0]),
                depth_ask=book.depth(+1),
                depth_bid=book.depth(-1),
                n_buy_step=info["n_buy_step"],
                n_sell_step=info["n_sell_step"],
                act_volume=a.volume,
                act_delta=a.delta,
            )
        else:
            base_env = env.env if hasattr(env, "env") else env
            flow = base_env._generator.auction_flow  # noqa: SLF001
            ev = info["events"]
            a = info["action"]
            row.update(
                S_a=info["S_a"],
                d_t=info["d_t"],
                n_mm=flow.n_mm,
                n_buy_auc=flow.n_buy,
                n_sell_auc=flow.n_sell,
                ev_new_mm=int(ev.new_mm),
                ev_mm_cancelled=int(ev.mm_cancelled),
                ev_new_buy_taker=int(ev.new_buy_taker),
                ev_new_sell_taker=int(ev.new_sell_taker),
                ev_buy_taker_cancelled=int(ev.buy_taker_cancelled),
                ev_sell_taker_cancelled=int(ev.sell_taker_cancelled),
                act_Ka=a.K_a,
                act_ell=a.ell if hasattr(a, "ell") else "",
                act_b=int(info["executed_b"]),
                auction_anchor=info["auction_anchor"],
                auction_anchor_b=int(info["auction_anchor_b"]),
                auction_anchor_price=info["auction_anchor_price"],
                act_cancel=a.cancel,
                degenerate_fallback=int(info["degenerate_fallback"]),
            )
            if "S_cl" in info:  # terminal step (folded clearing reward)
                row.update(
                    is_terminal=1,
                    S_cl=info["S_cl"],
                    Z=info["Z"],
                    I_final=info["I_final"],
                    terminal_reward=info["terminal_reward"],
                )
        self.rows.append(row)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRACE_COLUMNS)
            for row in self.rows:
                writer.writerow([_fmt(row[c]) for c in TRACE_COLUMNS])
