"""Algorithm 2 statistical tests (`alg:generative_model`; rulings D6, D7).

Seeded checks of the generative laws: Pareto market-order sizes (capped at
V), Beta top-of-book with geometric depth decay, the per-step arrival
guarantee (Assumption `assump:presence`), Poisson arrival counts on
[0, tau_op] (validating the memoryless redraw construction documented in
market/generator.py), and the auction event frequencies p1, p2, p3 and the
DIRECT Bernoulli taker-cancellation p4 = 0.05 (ruling D7).

All bands are +-4 standard errors of the seeded estimate (or generous
chi-square-style bands for variances); failures indicate a distributional
regression, not noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from lmm.env.action_spaces import ClobAction
from lmm.market.auction import ExogenousAuctionFlow
from lmm.market.clob import OrderBook, sample_mo_volume

from helpers import NOOP_CLOB, load_synthetic_cfg, new_env

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_cfg()


# ---------------------------------------------------------------------------
# Market-order sizes: min(Pareto(v_m, gamma_m), V)
# ---------------------------------------------------------------------------


def test_pareto_capped_volume_law(cfg):
    p = cfg.clob_flow
    rng = np.random.default_rng(0)
    n = 300_000
    xs = np.array([sample_mo_volume(rng, p.v_m, p.gamma_m, p.V_max) for _ in range(n)])

    assert xs.min() >= p.v_m and xs.max() <= p.V_max

    # E[min(X, c)] = v_m + v_m^g (c^{1-g} - v_m^{1-g}) / (1 - g).
    g, vm, c = p.gamma_m, p.v_m, float(p.V_max)
    mean_theory = vm + vm**g * (c ** (1 - g) - vm ** (1 - g)) / (1 - g)
    assert xs.mean() == pytest.approx(mean_theory, abs=4 * xs.std() / np.sqrt(n))

    # Tail P(V > v) = (v_m/v)^gamma_m for v < cap; atom P(V = cap) = (v_m/c)^g.
    for v in (3.0, 5.0, 10.0, 20.0):
        theo = (vm / v) ** g
        band = 4 * np.sqrt(theo * (1 - theo) / n)
        assert (xs > v).mean() == pytest.approx(theo, abs=band)
    atom = (vm / c) ** g
    assert (xs == c).mean() == pytest.approx(atom, abs=4 * np.sqrt(atom * (1 - atom) / n))


# ---------------------------------------------------------------------------
# Book refresh: V^{zeta,1} ~ V_inf * Beta(beta_a, beta_b), geometric decay
# ---------------------------------------------------------------------------


def test_beta_top_of_book_moments_and_geometric_decay(cfg):
    p = cfg.clob_flow
    book = OrderBook(p, cfg.grid)
    rng = np.random.default_rng(1)
    n = 50_000
    tops_ask = np.empty(n)
    tops_bid = np.empty(n)
    decay = p.depth_decay ** np.arange(p.Lc)
    for i in range(n):
        book.refresh(rng)
        tops_ask[i] = book.ask_volumes[0]
        tops_bid[i] = book.bid_volumes[0]
        if i < 100:  # the depth profile is deterministic given the top
            np.testing.assert_allclose(book.ask_volumes, tops_ask[i] * decay, rtol=1e-12)
            np.testing.assert_allclose(book.bid_volumes, tops_bid[i] * decay, rtol=1e-12)

    a, b = p.beta_a, p.beta_b
    mean_theory = a / (a + b)
    var_theory = a * b / ((a + b) ** 2 * (a + b + 1))
    for tops in (tops_ask, tops_bid):
        x = tops / p.V_inf
        assert (x.min() >= 0.0) and (x.max() <= 1.0)
        assert x.mean() == pytest.approx(mean_theory, abs=4 * np.sqrt(var_theory / n))
        assert x.var() == pytest.approx(var_theory, abs=0.0015)


# ---------------------------------------------------------------------------
# Arrivals: assump:presence on the hat_t grid; Poisson counts on [0, tau_op]
# ---------------------------------------------------------------------------


def test_presence_assumption_and_poisson_arrival_counts(cfg):
    """Every UNCAPPED CLOB step has >= 1 new arrival per side (steps
    truncated by the cap at n may lack one — documented legacy behavior,
    AUDIT A.1); episode totals N^{+/-}_{tau_op} have Poisson(lambda0 *
    tau_op) mean and variance, validating that the per-step exponential
    redraw is the paper's Poisson construction (memorylessness)."""
    env = new_env(cfg)
    n_cap = cfg.grid.tau_op - 1
    lam_total = cfg.clob_flow.lambda0 * cfg.grid.tau_op
    n_episodes = 300
    totals_buy = np.empty(n_episodes)
    totals_sell = np.empty(n_episodes)
    for e in range(n_episodes):
        env.reset(seed=10_000 + e)
        nb = ns = 0
        while env.phase == "clob":
            _, _, _, _, info = env.step(NOOP_CLOB)
            if info["t_next"] < n_cap:  # uncapped step
                assert info["n_buy_step"] >= 1 and info["n_sell_step"] >= 1
            nb += info["n_buy_step"]
            ns += info["n_sell_step"]
        totals_buy[e] = nb
        totals_sell[e] = ns

    for totals in (totals_buy, totals_sell):
        assert totals.mean() == pytest.approx(
            lam_total, abs=4 * np.sqrt(lam_total / n_episodes)
        )
        # Poisson: variance == mean (generous band for n = 300).
        assert totals.var() == pytest.approx(lam_total, abs=0.35 * lam_total)
    # Independence across sides (Algorithm 2: independent processes).
    corr = np.corrcoef(totals_buy, totals_sell)[0, 1]
    assert abs(corr) < 4 / np.sqrt(n_episodes)


# ---------------------------------------------------------------------------
# Auction event frequencies (p1, p2, p3, p4 = 0.05 per ruling D7)
# ---------------------------------------------------------------------------


def test_auction_event_frequencies(cfg):
    p = cfg.auction_flow
    flow = ExogenousAuctionFlow(p, cfg.grid, cfg.clob_flow)
    rng = np.random.default_rng(7)

    n_episodes, n_steps = 4_000, cfg.grid.tau_cl - cfg.grid.tau_op
    counts = {k: [0, 0] for k in ("p1", "p2", "p3_buy", "p3_sell", "p4_buy", "p4_sell")}

    def tally(key: str, gated: bool, fired: bool) -> None:
        if gated:
            counts[key][0] += int(fired)
            counts[key][1] += 1

    k_all = []
    for e in range(n_episodes):
        flow.reset(100.0)
        for _ in range(n_steps):
            mm_before, buy_before, sell_before = flow.n_mm, flow.n_buy, flow.n_sell
            ev = flow.step(rng)
            tally("p1", mm_before < p.La, ev.new_mm)
            tally("p2", mm_before > 0 or ev.new_mm, ev.mm_cancelled)
            tally("p3_buy", True, ev.new_buy_taker)
            tally("p3_sell", True, ev.new_sell_taker)
            tally("p4_buy", buy_before > 0 or ev.new_buy_taker, ev.buy_taker_cancelled)
            tally("p4_sell", sell_before > 0 or ev.new_sell_taker, ev.sell_taker_cancelled)

        # Exogenous MM law checks on the surviving end-of-episode ledger
        # (cancellation is independent of (K, S), so survivors keep the law).
        K, S = flow.supply_curves()
        k_all.extend(K.tolist())
        if len(S):
            offsets = S / cfg.grid.alpha - np.floor(100.0 / cfg.grid.alpha)
            np.testing.assert_allclose(offsets, np.round(offsets), atol=1e-9)  # on-tick
            assert np.all(np.abs(offsets) <= p.price_band_ticks)

    expected = {
        "p1": p.p1, "p2": p.p2, "p3_buy": p.p3, "p3_sell": p.p3,
        "p4_buy": p.p4, "p4_sell": p.p4,
    }
    for key, (fired, gated) in counts.items():
        prob = expected[key]
        freq = fired / gated
        band = 4 * np.sqrt(prob * (1 - prob) / gated)
        assert freq == pytest.approx(prob, abs=band), (key, freq, prob, gated)
    assert expected["p4_buy"] == 0.05  # ruling D7: the EFFECTIVE legacy rate

    # K^i ~ U(K_min, K_max): bounds and mean.
    k_all = np.asarray(k_all)
    assert k_all.min() >= p.K_min and k_all.max() <= p.K_max
    mean_k = (p.K_min + p.K_max) / 2
    sd_k = (p.K_max - p.K_min) / np.sqrt(12)
    assert k_all.mean() == pytest.approx(mean_k, abs=4 * sd_k / np.sqrt(len(k_all)))
