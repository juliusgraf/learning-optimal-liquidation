"""Revised auction projection: independent volume oracle and integration checks."""
from dataclasses import replace
import math

import numpy as np
import pytest

from helpers import drive_to_auction, load_synthetic_cfg, new_env
from lmm.config import (
    ConfigError, LEGACY_CLEARING, VOLUME_MAX_CLEARING, environment_contract,
    load_config, save_resolved,
)
from lmm.env.action_spaces import AuctionAction
from lmm.market.clearing import (
    CappedPositivePartSchedule, ClearingInputs, Eq2Cache, allocate_pro_rata,
    allocate_terminal, bracketing_tick_indices, clear_linear,
    clear_with_external_schedule, select_clearing_tick, solve_clearing,
    solve_clearing_with_hockey_stick, solve_linear_clearing,
)


def book(k=(9., 1.), s=(99.90, 100.94), ka=(), sa=(), buy=0., sell=0.):
    return ClearingInputs(np.array(k), np.array(s), np.array(ka), np.array(sa),
                          buy - sell, 100., buy_market_volume=buy, sell_market_volume=sell)


def oracle_quantities(b, p, external=None):
    """Independent participant accounting, deliberately not using allocation."""
    exo = [k * (p - s) for k, s in zip(b.K_exo, b.S_exo)]
    agent = sum(k * (p - s) for k, s in zip(b.K_agent, b.S_agent))
    if external:
        agent += external.value(p)
    return (sum(max(v, 0) for v in exo) + max(agent, 0) + b.sell_market_volume,
            sum(max(-v, 0) for v in exo) + max(-agent, 0) + b.buy_market_volume)


def test_review_counterexample_and_legacy_reproduction():
    b = book()
    revised = clear_linear(b, .01, .1)
    legacy = clear_linear(b, .01, .1, mechanism=LEGACY_CLEARING)
    assert revised.continuous_price == pytest.approx(100.004)
    assert revised.tick_index == 10001
    assert revised.tick_price == pytest.approx(100.01)
    assert revised.matched_volume == pytest.approx(.93)
    assert revised.residual_at_tick == pytest.approx(.06)
    assert abs(revised.residual_at_tick) > revised.D * .01 / 2
    assert abs(revised.residual_at_tick) <= revised.D * .01
    assert legacy.tick_price == 100.
    assert legacy.matched_volume == pytest.approx(.90)
    a = allocate_terminal(b, revised.tick_price)
    assert a.executed_supply == pytest.approx(.93)
    assert a.executed_demand == pytest.approx(.93)
    assert a.actual_agent == 0


@pytest.mark.parametrize('root,tick', [(0., 0), (100., 10000), (100.004, 10000),
                                      (100.006, 10001), (100.005, 10001), (.005, 1)])
def test_zero_volume_on_grid_zero_root_and_distance_ties(root, tick):
    r = clear_linear(book(k=(1.,), s=(root,)), .01, .1)
    assert r.tick_index == tick
    assert r.matched_volume == 0
    a = allocate_terminal(book(k=(1.,), s=(root,)), r.tick_price)
    assert a.executed_supply == a.executed_demand == 0


def test_positive_volume_tie_chooses_higher_at_half_tick():
    r = clear_linear(book(k=(1., 1.), s=(99.9, 100.11)), .01, .1)
    assert r.continuous_price == pytest.approx(100.005)
    assert r.tick_index == 10001
    assert r.matched_volume == pytest.approx(.1)


def test_volume_ties_nearest_then_higher_and_tolerance():
    # Flat quantities model a plateau; candidate tie-breaking is local to root.
    q = lambda p: (5., 5.)
    assert select_clearing_tick(1.004, .01, q) == 100
    assert select_clearing_tick(1.006, .01, q) == 101
    assert select_clearing_tick(1.005, .01, q) == 101
    # Sub-tolerance volume noise cannot flip the nearest candidate.
    assert select_clearing_tick(1.004, .01, lambda p: (5 + (p > 1)*1e-13, 6)) == 100
    assert select_clearing_tick(1.004, .01, lambda p: (5 + (p > 1)*1e-8, 6)) == 101


@pytest.mark.parametrize('alpha,k', [(.01, 10001), (.1, 3), (.01, 0), (.003, 173)])
def test_float_grid_boundaries_have_one_candidate(alpha, k):
    p = alpha * k
    roots = [p, np.nextafter(p, math.inf)]
    if k:
        roots += [np.nextafter(p, -math.inf)]
    for root in roots:
        assert bracketing_tick_indices(root, alpha) == (k,)
        calls = []
        assert select_clearing_tick(root, alpha, lambda p: calls.append(p)) == k
        assert calls == []  # no duplicate/adjacent candidate comparison
    assert bracketing_tick_indices(p + alpha * 1e-6, alpha) == (k, k + 1)
    if k:
        assert bracketing_tick_indices(p - alpha * 1e-6, alpha) == (k - 1, k)


@pytest.mark.parametrize('buy,sell', [(2., 0.), (0., 2.), (5., 3.), (3., 5.)])
def test_market_orders_in_selection_and_balanced_allocation(buy, sell):
    b = book(buy=buy, sell=sell, ka=(4., 3.), sa=(99.8, 100.2))
    r = clear_linear(b, .01, .1)
    q = oracle_quantities(b, r.tick_price)
    a = allocate_terminal(b, r.tick_price)
    assert (a.Q_supply, a.Q_demand) == pytest.approx(q)
    assert a.executed_supply == pytest.approx(min(q))
    assert a.executed_demand == pytest.approx(min(q))
    assert q[0] - q[1] == pytest.approx(r.residual_at_tick)
    # Net-only wrappers imply minimal sides; adding equal buy/sell flow adds
    # a constant to volume at every tick, hence cannot change selected price.
    net_only = replace(b, buy_market_volume=None, sell_market_volume=None)
    assert clear_linear(net_only, .01, .1).tick_index == r.tick_index


def test_aggregate_agent_no_self_trade_including_external_order():
    b = book(k=(1.,), s=(100.,), ka=(10., 10.), sa=(99., 101.))
    a = allocate_terminal(b, 100.)
    assert a.Q_supply == a.Q_demand == a.actual_agent == 0
    external = CappedPositivePartSchedule(10., 99., 10.)
    b = book(k=(1.,), s=(100.,), ka=(10.,), sa=(101.,))
    a = allocate_terminal(b, 100., external)
    assert a.requested_agent == a.Q_supply == a.Q_demand == 0
    assert a.self_trade_count == 0
    r = clear_with_external_schedule(b, external, .01, .1)
    assert r.tick_index == 10000
    assert r.matched_volume == 0


class SmoothSignedSchedule:
    def value(self, p):
        return 2. * math.tanh(p - 100.2)


@pytest.mark.parametrize('nonlinear', [False, 'capped', 'smooth'])
def test_random_admissible_books_against_exhaustive_ticks(nonlinear):
    rng = np.random.default_rng(20260907)
    for _ in range(150):
        b = book(k=rng.uniform(.1, 20, 7), s=rng.uniform(99.7, 100.3, 7),
                 ka=rng.uniform(0., 8, 5), sa=rng.uniform(99.6, 100.4, 5),
                 buy=rng.uniform(0., 3.), sell=rng.uniform(0., 3.))
        external = (CappedPositivePartSchedule(30., 99.9, .5) if nonlinear == 'capped'
                    else SmoothSignedSchedule() if nonlinear == 'smooth' else None)
        r = (clear_with_external_schedule(b, external, .01, .1) if external
             else clear_linear(b, .01, .1))
        assert 99. < r.continuous_price < 101.
        volumes = [min(oracle_quantities(b, k*.01, external)) for k in range(9900, 10101)]
        assert r.matched_volume == pytest.approx(max(volumes), abs=1e-10)
        a = allocate_terminal(b, r.tick_price, external)
        assert a.executed_supply == pytest.approx(r.matched_volume, abs=1e-10)
        assert a.executed_demand == pytest.approx(r.matched_volume, abs=1e-10)
        assert a.self_trade_count == 0
        assert abs(r.tick_price - r.continuous_price) <= .01 + 1e-12
        if not nonlinear:
            assert abs(r.residual_at_tick) <= r.D*.01 + 1e-10


def test_nonlinear_zero_root_and_no_unjustified_linear_bound():
    b = book(k=(1.,), s=(0.,))
    r = clear_with_external_schedule(b, CappedPositivePartSchedule(1000, 0, 10), .01, .1)
    assert r.continuous_price == r.tick_price == 0
    # Sharp external slope: imbalance can exceed background D*alpha.
    b = book(k=(1.,), s=(.014,))
    r = clear_with_external_schedule(b, CappedPositivePartSchedule(9., 0, 1), .01, .1)
    assert r.tick_index == 1
    assert r.nonlinear
    assert r.residual_at_tick > r.D * .01


def test_signed_external_schedule_admissibility_uses_complete_book():
    class SignedExternal:
        def value(self, p):
            return p - 3.
    b = book(k=(1.,), s=(0.,), sell=2.)
    r = clear_with_external_schedule(b, SignedExternal(), .01, .1)
    assert r.R == -2.  # background alone has no nonnegative root
    assert r.continuous_price == pytest.approx(.5, abs=1e-14)
    assert r.tick_price == .5
    assert r.matched_volume == 2.5
    a = allocate_terminal(b, .5, SignedExternal())
    assert a.actual_agent == -2.5
    assert a.executed_supply == a.executed_demand == 2.5


def test_public_wrappers_and_cache_share_projection(synthetic_cfg):
    b = book()
    assert solve_linear_clearing(b)[0] == pytest.approx(100.004)  # explicit raw API
    assert solve_linear_clearing(b, alpha=.01) == (100.01, False)
    assert solve_clearing(b, alpha=.01) == (100.01, False)
    assert solve_clearing_with_hockey_stick(b, 0., 100., alpha=.01) == (100.01, False)
    cache = Eq2Cache(synthetic_cfg.grid)
    cache.reset(99.1234)
    assert cache.read() == 99.1234  # inherited signal is not projected
    assert cache.recompute_result(b, .1).tick_index == 10001
    assert cache.recompute(b) == 100.01
    ext = CappedPositivePartSchedule(10, 100, .03)
    assert cache.recompute_result(b, .1, ext) == clear_with_external_schedule(b, ext, .01, .1)


@pytest.mark.parametrize('mechanism', [LEGACY_CLEARING, VOLUME_MAX_CLEARING])
def test_indicative_terminal_timing_cancellation_and_signed_accounting(monkeypatch, mechanism):
    cfg = load_synthetic_cfg(
        f'auction_flow.clearing_mechanism={mechanism}',
        f'experiment.artifact_schema_version={15 if mechanism == LEGACY_CLEARING else 16}',
        'algo1.clob_forecast_weights=[]', 'grid.tau_op=3', 'grid.tau_cl=6',
        'grid.h=3', 'grid.T_physical=6.', 'auction_flow.p1=0.',
        'auction_flow.p2=0.', 'auction_flow.p3=0.', 'auction_flow.p4=0.')
    env = new_env(cfg)
    clob_steps = drive_to_auction(env, 31)
    assert env.h_cl == clob_steps[-1][1]['H_used']
    flow = env._generator.auction_flow
    monkeypatch.setattr(flow, 'supply_curves', lambda: (book().K_exo, book().S_exo))
    monkeypatch.setattr(flow, 'net_market_volume', lambda: 0.)
    monkeypatch.setattr(flow, 'buy_market_volume', lambda: 0.)
    monkeypatch.setattr(flow, 'sell_market_volume', lambda: 0.)
    old_h = env.h_cl
    _, _, _, _, first = env.step(AuctionAction(0., 0, 0))
    assert first['H_used'] == old_h
    assert first['H_next'] == pytest.approx(100. if mechanism == LEGACY_CLEARING else 100.01)
    _, _, _, _, second = env.step(AuctionAction(2., 2, 0))
    assert second['H_used'] == first['H_next']
    _, _, done, _, final = env.step(AuctionAction(3., 4, 1))
    assert done
    assert final['H_used'] == second['H_next']
    assert final['S_cl'] == final['H_next'] == env.h_cl
    k, s = env._ledger.live_orders()
    np.testing.assert_array_equal(k, [3.])
    np.testing.assert_array_equal(s, [final['S_a']])
    expected = clear_linear(env._clearing_inputs(), .01, .1, mechanism=mechanism)
    assert final['S_cl'] == expected.tick_price
    a = allocate_terminal(env._clearing_inputs(), expected.tick_price)
    assert final['Z'] == a.actual_agent
    assert final['I_final'] == pytest.approx(env._I_tau_op - a.actual_agent)
    assert final['auction_economic_cash'] == pytest.approx(a.actual_agent * expected.tick_price)
    assert final['cancellation_fee'] == pytest.approx(2 * cfg.reward.d)


def test_config_versions_and_fitted_weights_fail_closed(tmp_path):
    old = load_synthetic_cfg()
    assert old.auction_flow.clearing_mechanism == LEGACY_CLEARING
    with pytest.raises(ConfigError, match='different clearing mechanism'):
        load_synthetic_cfg('auction_flow.clearing_mechanism=max_volume_v2', 'experiment.artifact_schema_version=16')
    new = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                      'configs/clearing/max_volume_v2.yaml')
    assert new.auction_flow.clearing_mechanism == VOLUME_MAX_CLEARING
    assert not new.algo1.clob_forecast_weights
    assert environment_contract(old) != environment_contract(new)
    path = tmp_path / 'resolved.yaml'
    save_resolved(new, path)
    assert load_config(path) == new


def test_reject_invalid_sides_and_roots():
    with pytest.raises(ValueError, match='net market'):
        clear_linear(replace(book(), buy_market_volume=1.), .01, .1)
    with pytest.raises(ValueError, match='nonnegative'):
        allocate_pro_rata(np.array([10.]), 0., -1., 0.)
    with pytest.raises(ValueError, match='nonnegative root'):
        clear_with_external_schedule(book(k=(1.,), s=(0.,)),
                                     CappedPositivePartSchedule(10, -1, 10), .01, .1)


@pytest.mark.parametrize('name', ['as', 'twap'])
def test_revised_benchmarks_complete_and_use_shared_terminal_book(name):
    from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
    from lmm.rl.loops import run_episode
    cfg = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                      'configs/clearing/max_volume_v2.yaml', overrides=['benchmark.as_n_samples=500'])
    env = new_env(cfg)
    agent = ASBenchmarkAgent(cfg) if name == 'as' else TWAPBenchmarkAgent(cfg)
    if name == 'as':
        agent.calibrate(rng_k=np.random.default_rng(1))
    agent.bind(env)
    result = run_episode(env, agent, 11, chi=1., train=False)
    b, external = env._clearing_inputs(), env._ledger.external_schedule()
    r = (clear_with_external_schedule(b, external, .01, .1) if external
         else clear_linear(b, .01, .1))
    allocation = allocate_terminal(b, r.tick_price, external)
    assert result.s_cl == r.tick_price
    assert result.z_tau_cl == allocation.actual_agent
    assert allocation.executed_supply == pytest.approx(allocation.executed_demand)
    assert result.self_trade_count == 0
    assert math.isfinite(result.risk_adjusted_pnl)
