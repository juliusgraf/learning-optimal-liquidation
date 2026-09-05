"""Generate the v17 review patch; never write either protected TeX source."""
from __future__ import annotations

import difflib
from pathlib import Path

from recommend_manuscript import apply_main_hunks


def main():
    main_path = Path('paper/main.tex')
    original = main_path.read_text()
    prior = Path('docs/manuscript_recommendations_v16.patch').read_text()
    proposed = apply_main_hunks(original, prior)
    old = r'''Here $k^\star$ is a training preference: $S_0/(k^\star\alpha)=1$ makes the
local CLOB shaping deduction approximately equal to the forgone price gap
per unit executed. The clipping threshold $k^\star\alpha=100$ is a consequence
of multiplying gross proceeds, not an empirically tolerated market gap.'''
    new = r'''The parameter $k^\star$ is a dimensionless number of ticks: its clipping
distance is indeed 10000 ticks, or $k^\star\alpha=100$ model-price units.
This is not a realistic tolerated market-price gap. To interpret the training
preference, define $\eta_{\mathrm C}=S_0/(k^\star\alpha)=1$ and
$g=(H-p)_+$. The exact per-execution CLOB reward is
\[
 r^{\mathrm C}(p,E,H)=pE-\eta_{\mathrm C}\frac{p}{S_0}E
 \min\{g,S_0/\eta_{\mathrm C}\}.
\]
Thus, locally at $p=S_0$, a one-tick shortfall deducts one tick per unit sold.
Reducing $k^\star$ to 100 would make the same local deduction 100 times
stronger. The economically interpretable preference is $\eta_{\mathrm C}$;
$k^\star$ retains compatibility with the manuscript's gross-proceeds formula.
Neither quantity is estimated from observed exchange price tolerances.'''
    assert proposed.count(old) == 1
    proposed = proposed.replace(old, new)
    anchor = 'Time and decision index divide by $\\tau^{\\mathrm{cl}}$; the cancellation\nindicator is unchanged. Validation and test never fit the transform.'
    addition = r'''

DDPG and TD3 additionally transform auction inventory $I$ and continuous-root
residual exposure $I-\widehat Z$ by $x\mapsto\operatorname{asinh}(x/I_s)$,
where $I_s=\alpha/\lambda$ (use $I_0$ if $\lambda=0$). The scale $I_s$ is
the inventory at which one-tick execution value $\alpha I_s$ equals the
quadratic penalty $\lambda I_s^2$. The transform is signed, unbounded and
invertible; it neither caps fills nor supplies additional information.
These two algorithms fit separate CLOB and auction population statistics
on the same dedicated training paths. DQN and SAC retain pooled linear
standardization. Consequently the headline compares the specified algorithm
implementations, including their preprocessing, rather than isolating an
algorithm-only causal effect. Supplementary matched representation controls
apply pooled linear preprocessing to all methods or apply the phase-specific
signed-asinh preprocessing to all methods. Rewards, controls and optimizer
settings are unchanged by those representation controls.
'''
    assert proposed.count(anchor) == 1
    proposed = proposed.replace(anchor, anchor+addition)
    anchor = 'Results are reported in normalized units or basis points of initial notional.'
    addition = r'''
For a dimensional dollar interpretation let $M$ be shares per inventory unit
and $A=P_0^{\mathrm{raw}}/S_0$ be dollars per model-price unit. Dollar PnL is
$MA$ times model PnL, physical slope quantum is $M\beta/A$ shares per dollar,
and the physical quadratic coefficient is $A\lambda/M$ dollars per share
squared. The rebased model tick is $A\alpha$ dollars, not necessarily one
raw cent. These conversions do not establish empirically identified auction
depth or transaction costs.'''
    assert proposed.count(anchor) == 1
    proposed = proposed.replace(anchor, anchor+addition)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True),
                                       fromfile='a/paper/main.tex', tofile='b/paper/main.tex'))
    # The protected parameter values are unchanged from the v16 recommendation;
    # sharpen only the explanation of the high k-star value in the review patch.
    param_patch = '--- a/paper/results/'+prior.split('--- a/paper/results/',1)[1]
    param_patch = param_patch.replace(
        r'Opportunity-cost preference, $S_0/(k^\star\alpha)=1$',
        r'Clipping distance in ticks; local weight $S_0/(k^\star\alpha)=1$')
    Path('docs/manuscript_recommendations_v17.patch').write_text(patch+param_patch)


if __name__ == '__main__': main()
