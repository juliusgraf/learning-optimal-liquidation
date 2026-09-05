"""Generate an exact review patch without writing either protected TeX file."""
from __future__ import annotations

import difflib
from pathlib import Path


def apply_main_hunks(original, patch):
    """Reuse the checked v15 prose corrections in memory, then supersede them."""
    text = original
    section = patch.split('--- a/paper/results/')[0]
    chunks = section.split('\n@@ ')
    for chunk in chunks[1:]:
        body = chunk.split('\n', 1)[1]
        old = ''.join(line[1:]+'\n' for line in body.splitlines() if line.startswith((' ', '-')))
        new = ''.join(line[1:]+'\n' for line in body.splitlines() if line.startswith((' ', '+')))
        assert text.count(old) == 1, old[:120]
        text = text.replace(old, new, 1)
    return text


def main():
    path = Path('paper/main.tex')
    original = path.read_text()
    text = apply_main_hunks(original, Path('docs/manuscript_recommendations.patch').read_text())
    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old[:120]
        text = text.replace(old, new, 1)
    replace(r'We assume that the difference between $X^3$ and $S^\bullet$ tolerated is given by $k^\star\alpha$ for some $k^\star$ fixed.',
            r'The multiplier has clipping scale $k^\star\alpha$ for fixed $k^\star$. We interpret $S_0/(k^\star\alpha)$ as the local opportunity-cost preference, rather than treating the clipping scale as an empirically tolerated market-price gap.')
    replace(r"The agent receives a ``fictive'' reward $K_t^aH_t^\cl(H_t^\cl-S_t^a)$",
            r"Fix a dimensionless shaping weight $\omega_{\mathrm{sh}}\in[0,1]$. The agent receives a ``fictive'' reward $\omega_{\mathrm{sh}}K_t^aH_t^\cl(H_t^\cl-S_t^a)$")
    replace(r'Here, $d_t$ is an exchange cancellation fee.',
            r'Here, $d_t$ is a stylized liquidity-withdrawal cost; its numerical value is not an estimate of a universal exchange cancellation fee.')
    replace(r'$u_t = K_t^a H_t^\cl (H_t^\cl-S_t^a) + f^a(K_t^a H_t^\cl(H_t^\cl-S_t^a))$ The reward',
            r'$u_t = \omega_{\mathrm{sh}}\left[K_t^a H_t^\cl (H_t^\cl-S_t^a) + f^a(K_t^a H_t^\cl(H_t^\cl-S_t^a))\right]$. The reward')
    replace(r'+ f^a\left (S_{\tau^\cl}^\cl Z_{\tau^\cl}\right)',
            r'+ \omega_{\mathrm{sh}} f^a\left (S_{\tau^\cl}^\cl Z_{\tau^\cl}\right)')
    replace(r'One can interpret the penalty as removing a fraction $q$ of the reward. With $q = 1$, the purchase-side component is neutralized.',
            r'The function adds back fraction $q$ of a negative cash-like component before applying $\omega_{\mathrm{sh}}$. At $q=1$ that fictive purchase component is neutralized; the headline preference $q=0$ gives no purchase subsidy. Actual purchase cash is always paid in full in economic evaluation.')
    replace(r'When a live order is canceled, the exact fictive reward originally credited to that order is subtracted once as a clawback.',
            r'When a live order is canceled, the exact weighted fictive reward originally credited to that order is subtracted once as a clawback. The weight does not multiply actual cash, inventory risk or cancellation costs. The original specification is recovered at $\omega_{\mathrm{sh}}=1$; the headline uses $\omega_{\mathrm{sh}}=\alpha/S_0=10^{-4}$. This changes the shaped objective, denoted $J_{\omega_{\mathrm{sh}}}$, and is not merely numerical reward rescaling.')
    replace(r'& J(\pi) = \mathbb{E}_\pi', r'& J_{\omega_{\mathrm{sh}}}(\pi) = \mathbb{E}_\pi')
    replace(r'Headline runs train on the shaped criterion $J$;',
            r'Headline runs train on the weighted shaped criterion $J_{\omega_{\mathrm{sh}}}$;')
    start = text.index('\nThe revised numerical calibration')
    end = text.index('\n\\section{Learning Algorithms}', start)
    text = text[:start] + r'''
The shared stylized calibration uses $S_0=100$, $\alpha=.01$, $I_0=100$,
$\lambda=.01$, $q=0$, $k^\star=10000$, $\beta=.25$ and $d=.001$.
Here $k^\star$ is a training preference: $S_0/(k^\star\alpha)=1$ makes the
local CLOB shaping deduction approximately equal to the forgone price gap
per unit executed. The clipping threshold $k^\star\alpha=100$ is a consequence
of multiplying gross proceeds, not an empirically tolerated market gap.
$q=0$ gives no purchase subsidy. The auction weight makes the fictive credit
approximately one tick per projected unit, reducing the spurious incentive
to sell beyond liquidation; it does not make shaped and economic objectives
equivalent. A 10-unit residual costs $\lambda10^2=1$, equal to one tick on
the initial inventory. A maximum schedule has slope $32\beta=8$ and offers
.8 units at a ten-tick local gap; 30 such schedules offer 24 nominal units.
These are scale diagnostics, not hard execution bounds. Actual signed
pro-rata fills remain unrestricted by terminal inventory.
The modeled two-tick spread is 2 basis points at $S_0$; historical prices are
rebased, so this is not necessarily a one-cent tick in raw share prices.
In the training split, median daily selected-quote spreads are 5.6275 bps
(CAT), .8679 (GOOGL), 2.0769 (JPM), 1.4095 (MSFT), and 1.3842 (PG).
The frozen sidecar mislabeled quote sizes as round lots: Alpaca's CTA/UTP
fields have been in shares since November 3, 2025.\footnote{\url{https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change}}
The pooled training median of daily selected best-quote sizes is 80 shares.
The model's $2\operatorname{Beta}(2,5)$ top-depth median is .5289 inventory
units. Approximately 150 shares per model inventory unit is therefore an
illustrative top-depth normalization. It is not a fit of full depth or trade
flow; intensities represent stylized eligible flow, not the complete SIP tape.
Results are reported in normalized units or basis points of initial notional.
Quote data cannot identify auction slopes or market-order intensities.
The exogenous slope support $[1,20]$ and order-flow
parameters are stylized; historical experiments replay midprices within the
same simulated market. The reference policies retain their stated formulas.
''' + text[end:]
    start = text.index('\\paragraph{Numerical conditioning and replay.}')
    end = text.index('\\subsection{Deep Q-Network}', start)
    text = text[:start] + r'''\paragraph{Numerical conditioning and replay.}
All learners use the same information in the 18-coordinate observation.
Before fitting normalization, replace $H$ by $H-S^{\mathrm{mid}}$ and the
slope-weighted quotes $W_a,W_e$ by $W_a-S^{\mathrm{mid}}G_a$ and
$W_e-S^{\mathrm{mid}}G_e$. In the auction define
\[
\Delta=\frac{W_a-S^{\mathrm{mid}}G_a+W_e-S^{\mathrm{mid}}G_e+D}{G_a+G_e},
\qquad \widehat Z=G_a\Delta-(W_a-S^{\mathrm{mid}}G_a),
\]
where $D$ is observed buy-minus-sell taker volume. Replace the two quote-moment
coordinates by $I-\widehat Z$ and $\Delta$, retaining the slopes, inventory
and imbalance. This is invertible when $G_a+G_e>0$, uses no future clearing
information, and does not require the ablated $H$ signal. CLOB-inactive
raw moments remain zero. Fit population means and standard deviations on
16 dedicated training trajectories covering persistence, cancellations,
abstention and large opening inventory, then freeze and checkpoint them.
Time and decision index divide by $\tau^{\mathrm{cl}}$; the cancellation
indicator is unchanged. Validation and test never fit the transform.

Initial-inventory centering subtracts the constant $S_0I_0$ over an episode.
Let $R_i$ be the resulting immediate reward including the terminal component
on the final transition. Replay uses
\[
\widetilde R_i=R_i+\Phi(y_{i+1})-\Phi(y_i)
-\mathbf1_{\{t_i<\tau^{\mathrm{op}}\}}q_{\mathrm{ref}}((t_i+t_{i+1})/2)
 (S^{\mathrm{mid}}_{t_{i+1}}-S^{\mathrm{mid}}_{t_i}).
\]
Set $\Phi(\mathrm{terminal})=0$. During CLOB trading,
\[
\Phi(y)=(S^{\mathrm{mid}}-S_0)I
-\lambda\max\{I-I_0(1-t/\tau^{\mathrm{op}}),0\}^2.
\]
In the auction, set $\Phi(y)=(S^{\mathrm{mid}}-S_0)I+
\Delta\widehat Z-\lambda(I-\widehat Z)^2$.
The potential telescopes and has initial value zero. The deterministic
reference curve $q_{\mathrm{ref}}$ is the mean CLOB inventory path from the
same training calibration, excluding the deliberately no-CLOB mode;
interpolate onto minute times and freeze it before learning. Subtracting its
realized price return removes the same amount from every policy on a common
exogenous price/clock path, preserving policy differences. Its historical
expectation need not be zero. Neither adjustment enters reported shaped J,
PnL or $\bar J_\lambda$. Actual settlement remains tick-price pro-rata.

All methods use one-step replay. CLOB junction rows continue through the
auction target network; only the final transition is terminal. The first
32 training episodes share a finite exploration design: four CLOB modes
(maximum volume at offset one, abstention, and two random fixed proposals)
cross four auction modes (no order, persistent sells, persistent buys,
and cancel/replace sells), twice. Random slopes and offsets are sampled
independently of future data; no benchmark actions or optimization labels
are supplied. Native learning begins after 2500 CLOB and 960 auction
transitions are stored. Subsequent exploration follows each method's stated
rule. Initial learning rates share the fixed factor
$\max\{.1,2^{-e/90}\}$, independent of outcomes. Evaluation is greedy.

Production checkpoint selection maximizes mean $\bar J_\lambda$ on 100
fixed validation paths, evaluated every 100 training episodes, with an
800-episode cap. A checkpoint becomes eligible only after 5000 CLOB and
2000 auction gradient updates (the auction requirement is omitted when
that phase is disabled). A reportable checkpoint must also improve on the
initial policy's economic validation score. The initial and intermediate
policies remain diagnostic records. Training stops after six eligible
validation evaluations without improvement in the best mature score;
the patience counter starts at the first eligible evaluation. A run with
no reportable checkpoint is flagged as a selection failure, preserving its
diagnostics. Neither benchmark performance nor test outcomes enter selection.

''' + text[end:]
    start = text.index('        \\STATE For DQN, accumulate')
    end = text.index('\n\n        \\IF', start)
    text = text[:start] + r'''        \STATE Form the centered and conditioned total reward $\widetilde R_i$
        defined above. Store $(y_i,a_i,c_r(\widetilde R_i-g_i),y_{i+1},
        \eta_i,c_rg_i)$ in $\mathcal D_{\kappa_i}$.
        The terminal target below therefore contains $G$ exactly once.
        During the first 32 episodes, replace the selected action by the
        common finite exploration proposal described above.''' + text[end:]
    # The exploration override must occur before environment execution.
    replace(r'''        The terminal target below therefore contains $G$ exactly once.
        During the first 32 episodes, replace the selected action by the
        common finite exploration proposal described above.''',
            r'''        The terminal target below therefore contains $G$ exactly once.''')
    replace(r'        \STATE Apply $a_i$ at $t_i$',
            r'''        \STATE If $e<32$, replace $a_i$ by the common finite exploration proposal.
        \STATE Apply $a_i$ at $t_i$''')
    replace(r'''        \eta_i
        G(X_{\tau^\cl})$''', r'''        \eta_i\bigl[G(X_{\tau^\cl})-S_0 I_{\tau^{\mathrm{op}}}\bigr]$''')
    replace(r'$512/256$ (CLOB/Auction)', r'$2500/960$ (CLOB/Auction)')
    replace(r'$2\times 64$', r'$2\times 256$')
    replace(r'Reward scale $c_r$                   & \multicolumn{4}{c}{$10^{-2}$}',
            r'Reward scale $c_r$                   & \multicolumn{4}{c}{$1$}')
    replace(r'Replay horizon & $5$ & $1$ & $1$ & $1$', r'Replay horizon & $1$ & $1$ & $1$ & $1$')
    replace(r'Actor               & -                    & $3\times10^{-4}$ & $3\times10^{-4}$ & $3\times10^{-4}$',
            r'Actor               & -                    & $1\times10^{-4}$ & $3\times10^{-4}$ & $3\times10^{-4}$')
    replace('Before each phase replay warm-up is reached, proposals are sampled uniformly from the normalized proposal box.',
            'The common finite exploration design overrides behavior in the first 32 episodes. If a phase buffer is still below its warm-up threshold afterward, its proposals are uniform in the normalized proposal box.')
    text = text.replace('Conditional on exploration, an action is drawn uniformly',
                        'After the common first 32 exploration episodes, conditional on epsilon exploration, an action is drawn uniformly')
    # No protected source is written; only a review patch is produced.
    patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                       fromfile='a/paper/main.tex', tofile='b/paper/main.tex'))
    param_path = Path('paper/results/tables_params/params_generative.tex')
    old = param_path.read_text()
    new = old
    for left, right in [
        ('$U_1$ & 0.1', '$U_1$ & 1'), ('$U_2$ & 2', '$U_2$ & 20'),
        (r'$\lambda$ & 2 & Inventory penalty', r'$\lambda$ & 0.01 & Inventory penalty'),
        ('$q$ & 1 & Shaping-treatment purchase-side coefficient; inactive in headline runs',
         '$q$ & 0 & No purchase subsidy; headline shaping is active'),
        (r'$k^\star$ & 1000 & CLOB shaping tolerance; inactive in headline runs',
         r'$k^\star$ & 10000 & Opportunity-cost preference, $S_0/(k^\star\alpha)=1$'),
        ('$d$ & 0.1 & Cancellation-fee increment', '$d$ & 0.001 & Stylized withdrawal-cost increment'),
        (r'$\alpha$ & 0.01 & Tick size', r'$\alpha$ & 0.01 & Tick in rebased model-price units'),
        (r'$\beta$ & 1 & Tick size of grid on $K^a$', r'$\beta$ & 0.25 & Slope step (inventory units per price unit)'),
    ]:
        assert new.count(left) == 1, left
        new = new.replace(left, right, 1)
    new = new.replace('$d$ & 0.001', r'$\omega_{\mathrm{sh}}$ & 0.0001 & Fictive-credit weight $\alpha/S_0$ \\'+'\n$d$ & 0.001')
    patch += ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/'+str(param_path), tofile='b/'+str(param_path)))
    Path('docs/manuscript_recommendations_v16.patch').write_text(patch)


if __name__ == '__main__': main()
