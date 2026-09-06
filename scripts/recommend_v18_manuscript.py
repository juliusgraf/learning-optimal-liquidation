"""Generate exact v18 recommendations in memory; protected TeX is never written."""
from pathlib import Path
import difflib
from recommend_manuscript import apply_main_hunks


def main():
    original = Path('paper/main.tex').read_text()
    prior = Path('docs/manuscript_recommendations_v17.patch').read_text()
    text = apply_main_hunks(original, prior)
    old = r'\left\langle u_{\theta_\kappa}(y),e_{\theta_\kappa}(a)\right\rangle'
    assert text.count(old) == 1
    text = text.replace(old, r'\left\langle u_{\theta_\kappa}(y),\widetilde e_{\theta_\kappa}(a)\right\rangle')
    old = r'where $\omega_\kappa$ is a learned scalar shared by all nonzero actions.'
    assert text.count(old) == 1
    text = text.replace(old, r'''where $\widetilde e_{\theta_c}(a)=e_{\theta_c}(a)$ and
$\widetilde e_{\theta_a}(a)=e_{\theta_a}(a)-e_{\theta_a}(0)$.
The CLOB coefficient $\omega_c$ is learned, whereas $\omega_a=0$ is fixed.
Thus $Q_{\theta_a}(y,0)=V_{\theta_a}(y)$: the auction reference value is learned
directly, and the action encoder learns differences from that reference.
The greedy policy still maximizes the learned Q-function over every admissible
action; no benchmark policy or economic-value override is used.''')
    old = "After the common first 32 exploration episodes, conditional on epsilon exploration, an action is drawn uniformly from $\\operatorname{Adm}(y)$. In the final evaluation we use $\\varepsilon=0$. Both phase networks use the default PyTorch linear-layer initialization, with $\\omega_\\kappa=0$ initially and no imposed no-order preference. All network weights and $\\omega_\\kappa$ remain trainable."
    assert text.count(old) == 1
    text = text.replace(old, r'''After the common first 32 exploration episodes, conditional on epsilon
exploration, CLOB actions are uniform over $\operatorname{Adm}(y)$. In the
auction, define $\mathcal C(y)=\{a\in\operatorname{Adm}(y):K^a=0\}$, containing
waiting and, when admissible, cancellation without a new order. The exploration
distribution is
\[
\nu_c(y)=\mathcal U(\operatorname{Adm}(y)),\qquad
\nu_a(y)=\tfrac12\mathcal U(\mathcal C(y))+
         \tfrac12\mathcal U(\operatorname{Adm}(y)).
\]
This gives the two distinct zero-slope controls sufficient coverage without
removing any admissible action. In final evaluation $\varepsilon=0$.
Both phase networks use ordinary PyTorch linear-layer initialization, with
$\omega_c=0$ initially and $\omega_a=0$ fixed as above. No no-order margin
is imposed at initialization.''')
    old = '        \\mathcal U(\\operatorname{Adm}(y_i)),\n'
    assert text.count(old) == 1
    text = text.replace(old, '        \\nu_{\\kappa_i}(y_i),\n')
    text = text.replace('DDPG and TD3 additionally transform auction inventory',
                        'DQN, DDPG and TD3 additionally transform auction inventory')
    text = text.replace('These two algorithms fit separate CLOB and auction population statistics\non the same dedicated training paths. DQN and SAC retain pooled linear\nstandardization.',
                        'These three algorithms fit separate CLOB and auction population statistics\non the same dedicated training paths. SAC retains pooled linear\nstandardization.')
    text = text.replace('Native learning begins after 2500 CLOB and 960 auction\ntransitions are stored.',
                        'Both phase learners begin updating after the 32 complete warm-up episodes\nand at least 512 transitions have been stored in their respective buffers.')
    text = text.replace(r'\multicolumn{4}{c}{$2500/960$ (CLOB/Auction)}',
                        r'\multicolumn{4}{c}{$512/512$ (CLOB/Auction), after 32 episodes}')
    text = text.replace(r'Optimizer                            & \multicolumn{4}{c}{Adam}',
                        r'Optimizer                            & AdamW & Adam & Adam & Adam')
    text = text.replace(r'Gradient-norm clip                   & \multicolumn{4}{c}{$1.0$}',
                        r'Gradient-norm clip                   & $10.0$ & $1.0$ & $1.0$ & $1.0$')
    start = text.index('Production checkpoint selection maximizes')
    end = text.index('\\subsection{Deep Q-Network}', start)
    text = text[:start] + r'''Production checkpoint selection maximizes mean $\bar J_\lambda$ on 128
fixed validation paths, evaluated every 50 training episodes, with an
800-episode cap. A checkpoint becomes eligible after 2000 gradient updates
in each active phase. The first eligible checkpoint is evaluated immediately,
even between periodic validation dates. Training stops after four eligible
evaluations without improvement. Every training seed is retained regardless
of improvement over initialization; the untrained policy remains a diagnostic.
Neither reference-policy performance nor test outcomes enter selection.

''' + text[end:]
    anchor = '\\label{subsec:dqn}\n'
    text = text.replace(anchor, anchor + r'''
The v18 Q approximator applies non-affine layer normalization before each
hidden activation in the state encoder. Its learned action embedding is
layer-normalized and passed through a hyperbolic tangent. This bounds the
embedding while retaining unrestricted linear query and value heads; neither
rewards nor Q targets are clipped. AdamW uses weight decay $10^{-4}$.
This addresses amplification between the two learned factors in the bilinear
Q representation. Replay uses the one-step Double-DQN target below; a
development trial with uncorrected multi-step returns was rejected because
it included subsequent exploratory actions in an earlier action's target.

''', 1)
    anchor = '\\label{sec:numerics}\n'
    text = text.replace(anchor, anchor + r'''
The v18 campaign fixes ten master seeds
$\{7,42,99,123,314,577,811,1618,2024,2718\}$ and retains all 440
algorithm--market--treatment runs. Each selected policy is evaluated on 100
test episodes with matched environment seeds across policies and treatments.
Uncertainty intervals resample training-seed means; test episodes are not
independent training replications. In addition to the bundled headline versus
no-auction comparison, we isolate auction access by comparing the two arms
with both $H$/anchoring and reward shaping disabled.

The continuous-market intensity is $\lambda_0=.5$ arrivals per minute per
side. The capped Pareto mean is approximately $3.3104$ inventory units,
giving expected contra-side volume $198.62$ over 120 minutes for a
100-unit parent. This is a stylized substantial-participation liquidation
scenario, not a fit of total exchange tape volume. Quarter-intensity and
unit-intensity configurations are liquidity sensitivities; the auction's
value depends on the inventory arriving at its opening.

Historical training samples whole rebased sessions from all five stocks
on August 3--14, 2026 (50 stock--session paths). Normalization uses only this
training pool. Validation and evaluation remain specific to the requested
stock on their disjoint date partitions. V18 uses a fresh simulation seed
namespace (18001); the August 24--28 historical test dates were previously
examined in v17, so this is not a new untouched historical holdout. The
experiment measures generalization across the stated dates and simulated
order flow, not profitability on unseen securities or an empirically fitted
closing auction. Broader empirical claims require an additional frozen date
block that was not used during model development.

''', 1)
    # Apply the existing parameter recommendations to an in-memory copy.
    param_path = Path('paper/results/tables_params/params_generative.tex')
    old_params = param_path.read_text()
    param_section = prior[prior.index('--- a/paper/results/'):]
    parameters = apply_main_hunks(old_params, param_section.replace('--- a/paper/results/', '--- a/parameters/'))
    parameters = parameters.replace('$\\lambda_0$ & 1/min/side & Continuous phase Poisson intensity',
                                    '$\\lambda_0$ & 0.5/min/side & Stylized continuous-phase arrival intensity')
    patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                       fromfile='a/paper/main.tex', tofile='b/paper/main.tex'))
    patch += ''.join(difflib.unified_diff(old_params.splitlines(True), parameters.splitlines(True),
                                        fromfile='a/'+str(param_path), tofile='b/'+str(param_path)))
    Path('docs/manuscript_recommendations_v18.patch').write_text(patch)


if __name__ == '__main__':
    main()
