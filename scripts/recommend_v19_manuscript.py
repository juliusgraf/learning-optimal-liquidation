"""Generate a consolidated review diff; never write either protected TeX file."""
from pathlib import Path
import difflib
from recommend_manuscript import apply_main_hunks


def main():
    main_path = Path('paper/main.tex')
    original = main_path.read_text()
    prior = Path('docs/manuscript_recommendations_v18.patch').read_text()
    text = apply_main_hunks(original, prior)

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old[:100]
        text = text.replace(old, new, 1)

    replace('The v18 campaign fixes ten master seeds', 'The v19 campaign fixes ten master seeds')
    replace('''independent training replications. In addition to the bundled headline versus
no-auction comparison, we isolate auction access by comparing the two arms
with both $H$/anchoring and reward shaping disabled.''', r'''independent training replications. The headline retains the calibrated $H$
observation, both manuscript reward preferences, the indicative auction
anchor and dense potential conditioning. Five additional synthetic cells
separate the following mechanisms. A fixed-mid-anchor full-$J$ cell isolates
anchoring from information. Economic dense and economic sparse cells retain
$H$ and that fixed anchor, disable both manuscript preferences, and differ
only in action-dependent auction potential credit. An economic dense $H$-off
cell isolates the forecast observation. The fixed-anchor full-$J$ cell
measures the combined preference effect. CLOB-only and auction-only shaping
are optional follow-ups outside this final matrix; separate contributions
cannot be inferred from the combined contrast. Finally,
auction access compares the $H$-off economic cell with a no-auction cell.
All comparisons retain the same market parameters and matched evaluation
paths. The original bundled $H$/anchor and no-cancellation controls are not
part of this canonical matrix.

We report five paired economic effects, as well as raw dense-minus-sparse
validation curves at common observed training budgets. We never forward-fill
stopped runs or replace observed learning curves with a running maximum.
This distinguishes the credit-assignment value of terminal-corrected guidance
from the economic effect of extra shaped preferences. Either effect may be
negative with function approximation and finite training. A separate
same-CLOB no-order counterfactual decomposes direct auction execution value
into price edge net of fees and terminal-inventory risk relief; it is not a
retrained no-auction optimum.''')
    replace('V18 uses a fresh simulation seed\nnamespace (18001);', 'V19 uses a fresh simulation seed\nnamespace (19001);')
    replace('examined in v17, so this is not a new untouched historical holdout.',
            'examined in v17 and v18, so this is not a new untouched historical holdout.')
    anchor = r'\paragraph{Numerical conditioning and replay.}'
    replace(anchor, r'''\paragraph{Forecast reliability.}
Let $H_t^{\mathrm{raw}}$ denote the output of Algorithm
\ref{alg:hyp_clearing_price}. During the CLOB we use the future-close forecast
\[
H_t^{\cl}=S_t^{\mathrm{mid}}+
 w_{b(t)}(H_t^{\mathrm{raw}}-S_t^{\mathrm{mid}}),\qquad
 b(t)=\min\{3,\lfloor4t/\tau^{\mathrm{op}}\rfloor\}.
\]
For each of the four equal time bins, define $x_i=H_i^{\mathrm{raw}}-S_i^{\mathrm{mid}}$
and $y_i=S_i^{\cl}-S_i^{\mathrm{mid}}$ on 256 independent no-order training paths,
and fit $w_b=\operatorname{clip}_{[0,1]}(\sum_i x_i y_i/\sum_i x_i^2)$.
A zero denominator gives zero weight. The fit has no intercept and uses no
validation or test outcomes. The weights are frozen across all policies and
mechanism cells. They are $(0,.190153,.234565,.414256)$ in the synthetic
setting and $(.192784,.314044,.386180,.525631)$ from the pooled historical
training sessions. These are predictive reliability coefficients, not
exchange microstructure parameters. The raw Algorithm 1 recursion and its
exogenous carryover are unchanged. During the auction we retain the original
indicative calculation without this shrinkage. Forecast errors are reported
separately by market phase and horizon; pooled error need not describe the
signal's usefulness earlier in the CLOB. A more accurate forecast need not
improve the economic value of a policy trained on the shaped criterion.

''' + anchor)
    replace('''The potential telescopes and has initial value zero. The deterministic''', r'''The potential telescopes and has initial value zero. In the economic-sparse
auction control, omit $\Delta\widehat Z-\lambda(I-\widehat Z)^2$ during the
auction while retaining the mark-to-mid term, CLOB potential and terminal
correction. The midprice is frozen during this phase, so persistent unchanged
orders receive no action-dependent intermediate potential credit. Dense and
sparse auction conditioning therefore share the same episode objective.
The deterministic''')
    replace('Production checkpoint selection maximizes', r'''DDPG additionally applies LayerNorm with learned affine parameters after
each hidden critic linear layer and before its ReLU activation. Both hidden
layers have width 256; the final scalar value remains unrestricted. The
Stable-Baselines3 policy factory constructs both online and target critics
with this architecture. The actor, single-critic update, actor update at every
step, absence of target smoothing, learning rates and Polyak coefficient are
unchanged. TD3 and SAC retain their specified critic architectures. Thus this
is critic conditioning within native DDPG, without substituting TD3 updates.

Production checkpoint selection maximizes''')
    param_path = Path('paper/results/tables_params/params_generative.tex')
    old_params = param_path.read_text()
    section = prior[prior.index('--- a/paper/results/'):]
    parameters = apply_main_hunks(old_params, section.replace('--- a/paper/results/', '--- a/parameters/'))
    # The weights differ between settings, so describe them in the numerical
    # forecast paragraph rather than mislabeling them as shared simulator inputs.
    patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                       fromfile='a/'+str(main_path), tofile='b/'+str(main_path)))
    patch += ''.join(difflib.unified_diff(old_params.splitlines(True), parameters.splitlines(True),
                                        fromfile='a/'+str(param_path), tofile='b/'+str(param_path)))
    assert apply_main_hunks(original, patch) == text
    Path('docs/manuscript_recommendations_v19.patch').write_text(patch)


if __name__ == '__main__':
    main()
