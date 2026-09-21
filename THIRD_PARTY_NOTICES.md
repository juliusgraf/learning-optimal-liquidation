# License and distribution scope

The author confirmed during release preparation on 2026-09-08 (UTC) that Julius
Graf and Thibaut Mastrolia are the authorized copyright holders, copyright 2026,
and both authorize MIT licensing of first-party software and its documentation.
The existing standard MIT LICENSE is preserved without additional conditions.
Citation is requested separately in CITATION.cff; it is not a license condition.
Manuscript sources/PDF, the top-level development audit, legacy implementations
and v19 documentation history are excluded from this submission branch.

| Material / paths | Basis and scope |
|---|---|
| `src/`, `scripts/`, `configs/`, `tests/`, software documentation, release tooling | First-party MIT, subject to preserved third-party notices. No separately vendored implementation identified in the inspected current tree. |
| `docs/rough_heston_refinement/*.json`, `release/evidence/` | Author confirmed permission to publicly distribute manuscript, figures, derived historical-market summaries and checkpoints. This confirmation does not select a new license for non-code assets. These research assets are excluded from the blanket software MIT grant; no separate reuse license is assigned here. Python analysis scripts remain software. |
| `data/*.csv`, `data/*.meta.json`, `data/raw/**`, `.cache/**` | Vendor data / caches, excluded from the proposed package. Original SIP quote archives require authorized Alpaca access. No right to redistribute raw or processed vendor prices established. Historical Git snapshots also require review; ignore rules do not remove history. |
| `results/**` checkpoints/evaluations/logs | Local originals retained. Author permits checkpoint and derived-result distribution, but the proposed compact bundle contains extracted metadata and numerical summaries, not model weights or raw paths. Original checkpoint hashes identify the local objects; hashes alone do not make them accessible. |
| Local archives and prior Git history | Outside the software/compact-evidence distribution scope. Earlier manuscript, development and vendor-data files are not restored by this notice. |

Installed dependencies are not vendored in this repository or its source archive.
The following licenses were inspected in the local installed distributions; these
are the versions recorded in the v20 runtime files, not invented historical pins.
Distribution wheels can include additional notices for bundled native libraries.

| Dependency | Observed version | License / upstream |
|---|---|---|
| NumPy | 2.4.6 | BSD-3-Clause plus bundled notices (0BSD, MIT, Zlib, CC0); https://numpy.org/ |
| pandas | 3.0.3 | BSD-3-Clause; https://pandas.pydata.org/ |
| SciPy | 1.17.1 | BSD-3-Clause plus bundled notices; https://scipy.org/ |
| PyTorch | 2.12.0 | BSD-3-Clause plus bundled notices; https://pytorch.org/ |
| Stable-Baselines3 | 2.7.1 | MIT; https://github.com/DLR-RM/stable-baselines3 |
| Gymnasium | 1.2.3 | MIT; https://github.com/Farama-Foundation/Gymnasium |
| PyYAML | 6.0.3 | MIT; https://pyyaml.org/ |
| Matplotlib | 3.11.0 | Matplotlib license (PSF-based), bundled notices; https://matplotlib.org/ |
| certifi | 2026.5.20 | MPL-2.0; https://github.com/certifi/python-certifi |

There are no local modifications to dependency distributions proposed here.
MPL-2.0 certifi is an installed dependency, not automatically incompatible with
MIT first-party software. Redistributing a wheelhouse, runtime image, or modified
CA bundle would require preserving its applicable notices/source obligations;
no such dependency bundle is part of this release candidate. CI uses GitHub's
checkout/setup-python Actions; no Actions implementation is vendored.

Sources for standard MIT wording and matching: https://spdx.org/licenses/MIT.html.
This inventory is a scoped distribution review, not an unconditional legal assurance.
