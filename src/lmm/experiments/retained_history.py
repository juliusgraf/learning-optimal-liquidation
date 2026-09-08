"""Bind reused historical results to their previously published bytes and sources.

This is a selective campaign replacement, not a claim that old runs used the
new source revision. Only the explicitly inventoried historical runs qualify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASELINE = Path("docs/rough_heston_refinement/v20_replacement_baseline.json")
STATE = Path("_provenance/synthetic_replacement.json")
MESH = Path("_provenance/synthetic_refinement.yaml")
MESH_SOURCE = Path("configs/campaign/v20_synthetic_refinement.yaml")
ARCHIVE = Path("_superseded/synthetic_refinement_v1")
HISTORICAL = "historical_sp500_midquotes"
ROOT_NAMES = ("revision_v20", "revision_v20_cashflow", "revision_v20_economic_dense_h")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(directory):
    """Bind paths and all file contents, including logs omitted by completion manifests."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"expected real directory: {directory}")
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink not permitted in retained artifacts: {path}")
        if path.is_file():
            entries.append((path.relative_to(directory).as_posix(), digest(path)))
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode()
    ).hexdigest()


class RetainedHistory:
    def __init__(self, repo, root, baseline, state):
        self.repo, self.root = Path(repo).resolve(), Path(root).resolve()
        self.baseline, self.state = baseline, state
        self._validated = set()
        self._fit_validated = False

    @classmethod
    def load(cls, repo, root):
        repo, root = Path(repo).resolve(), Path(root).resolve()
        baseline = json.loads((repo / BASELINE).read_text())
        state = json.loads((root / STATE).read_text())
        if (
            baseline.get("schema") != "v20-synthetic-replacement-baseline-v1"
            or state.get("schema") != "v20-synthetic-replacement-v1"
            or root != repo / "results/revision_v20"
            or state.get("baseline_sha256") != digest(repo / BASELINE)
            or state.get("mesh_sha256") != digest(repo / MESH_SOURCE)
            or digest(root / MESH) != state["mesh_sha256"]
        ):
            raise ValueError("synthetic replacement provenance/config mismatch")
        return cls(repo, root, baseline, state)

    def validate_run(self, path):
        path = Path(path).resolve()
        try:
            name = path.relative_to(self.repo).as_posix()
        except ValueError:
            raise ValueError("retained run is outside repository") from None
        expected = self.baseline["historical_runs"].get(name)
        if expected is None or path.parent != self.root / HISTORICAL:
            raise ValueError(
                f"run is not an authorized retained historical result: {path}"
            )
        if name not in self._validated:
            if tree_digest(path) != expected["tree_sha256"]:
                raise ValueError(f"retained historical artifacts changed: {path}")
            self._validated.add(name)
        return expected["git_revision"]

    def forecast_overlay(self):
        path = self.root / "_forecasts" / HISTORICAL
        if not self._fit_validated:
            if tree_digest(path) != self.baseline["historical_forecast_tree_sha256"]:
                raise ValueError("retained historical forecast artifacts changed")
            self._fit_validated = True
        return path / "forecast_overlay.yaml"

    def revalidate(self):
        self._validated.clear()
        self._fit_validated = False
        observed = {
            p.parent.relative_to(self.repo).as_posix()
            for p in (self.root / HISTORICAL).glob("*/config_resolved.yaml")
        }
        if observed != set(self.baseline["historical_runs"]):
            raise ValueError("retained historical inventory changed")
        for name in observed:
            self.validate_run(self.repo / name)
        self.forecast_overlay()

    @property
    def disclosure(self):
        return (
            "Synthetic v20 results were replaced after retraining at a maximum internal "
            "rough-Heston step of 0.25 minutes, with a fresh training-only forecast fit, "
            "normalization and reference conditioning. This mesh is a selected numerical "
            "setting, not an established accuracy threshold. The 200 historical runs "
            "and their forecast fit retain their original bytes and recorded source "
            "revisions; they were not retrained or relabeled as new-source runs. Their "
            "retention is bound to the checked-in pre-replacement inventory and the "
            "previously published v20 report. The original source disclosure was: "
            + self.baseline["prior_report_source_attestation"]["disclosure"]
        )
