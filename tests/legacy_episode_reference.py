"""Exact, same-runtime comparison with the frozen pre-refinement simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


REFERENCE = Path(__file__).resolve().parent / "fixtures" / "pre_refinement"
MANIFEST_SHA256 = "6edbf7f0a3750401d6a0583f33948928a98f7a5894b506f214565fc5436696ed"


def validate_reference(root: Path) -> None:
    manifest_bytes = (root / "provenance.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen reference manifest changed; review its original source")
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        path = root / row["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("frozen reference source must stay inside the fixture")
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"frozen reference source changed: {row['path']}")


def capture_episode(root: Path) -> dict:
    # Imports are delayed so the child can select the frozen package first.
    from lmm.config import load_config
    from lmm.env.action_spaces import AuctionAction, ClobAction
    from lmm.env.mdp import make_env

    cfg = load_config(root / "configs/base.yaml", root / "configs/synthetic_rough_heston.yaml")
    env = make_env(cfg, repo_root=root)
    env.reset(seed=4201)
    rows = []
    while True:
        action = ClobAction(0.0, 0) if env.phase == "clob" else AuctionAction(0.0, 0, 0)
        obs, reward, done, _, info = env.step(action)
        rows.append([obs.tolist(), reward, info["t"], info["H_used"], info["H_next"]])
        if done:
            break
    result = {"rows": rows, "rng_state": env.np_random.bit_generator.state}
    env.close()
    return result


def reference_episode() -> dict:
    validate_reference(REFERENCE)
    output = subprocess.check_output(
        [sys.executable, "-I", str(Path(__file__).resolve()), str(REFERENCE)],
        cwd=REFERENCE,
        text=True,
        timeout=30,
    )
    return json.loads(output)


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    validate_reference(root)
    sys.path.insert(0, str(root / "src"))
    import lmm

    if Path(lmm.__file__).resolve() != root / "src/lmm/__init__.py":
        raise RuntimeError("reference subprocess imported the current simulator")
    print(json.dumps(capture_episode(root), sort_keys=True, allow_nan=False))
