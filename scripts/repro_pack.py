#!/usr/bin/env python3
"""Phase 1 reproducibility pack runner.

Runs the canonical data/feature/scoring pipeline and writes a reproducibility
manifest with hashes + provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "repro_manifest.json"


@dataclass
class StepResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = sum(1 for _ in reader)
    return max(0, rows - 1)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        return out.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return None


def _run_step(name: str, command: list[str], allow_failure: bool = False) -> StepResult:
    print(f"\n[STEP] {name}")
    print("$ " + " ".join(command))
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())

    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Step failed ({name}) with exit code {proc.returncode}")

    return StepResult(
        name=name,
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _must_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs:\n" + "\n".join(missing))


def _file_meta(path: Path) -> dict:
    meta = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix.lower() == ".csv":
        meta["row_count"] = _csv_row_count(path)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 reproducibility pack.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child script runs (default: current interpreter).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python

    generated_at = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    step_results: list[StepResult] = []

    # A) Mechanistic feature build
    step_results.append(
        _run_step(
            "Process Crumb text",
            [py, "scripts/process_mechanistic_multichannel_crumb_text.py", "--force"],
        )
    )

    chembl_mode = "diagnose"
    chembl_step = _run_step(
        "Fetch ChEMBL multichannel (--diagnose)",
        [py, "scripts/fetch_mechanistic_multichannel_chembl.py", "--diagnose"],
        allow_failure=True,
    )
    step_results.append(chembl_step)
    if chembl_step.returncode != 0:
        chembl_mode = "standard_fallback"
        step_results.append(
            _run_step(
                "Fetch ChEMBL multichannel fallback",
                [py, "scripts/fetch_mechanistic_multichannel_chembl.py"],
            )
        )

    step_results.append(
        _run_step(
            "Build multichannel features",
            [py, "scripts/build_mechanistic_multichannel_features.py", "--force", "--print-summary"],
        )
    )

    # B) Mechanistic plausibility
    step_results.append(
        _run_step(
            "Score mechanistic plausibility",
            [py, "scripts/score_mech_plausibility.py"],
        )
    )

    # C) Trust policy
    step_results.append(
        _run_step(
            "Run trust policy",
            [py, "scripts/run_trust_policy.py"],
        )
    )
    step_results.append(
        _run_step(
            "Run trust policy + mech",
            [py, "scripts/run_trust_policy_mech.py"],
        )
    )

    required_outputs = [
        ROOT / "data/processed/mechanistic_multichannel_features_cipa28.csv",
        ROOT / "results/mechanistic_multichannel_feature_join_summary.json",
        ROOT / "results/mechanistic_multichannel_concordance.json",
        ROOT / "reports/mechanistic_multichannel_concordance.md",
        ROOT / "results/mechanistic_multichannel_crumb_text_join_summary.json",
        ROOT / "results/mech_plausibility_scores.csv",
        ROOT / "results/mech_plausibility_summary.json",
        ROOT / "results/abstention_trust_policy_curve.csv",
        ROOT / "results/abstention_trust_policy_summary.json",
        ROOT / "results/abstention_trust_policy_mech_curve.csv",
        ROOT / "results/abstention_trust_policy_mech_summary.json",
        ROOT / "results/trust_policy_scores.csv",
        ROOT / "results/trust_policy_scores_with_mech.csv",
    ]
    _must_exist(required_outputs)

    predictions_path = ROOT / "results/calibration_predictions.csv"
    predictions_meta = None
    if predictions_path.exists():
        predictions_meta = _file_meta(predictions_path)

    manifest = {
        "generated_at_utc": generated_at,
        "git": {
            "commit_hash": _git_commit(),
            "dirty": _git_dirty(),
        },
        "python_version": sys.version,
        "runner_python": py,
        "chembl_fetch_mode": chembl_mode,
        "inputs": {
            "predictions_path": str(predictions_path),
            "predictions_meta": predictions_meta,
        },
        "steps": [
            {
                "name": s.name,
                "command": s.command,
                "returncode": s.returncode,
            }
            for s in step_results
        ],
        "outputs": {
            str(path): _file_meta(path)
            for path in required_outputs
        },
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nrepro_manifest.json written: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
