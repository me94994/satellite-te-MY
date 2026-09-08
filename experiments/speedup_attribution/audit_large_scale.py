"""Capture environment, local-data provenance, and K=10 checkpoint availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Dict

import torch


OFFICIAL_DATA_URL = "https://drive.google.com/drive/folders/1h6kbOj4HpqofPNd7lkIJDTut4XF4ipAF?usp=sharing"


def audit_checkpoints(root: Path) -> Dict[str, object]:
    """Inspect dimensions only; never resize or reinterpret an old checkpoint."""
    shapes: Dict[str, int] = {}
    k10 = []
    errors = []
    files = [
        path for path in (root / "output").rglob("*.pt")
        if "speedup_attribution/large_scale" not in path.as_posix()
    ] if (root / "output").is_dir() else []
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            weight = payload.get("mean_linear.weight") if isinstance(payload, dict) else None
            if weight is None:
                continue
            shape = "x".join(str(value) for value in weight.shape)
            shapes[shape] = shapes.get(shape, 0) + 1
            if list(weight.shape) == [10, 10]:
                k10.append(str(path.relative_to(root)))
        except Exception as exc:
            errors.append({"path": str(path.relative_to(root)), "error": str(exc).splitlines()[0]})
    diagnostic_metadata = root / "output/speedup_attribution/large_scale/k10_training.json"
    diagnostic = json.loads(diagnostic_metadata.read_text(encoding="utf-8")) if diagnostic_metadata.is_file() else None
    return {
        "status": "K10_CHECKPOINT_AVAILABLE" if k10 else "PAPER_K10_ARTIFACT_NOT_AVAILABLE",
        "checkpoint_files_scanned": len(files),
        "mean_linear_weight_shapes": shapes,
        "k10_checkpoints": k10,
        "diagnostic_k10_training": None if diagnostic is None else {
            "status": diagnostic.get("status"),
            "checkpoint": diagnostic.get("checkpoint"),
            "mean_linear_weight_shape": diagnostic.get("checkpoint_audit", {}).get("mean_linear_weight_shape"),
        },
        "load_errors": errors,
        "validation_rule": "mean_linear.weight == [10,10] and training metadata must confirm num_path=10",
    }


def write_audits(repo: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        import gurobipy as gp
        gurobi = {"available": True, "version": list(gp.gurobi.version()), "license_policy": "RESTRICTED_LICENSE_SAFE_LIMIT_1900"}
    except Exception as exc:
        gurobi = {"available": False, "error": str(exc).splitlines()[0]}
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gurobi": gurobi,
        "threads": 1,
        "configuration": "CODE_CONFIG",
    }
    input_files = [str(path.relative_to(repo)) for path in (repo / "input").rglob("*") if path.is_file()] if (repo / "input").is_dir() else []
    provenance = {
        "official_artifact_url": OFFICIAL_DATA_URL,
        "local_input_files": input_files[:100],
        "local_input_file_count": len(input_files),
        "paper_faithful_k10_data": "NOT_AVAILABLE" if not input_files else "NOT_VALIDATED",
        "download_status": "NOT_DOWNLOADED_NO_MINIMAL_K10_FILE_IDENTIFIED",
        "synthetic_evidence_label": "DIAGNOSTIC_SCALE",
        "controlled_stress_label": "CONTROLLED_STRESS_DIAGNOSTIC",
        "paper_workload_claim": False,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "data_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "k10_audit.json").write_text(json.dumps(audit_checkpoints(repo), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/large_scale"))
    args = parser.parse_args()
    write_audits(args.repo.resolve(), args.output)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
