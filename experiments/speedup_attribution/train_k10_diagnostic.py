"""Train a small, explicitly diagnostic K=10 SaTE checkpoint on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from lib.spaceTE.sate_actor import SaTEActor
from lib.spaceTE.sate_model import SaTE

from .benchmark_k10 import _to_sate_data, make_sate_env, validate_checkpoint_dimension
from .benchmark_large_scale import constellation_instance


def train(nodes: int, active_flows: int, samples: int, epochs: int, batch_size: int, output_dir: Path):
    """Use production SaTE loss/model classes while preserving a held-out seed range."""
    if not torch.cuda.is_available():
        return {"status": "BLOCKED_CUDA_NOT_AVAILABLE"}
    if samples < 10:
        raise ValueError("at least ten samples are required for a separated diagnostic split")
    torch.manual_seed(42)
    records = []
    for index in range(samples):
        instance = constellation_instance(nodes, active_flows, 10, seed=1000 + index)
        records.append(_to_sate_data(instance, nodes, data_idx=index))
    reference = constellation_instance(nodes, active_flows, 10, seed=1000)
    work_dir = output_dir / "training"
    env = make_sate_env(reference, nodes, 10, records, penalized=True, work_dir=str(work_dir))
    actor = SaTEActor(env, "EdgeGAT", 0, "linear", "diagnostic-k10", torch.device("cuda:0"))
    actor.init_model()
    model = SaTE(env, actor, lr=0.001, supervised=False, penalized=True, loss="kl_div", early_stop=False)
    start = time.perf_counter()
    model.train(num_epoch=epochs, batch_size=batch_size, num_sample=0, save_model=True, need_topo=False)
    torch.cuda.synchronize()
    training_s = time.perf_counter() - start
    checkpoint = work_dir / "models" / "diagnostic-k10" / f"epoch_{epochs}.pt"
    audit = validate_checkpoint_dimension(checkpoint, 10)
    result = {
        "status": "DIAGNOSTIC_K10_TRAINED" if audit["status"] == "VALID" else audit["status"],
        "evidence_label": "DIAGNOSTIC_K10_TRAINED",
        "checkpoint": str(checkpoint),
        "checkpoint_audit": audit,
        "train_scale": nodes,
        "active_flows": active_flows,
        "samples_total": samples,
        "train_samples": len(env.train_dataset),
        "test_samples": len(env.test_dataset),
        "train_data_indices": sorted(int(row["data_idx"]) for row in env.train_dataset),
        "test_data_indices": sorted(int(row["data_idx"]) for row in env.test_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": "production_penalized_total_flow",
        "hidden_dimension": 128,
        "topognn": "EdgeGAT",
        "allognn_layers": 0,
        "decoder": "linear",
        "training_time_s": training_s,
        "final_training_row": model.losses[-1] if model.losses else None,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "k10_training.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=66)
    parser.add_argument("--active-flows", type=int, default=40)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("output/speedup_attribution/large_scale"))
    args = parser.parse_args()
    result = train(args.nodes, args.active_flows, args.samples, args.epochs, args.batch_size, args.output_dir)
    print(json.dumps({"status": result["status"], "checkpoint": result.get("checkpoint")}, sort_keys=True))


if __name__ == "__main__":
    main()
