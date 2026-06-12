import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from oarlvla.models.collate import vla_collate_fn
from oarlvla.models.datasets import SyntheticVLADataset
from oarlvla.models.losses import compute_vla_loss
from oarlvla.models.trainer import make_optimizer, model_inputs
from oarlvla.models.vla_model import OARLVLAConfig, OARLVLAModel

from model_test_utils import write_tiny_model_dataset


def test_model_training_step_backward(tmp_path):
    dataset = SyntheticVLADataset(write_tiny_model_dataset(tmp_path / "tiny.jsonl", 4))
    batch = vla_collate_fn([dataset[i] for i in range(4)])
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=32,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
        dropout=0.0,
    )
    model = OARLVLAModel(config)
    optimizer = make_optimizer(model, lr=1e-3)
    outputs = model(**model_inputs(batch))
    loss, metrics = compute_vla_loss(outputs, batch)
    loss.backward()
    optimizer.step()
    assert metrics["loss"] > 0
    assert any(param.grad is not None for param in model.parameters())


def test_tiny_overfit_script_reaches_high_accuracy():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/overfit_tiny_batch.py",
            "--steps",
            "40",
            "--num-samples",
            "8",
            "--hidden-dim",
            "64",
            "--output",
            "checkpoints/test_overfit.pt",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Final tiny-batch target accuracy: 1.00" in proc.stdout

