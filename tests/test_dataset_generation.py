import numpy as np
from pathlib import Path

from scripts.generate_dataset import generate_dataset


def test_generate_dataset_creates_dataset(tmp_path):
    config_path = Path("configs/example_config.json")
    output_path = tmp_path / "example_dataset.npz"
    generated = generate_dataset(config_path, output_path, seed=0)
    assert generated == output_path

    loaded = np.load(output_path)
    required_keys = {"traits", "node_loglik", "parents", "branch_lengths", "param_shape", "init_params"}
    assert required_keys.issubset(loaded.files)

    traits = loaded["traits"]
    node_loglik = loaded["node_loglik"]
    assert traits.ndim == 2
    assert node_loglik.shape[0] == traits.shape[0]
    assert node_loglik.shape[1] == loaded["param_shape"][3]
    assert loaded["param_shape"][0] + loaded["param_shape"][1] + loaded["param_shape"][2] + loaded["param_shape"][3] == loaded["init_params"].shape[0]
