from torcherist.config import dataset_dir
from pathlib import Path


def test_dataset_dir():
    assert Path(dataset_dir).exists()
