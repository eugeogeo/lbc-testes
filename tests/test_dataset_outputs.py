import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from torch import nn

import datasets_loader
import resnet_baseline
import train_integrated_cv
import train_integrated_scratch_cv
import train_resnet_scratch_cv


class TinyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, images):
        return self.fc(images.mean(dim=(2, 3)))


class DatasetOutputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "kth"
        for cls, value in (("dark", 0), ("light", 255)):
            for letter in "abcd":
                for i in range(2):
                    path = self.data / cls / f"1{letter}-scale_{i}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), (value, value, value)).save(path)
        patcher = patch.dict(datasets_loader.DATASETS, {
            "kth": {"default_dir": self.data, "num_classes": 2, "num_images": 16, "num_folds": 4},
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.work = self.root / "work"
        self.work.mkdir()
        previous = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, previous)

    def check(self, json_path, checkpoints, prefix):
        results = json.loads(Path(json_path).read_text())
        self.assertEqual(results["dataset"], "kth")
        self.assertEqual(results["num_folds"], 4)
        self.assertEqual([f["fold"] for f in results["folds"]], [1, 2, 3, 4])
        if checkpoints is not None:
            self.assertEqual(len(list(Path(checkpoints).glob(f"{prefix}_fold_*.pth"))), 4)

    def test_resnet_baseline_cli_names_output_with_dataset(self):
        argv = ["resnet_baseline.py", "--dataset", "kth", "--epochs", "1", "--batch-size", "8",
                "--device", "cpu"]
        with patch.object(sys, "argv", argv), \
             patch("resnet_baseline.create_model", TinyModel), \
             contextlib.redirect_stdout(io.StringIO()):
            resnet_baseline.main()
        self.check("results/resnet18_kth.json", None, "")

    def test_scratch_resnet_defaults_include_dataset(self):
        with patch("train_resnet_scratch_cv.create_scratch_model", TinyModel), \
             contextlib.redirect_stdout(io.StringIO()):
            train_resnet_scratch_cv.train_and_evaluate_cv(
                epochs=1, batch_size=8, device="cpu", dataset_name="kth")
        self.check("results/resnet_scratch_kth.json", "saved_models/resnet_scratch_kth",
                   "resnet_scratch")

    def test_integrated_defaults_include_dataset(self):
        with patch("train_integrated_cv.create_integrated_model", TinyModel), \
             contextlib.redirect_stdout(io.StringIO()):
            train_integrated_cv.train_and_evaluate_cv(
                num_epochs=1, batch_size=8, device="cpu", dataset_name="kth")
        self.check("results/integrated_kth.json", "saved_models/integrated_kth",
                   "integrated_resnet")
        self.assertTrue(Path("results/integrated_kth_curves.png").is_file())

    def test_integrated_scratch_cli_matches_requested_naming(self):
        argv = ["train_integrated_scratch_cv.py", "--dataset", "kth", "--epochs", "1",
                "--batch-size", "8", "--device", "cpu"]
        with patch.object(sys, "argv", argv), \
             patch("train_integrated_scratch_cv.create_integrated_model_scratch", TinyModel), \
             contextlib.redirect_stdout(io.StringIO()):
            train_integrated_scratch_cv.main()
        self.check("results/integrated_scratch_kth.json", "saved_models/integrated_scratch_kth",
                   "integrated_resnet")
        self.assertTrue(Path("results/integrated_scratch_kth_curves.png").is_file())

    def test_incompatible_fold_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "4 folds"), contextlib.redirect_stdout(io.StringIO()):
            resnet_baseline.run_experiment(dataset_name="kth", num_folds=10, epochs=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
