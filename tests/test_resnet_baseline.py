import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights

import datasets_loader
import resnet_baseline


def setUpModule():
    global _fmd_patch
    _fmd_patch = patch.dict(
        datasets_loader.DATASETS,
        {"fmd": {"default_dir": None, "num_classes": 2, "num_images": 20, "num_folds": 10}},
    )
    _fmd_patch.start()


def tearDownModule():
    _fmd_patch.stop()


class TinyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, images):
        return self.fc(images.mean(dim=(2, 3)))


class ResNetBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        for class_name, value in (("dark", 0), ("light", 255)):
            directory = self.root / class_name
            directory.mkdir()
            for index in range(10):
                Image.new("RGB", (8, 8), (value, value, value)).save(directory / f"{index}.png")

    def test_create_model_uses_pretrained_weights_and_replaces_fc(self):
        model = TinyModel(1000)
        with patch("resnet_baseline.resnet18", return_value=model) as constructor:
            result = resnet_baseline.create_model(10)
        constructor.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.assertEqual(result.fc.in_features, 3)
        self.assertEqual(result.fc.out_features, 10)
        self.assertTrue(all(parameter.requires_grad for parameter in result.parameters()))

    def test_train_updates_weights_and_evaluation_uses_inference_mode(self):
        class TrackingModel(TinyModel):
            def forward(self, images):
                self.last_training = self.training
                self.last_grad_enabled = torch.is_grad_enabled()
                return super().forward(images)

        model = TrackingModel(2)
        images = torch.ones((4, 3, 4, 4))
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.fc.weight.detach().clone()
        loss = resnet_baseline.train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), "cpu")
        self.assertGreater(loss, 0)
        self.assertTrue(model.last_training)
        self.assertTrue(model.last_grad_enabled)
        self.assertFalse(torch.equal(before, model.fc.weight))
        metrics = resnet_baseline.evaluate(model, loader, "cpu", 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_macro"], 1 / 3)
        self.assertFalse(model.last_training)
        self.assertFalse(model.last_grad_enabled)

    def test_ten_folds_fresh_models_reproducibility_and_json(self):
        created = []

        def create_model(num_classes):
            model = TinyModel(num_classes)
            created.append(model)
            return model

        output = self.root / "scores.json"
        arguments = [
            "resnet_baseline.py", "--data-dir", str(self.root), "--epochs", "1",
            "--batch-size", "18", "--device", "cpu", "--output", str(output),
        ]
        with patch("resnet_baseline.create_model", side_effect=create_model):
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                resnet_baseline.main()
            results = json.loads(output.read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                repeated = resnet_baseline.run_experiment(
                    self.root, epochs=1, batch_size=18, device="cpu"
                )
        self.assertEqual(results, repeated)
        self.assertEqual(len(created), 20)
        self.assertEqual(len({id(model) for model in created}), 20)
        self.assertEqual([fold["fold"] for fold in results["folds"]], list(range(1, 11)))
        for metric in ("accuracy", "f1_macro"):
            values = [fold[metric] for fold in results["folds"]]
            self.assertAlmostEqual(results["summary"][metric]["mean"], np.mean(values))
            self.assertAlmostEqual(results["summary"][metric]["std"], np.std(values))

    def test_rejects_invalid_hyperparameters(self):
        for kwargs in ({"epochs": 0}, {"batch_size": 0}, {"num_workers": -1},
                       {"learning_rate": 0}, {"learning_rate": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resnet_baseline.run_experiment(self.root, **kwargs)


if __name__ == "__main__":
    unittest.main()
