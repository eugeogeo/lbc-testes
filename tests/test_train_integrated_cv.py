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

from fmd_data import FMDDataset, image_transform, make_fold_loaders
import resnet_baseline
import datasets_loader
import train_integrated_cv


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
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1)
        with torch.no_grad():
            self.conv1.weight.fill_(0.25)
            self.conv1.bias.zero_()
        self.fc = nn.Linear(2, num_classes)

    def forward(self, x):
        return self.fc(self.conv1(x).mean(dim=(2, 3)))


class IntegratedCVTests(unittest.TestCase):
    def test_ten_independent_models_optimizers_and_saved_final_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            for name, value in (("dark", 0), ("light", 255)):
                class_dir = data / name
                class_dir.mkdir(parents=True)
                for i in range(10):
                    Image.new("RGB", (8, 8), (value, value, value)).save(class_dir / f"{i}.png")

            models, optimizers, evaluation_epochs = [], [], []
            original_adam = torch.optim.Adam
            original_evaluate = resnet_baseline.evaluate
            train_steps = []

            def create_model(num_classes):
                model = TinyModel(num_classes)
                torch.testing.assert_close(model.conv1.weight, torch.full_like(model.conv1.weight, 0.25))
                models.append(model)
                return model

            def create_optimizer(parameters, **kwargs):
                optimizer = original_adam(parameters, **kwargs)
                self.assertEqual(len(optimizer.state), 0)
                optimizers.append(optimizer)
                return optimizer

            def record_train(*args, **kwargs):
                train_steps.append(1)
                return original_train(*args, **kwargs)

            def record_evaluate(*args, **kwargs):
                evaluation_epochs.append(len(train_steps))
                return original_evaluate(*args, **kwargs)

            original_train = resnet_baseline.train_epoch
            output = root / "results.json"
            model_dir = root / "models"
            with patch("train_integrated_cv.create_integrated_model", side_effect=create_model), \
                 patch("resnet_baseline.torch.optim.Adam", side_effect=create_optimizer), \
                 patch("resnet_baseline.train_epoch", side_effect=record_train), \
                 patch("resnet_baseline.evaluate", side_effect=record_evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = train_integrated_cv.train_and_evaluate_cv(
                    data, num_epochs=2, batch_size=18, device="cpu",
                    model_dir=model_dir, output=output,
                )

            self.assertEqual(len(models), 10)
            self.assertEqual(len({id(model) for model in models}), 10)
            self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 10)
            self.assertEqual(evaluation_epochs, list(range(2, 21, 2)))
            for model, optimizer in zip(models, optimizers):
                self.assertEqual(
                    {id(p) for group in optimizer.param_groups for p in group["params"]},
                    {id(p) for p in model.parameters()},
                )

            results = json.loads(output.read_text(encoding="utf-8"))
            with Image.open(root / "results_curves.png") as graph:
                self.assertEqual(graph.format, "PNG")
                self.assertGreater(graph.width, 100)
            self.assertEqual(results["epochs"], 2)
            self.assertEqual(results["evaluation"], "final epoch")
            self.assertEqual(len(results["folds"]), 10)
            for fold in results["folds"]:
                self.assertEqual([entry["epoch"] for entry in fold["history"]], [1, 2])
                for entry in fold["history"]:
                    self.assertGreaterEqual(entry["train_loss"], 0)
                    self.assertGreaterEqual(entry["train_accuracy"], 0)
                    self.assertLessEqual(entry["train_accuracy"], 1)
            for metric, mean, std in (
                ("accuracy", summary[0], summary[1]), ("f1_macro", summary[2], summary[3])
            ):
                values = [fold[metric] for fold in results["folds"]]
                self.assertAlmostEqual(mean, np.mean(values))
                self.assertAlmostEqual(std, np.std(values))

            dataset = FMDDataset(data, image_transform())
            for fold, _, val in make_fold_loaders(dataset, batch_size=18):
                state = torch.load(model_dir / f"integrated_resnet_fold_{fold}.pth",
                                   map_location="cpu", weights_only=True)
                for name, tensor in models[fold - 1].state_dict().items():
                    torch.testing.assert_close(state[name], tensor)
                reloaded = TinyModel(2)
                reloaded.load_state_dict(state)
                scores = original_evaluate(reloaded, val, "cpu", 2)
                self.assertEqual(scores["accuracy"], results["folds"][fold - 1]["accuracy"])
                self.assertEqual(scores["f1_macro"], results["folds"][fold - 1]["f1_macro"])

    def test_rejects_non_ten_folds(self):
        with self.assertRaisesRegex(ValueError, "10 folds"):
            train_integrated_cv.train_and_evaluate_cv(num_folds=5)

    def test_training_history_uses_sample_weighted_batch_metrics(self):
        model = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        images = torch.tensor([[1.0], [1.0], [1.0], [-1.0], [-1.0]])
        labels = torch.tensor([0, 0, 1, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        expected_loss = nn.CrossEntropyLoss()(model(images), labels).item()
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
        loss, accuracy = resnet_baseline.train_epoch(
            model, loader, optimizer, nn.CrossEntropyLoss(), "cpu", return_accuracy=True
        )
        self.assertAlmostEqual(loss, expected_loss, places=6)
        self.assertAlmostEqual(accuracy, 3 / 5)


if __name__ == "__main__":
    unittest.main()
