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
from torchvision.models import resnet18

from fmd_data import FMDDataset, image_transform, make_fold_loaders
import resnet_baseline
import datasets_loader
import train_resnet_scratch_cv


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


class ScratchCVTests(unittest.TestCase):
    def test_scratch_model_uses_no_weights_and_adapts_fc(self):
        model = resnet18(weights=None)
        original_conv = model.conv1.weight.detach().clone()
        with patch("train_resnet_scratch_cv.resnet18", return_value=model) as constructor:
            result = train_resnet_scratch_cv.create_scratch_model(10)
        constructor.assert_called_once_with(weights=None)
        self.assertEqual(result.fc.in_features, 512)
        self.assertEqual(result.fc.out_features, 10)
        self.assertTrue(all(parameter.requires_grad for parameter in result.parameters()))
        torch.testing.assert_close(result.conv1.weight, original_conv)

    def test_ten_folds_fresh_models_and_final_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            for name, value in (("dark", 0), ("light", 255)):
                class_dir = data / name
                class_dir.mkdir(parents=True)
                for index in range(10):
                    Image.new("RGB", (8, 8), (value, value, value)).save(class_dir / f"{index}.png")
            models, optimizers, eval_steps = [], [], []
            train_steps = []
            original_adam = torch.optim.Adam
            original_train = resnet_baseline.train_epoch
            original_eval = resnet_baseline.evaluate

            def create(num_classes):
                model = TinyModel(num_classes)
                models.append(model)
                return model

            def adam(parameters, **kwargs):
                optimizer = original_adam(parameters, **kwargs)
                self.assertEqual(len(optimizer.state), 0)
                optimizers.append(optimizer)
                return optimizer

            def train(*args, **kwargs):
                train_steps.append(1)
                return original_train(*args, **kwargs)

            def evaluate(*args, **kwargs):
                eval_steps.append(len(train_steps))
                return original_eval(*args, **kwargs)

            output, model_dir = root / "result.json", root / "models"
            with patch("train_resnet_scratch_cv.create_scratch_model", side_effect=create), \
                 patch("resnet_baseline.torch.optim.Adam", side_effect=adam), \
                 patch("resnet_baseline.train_epoch", side_effect=train), \
                 patch("resnet_baseline.evaluate", side_effect=evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                results = train_resnet_scratch_cv.train_and_evaluate_cv(
                    data, epochs=2, batch_size=18, device="cpu", output=output, model_dir=model_dir
                )
            self.assertEqual(results, json.loads(output.read_text(encoding="utf-8")))
            self.assertIsNone(results["weights"])
            self.assertEqual(results["evaluation"], "final epoch")
            self.assertEqual(len({id(model) for model in models}), 10)
            self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 10)
            self.assertEqual(eval_steps, list(range(2, 21, 2)))
            for model, optimizer in zip(models, optimizers):
                self.assertEqual(
                    {id(p) for group in optimizer.param_groups for p in group["params"]},
                    {id(p) for p in model.parameters()},
                )
            for metric in ("accuracy", "f1_macro"):
                values = [fold[metric] for fold in results["folds"]]
                self.assertEqual(results["summary"][metric]["mean"], float(np.mean(values)))
                self.assertEqual(results["summary"][metric]["std"], float(np.std(values)))

            dataset = FMDDataset(data, image_transform())
            for fold, _, loader in make_fold_loaders(dataset, batch_size=18):
                state = torch.load(model_dir / f"resnet_scratch_fold_{fold}.pth",
                                   map_location="cpu", weights_only=True)
                for name, tensor in models[fold - 1].state_dict().items():
                    torch.testing.assert_close(state[name], tensor)
                model = TinyModel(2)
                model.load_state_dict(state)
                scores = original_eval(model, loader, "cpu", 2)
                self.assertEqual(scores["accuracy"], results["folds"][fold - 1]["accuracy"])
                self.assertEqual(scores["f1_macro"], results["folds"][fold - 1]["f1_macro"])


if __name__ == "__main__":
    unittest.main()
