import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

import resnet_baseline
import datasets_loader
import train_integrated_scratch_cv


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

    def forward(self, inputs):
        return self.fc(inputs.mean(dim=(2, 3)))


class IntegratedScratchCVTests(unittest.TestCase):
    def test_cli_writes_histories_checkpoints_and_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            for name, value in (("dark", 0), ("light", 255)):
                directory = data / name
                directory.mkdir(parents=True)
                for index in range(10):
                    Image.new("RGB", (8, 8), (value, value, value)).save(directory / f"{index}.png")
            output, model_dir, graph = root / "result.json", root / "models", root / "curves.png"
            models, optimizers, evaluation_steps = [], [], []
            train_steps = []
            original_adam = torch.optim.Adam
            original_train, original_eval = resnet_baseline.train_epoch, resnet_baseline.evaluate

            def create(num_classes):
                model = TinyModel(num_classes)
                models.append(model)
                return model

            def adam(parameters, **kwargs):
                optimizer = original_adam(parameters, **kwargs)
                self.assertFalse(optimizer.state)
                optimizers.append(optimizer)
                return optimizer

            def train(*args, **kwargs):
                train_steps.append(1)
                return original_train(*args, **kwargs)

            def evaluate(*args, **kwargs):
                evaluation_steps.append(len(train_steps))
                return original_eval(*args, **kwargs)

            args = [
                "train_integrated_scratch_cv.py", "--data-dir", str(data), "--epochs", "2",
                "--batch-size", "18", "--device", "cpu", "--output", str(output),
                "--model-dir", str(model_dir), "--plot-output", str(graph),
            ]
            with patch("sys.argv", args), \
                 patch("train_integrated_scratch_cv.create_integrated_model_scratch", side_effect=create), \
                 patch("resnet_baseline.torch.optim.Adam", side_effect=adam), \
                 patch("resnet_baseline.train_epoch", side_effect=train), \
                 patch("resnet_baseline.evaluate", side_effect=evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_integrated_scratch_cv.main()
            results = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(results["weights"])
            self.assertEqual(results["evaluation"], "final epoch")
            self.assertIn("from scratch", results["experiment"])
            self.assertEqual(len(models), 10)
            self.assertEqual(len({id(model) for model in models}), 10)
            self.assertEqual(len({id(opt) for opt in optimizers}), 10)
            self.assertEqual(evaluation_steps, list(range(2, 21, 2)))
            for fold in results["folds"]:
                self.assertEqual([x["epoch"] for x in fold["history"]], [1, 2])
                state = torch.load(model_dir / f"integrated_resnet_fold_{fold['fold']}.pth",
                                   map_location="cpu", weights_only=True)
                for name, tensor in models[fold["fold"] - 1].state_dict().items():
                    torch.testing.assert_close(state[name], tensor)
            with Image.open(graph) as image:
                self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
