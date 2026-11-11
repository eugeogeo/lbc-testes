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

import plot_confusion
import plot_filters


class VisualizationTests(unittest.TestCase):
    def test_selects_lowest_fold_on_tied_accuracy(self):
        results = {"folds": [
            {"fold": 3, "accuracy": 0.8},
            {"fold": 2, "accuracy": 0.8},
            {"fold": 1, "accuracy": 0.7},
        ]}
        self.assertEqual(plot_filters.best_fold(results)["fold"], 2)

    def test_filter_normalization_uses_shared_signed_scale(self):
        initial = torch.zeros(64, 3, 7, 7)
        trained = initial.clone()
        initial[0, 0, 0, 0] = 1
        trained[0, 0, 0, 0] = 2
        trained[0, 1, 0, 0] = -2
        first, second = plot_filters.normalize_filter_pair(initial, trained)
        self.assertEqual(first.shape, (64, 7, 7, 3))
        self.assertEqual(first[0, 0, 0, 0], 0.75)
        self.assertEqual(second[0, 0, 0, 0], 1)
        self.assertEqual(second[0, 0, 0, 1], 0)
        self.assertEqual(first[0, 0, 0, 1], 0.5)
        with self.assertRaises(ValueError):
            plot_filters.normalize_filter_pair(initial[:2], trained)
        trained[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            plot_filters.normalize_filter_pair(initial, trained)

    def test_grid_written_at_300_dpi(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "grid.png"
            plot_filters.save_grid(np.full((64, 7, 7, 3), 0.5), "Teste", path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (3000, 3000))
                self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.1)

    def test_experiment_validation(self):
        experiment = {
            "experiment": "Integrated ResNet-18 Gabor and LBP-inspired from scratch",
            "weights": None,
            "class_to_idx": {"a": 0, "b": 1},
            "folds": [{"fold": i, "accuracy": 0.5} for i in range(1, 11)],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.json"
            path.write_text(json.dumps(experiment), encoding="utf-8")
            self.assertEqual(plot_filters.load_experiment(path), experiment)
            experiment["folds"][0]["accuracy"] = 1.1
            path.write_text(json.dumps(experiment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Acurácia inválida"):
                plot_filters.load_experiment(path)

    def test_validation_predictions_use_eval_and_inference_mode(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                self.checks = (self.training, torch.is_grad_enabled())
                return inputs

        model = Model()
        loader = [(torch.tensor([[2., 1.], [1., 3.]]), torch.tensor([0, 1]))]
        targets, predictions = plot_confusion.validation_predictions(model, loader, "cpu")
        self.assertEqual(targets, [0, 1])
        self.assertEqual(predictions, [0, 1])
        self.assertEqual(model.checks, (False, False))

    def test_confusion_rejects_mismatched_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "wrong").mkdir()
            Image.new("RGB", (4, 4)).save(root / "wrong" / "sample.png")
            result = {
                "experiment": "Integrated ResNet-18 Gabor and LBP-inspired from scratch",
                "weights": None, "class_to_idx": {"a": 0, "b": 1},
                "num_images": 1,
                "batch_size": 4, "seed": 42,
                "folds": [{"fold": i, "accuracy": 0.5} for i in range(1, 11)],
            }
            path = root / "results.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            with patch("sys.argv", ["plot_confusion.py", "--results", str(path),
                                    "--data-dir", str(root)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "esperado: 10 classes"):
                    plot_confusion.main()


if __name__ == "__main__":
    unittest.main()
