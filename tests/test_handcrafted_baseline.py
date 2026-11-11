import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from fmd_data import FMDDataset, make_fold_loaders
from handcrafted_baseline import evaluate_fold, extract_features, run_experiment


class HandcraftedTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        for class_name in ("plain", "stripes"):
            class_dir = self.root / class_name
            class_dir.mkdir()
            pixels = np.full((16, 16), 100, dtype=np.uint8)
            if class_name == "stripes":
                pixels[:, ::2] = 255
            for index in range(10):
                Image.fromarray(pixels).save(class_dir / f"{index:02}.png")

    def test_constant_image_features(self):
        features = extract_features(Image.new("RGB", (16, 16), (0, 0, 0)))
        self.assertEqual(features.shape, (14,))
        self.assertAlmostEqual(features[:10].sum(), 1.0)
        np.testing.assert_allclose(features[10:], [0.0, 1.0, 1.0, 1.0])

    def test_scaler_fits_only_training_features_and_cache_reuses_images(self):
        dataset = FMDDataset(self.root, transforms.Resize((224, 224)))
        folds = list(make_fold_loaders(dataset))
        sizes = []

        class TrackingScaler(StandardScaler):
            def fit(self, X, y=None, **kwargs):
                sizes.append(len(X))
                return super().fit(X, y, **kwargs)

        cache = {}
        with patch("handcrafted_baseline.StandardScaler", TrackingScaler):
            with patch("handcrafted_baseline.extract_features", wraps=extract_features) as extractor:
                for _, train, val in folds:
                    metrics = evaluate_fold(train.dataset, val.dataset, cache)
                    self.assertEqual(metrics, {"accuracy": 1.0, "f1_macro": 1.0})
                self.assertEqual(extractor.call_count, len(dataset))
        self.assertEqual(sizes, [18] * 10)

    def test_experiment_summary_and_cli_output(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_experiment(self.root)
        self.assertEqual(len(results["folds"]), 10)
        for name in ("accuracy", "f1_macro"):
            values = [fold[name] for fold in results["folds"]]
            self.assertAlmostEqual(results["summary"][name]["mean"], np.mean(values))
            self.assertAlmostEqual(results["summary"][name]["std"], np.std(values, ddof=0))
        self.assertEqual(results["class_to_idx"], {"plain": 0, "stripes": 1})
        output = self.root / "output" / "scores.json"
        script = Path(__file__).resolve().parents[1] / "handcrafted_baseline.py"
        command = [sys.executable, str(script), "--data-dir", str(self.root), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIn("Fold 10:", completed.stdout)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), results)

    def test_rejects_single_class(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "only").mkdir()
            Image.new("RGB", (2, 2)).save(Path(root) / "only" / "image.png")
            with self.assertRaisesRegex(ValueError, "pelo menos duas classes"):
                run_experiment(root)


if __name__ == "__main__":
    unittest.main()
