import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import datasets_loader


def save(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


def spec(classes, images, folds):
    return {"default_dir": None, "num_classes": classes, "num_images": images, "num_folds": folds}


def indices(loader):
    return sorted(loader.dataset.indices)


class DatasetsLoaderTests(unittest.TestCase):
    def test_fmd_is_stratified_10_fold_with_fixed_seed(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(datasets_loader.DATASETS, {"fmd": spec(3, 30, 10)}):
            for c in range(3):
                for i in range(10):
                    save(Path(temp) / f"class{c}" / f"{i}.png")
            _, num_classes, folds = datasets_loader.get_dataset_and_loaders("fmd", temp, 4, 10, 42)
            _, _, again = datasets_loader.get_dataset_and_loaders("fmd", temp, 4, None, 42)
        self.assertEqual(num_classes, 3)
        self.assertEqual([f[0] for f in folds], list(range(1, 11)))
        for (_, train, val), (_, train2, val2) in zip(folds, again):
            self.assertEqual(indices(val), indices(val2))
            self.assertEqual(indices(train), indices(train2))
            self.assertEqual(len(val.dataset), 3)

    def test_kth_uses_one_physical_sample_per_fold(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(datasets_loader.DATASETS, {"kth": spec(2, 16, 4)}):
            for cls, number in (("cork", 16), ("wool", 22)):
                for letter in "abcd":
                    for i in range(1):
                        save(Path(temp) / cls / f"{number}{letter}-scale_{i}_col.png")
                        save(Path(temp) / cls / f"{number}{letter}-scale_{i + 5}_col.png")
            dataset, num_classes, folds = datasets_loader.get_dataset_and_loaders("kth", temp)
            self.assertEqual(num_classes, 2)
            self.assertEqual(len(folds), 4)
            for (fold, train, val), letter in zip(folds, "abcd"):
                val_names = [dataset.samples[i][0].name for i in indices(val)]
                train_names = [dataset.samples[i][0].name for i in indices(train)]
                self.assertTrue(all(f"{letter}-" in n for n in val_names), fold)
                self.assertFalse(any(f"{letter}-" in n for n in train_names), fold)
                self.assertEqual(len(val.dataset), 4)
                self.assertEqual({dataset.targets[i] for i in indices(val)}, {0, 1})
                self.assertEqual(len(train.dataset), 12)
            with self.assertRaisesRegex(ValueError, "4 folds"):
                datasets_loader.get_dataset_and_loaders("kth", temp, num_folds=10)

    def make_dtd(self, root, missing_val=False, overlap=False):
        names = [f"{c}/{c}_{i}.jpg" for c in ("banded", "braided") for i in range(10)]
        for name in names:
            save(root / "images" / name)
        labels = root / "labels"
        labels.mkdir()
        for n in range(1, 11):
            test = [names[n - 1], names[10 + n - 1]]
            val = [names[n % 10]]
            rest = [x for x in names if x not in test and x not in val]
            if overlap and n == 1:
                rest.append(test[0])
            (labels / f"test{n}.txt").write_text("\n".join(test) + "\n")
            if not (missing_val and n == 3):
                (labels / f"val{n}.txt").write_text("\n".join(val) + "\n")
            (labels / f"train{n}.txt").write_text("\n".join(rest) + "\n")
        return root / "images"

    def test_dtd_follows_official_split_files(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(datasets_loader.DATASETS, {"dtd": spec(2, 20, 10)}):
            images = self.make_dtd(Path(temp))
            dataset, num_classes, folds = datasets_loader.get_dataset_and_loaders("dtd", images)
            self.assertEqual(num_classes, 2)
            self.assertEqual(len(folds), 10)
            for fold, train, val in folds:
                names = [dataset.samples[i][0] for i in indices(val)]
                self.assertEqual(
                    sorted(f"{p.parent.name}/{p.name}" for p in names),
                    sorted([f"banded/banded_{fold - 1}.jpg", f"braided/braided_{fold - 1}.jpg"]))
                self.assertEqual(len(train.dataset) + len(val.dataset), 20)
                self.assertFalse(set(indices(train)) & set(indices(val)))

    def test_dtd_rejects_missing_or_inconsistent_split_files(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(datasets_loader.DATASETS, {"dtd": spec(2, 20, 10)}):
            images = self.make_dtd(Path(temp), missing_val=True)
            with self.assertRaisesRegex(FileNotFoundError, "val3.txt"):
                datasets_loader.get_dataset_and_loaders("dtd", images)
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(datasets_loader.DATASETS, {"dtd": spec(2, 20, 10)}):
            images = self.make_dtd(Path(temp), overlap=True)
            with self.assertRaisesRegex(ValueError, "treino e teste|repetidas"):
                datasets_loader.get_dataset_and_loaders("dtd", images)

    def test_rejects_unknown_dataset_and_mismatched_content(self):
        with self.assertRaisesRegex(ValueError, "desconhecida"):
            datasets_loader.get_dataset_and_loaders("outra")
        with tempfile.TemporaryDirectory() as temp:
            save(Path(temp) / "a" / "0.png")
            with self.assertRaisesRegex(ValueError, "esperado: 47 classes"):
                datasets_loader.get_dataset_and_loaders("dtd", temp)

    def test_kth_real_data_has_four_balanced_sample_folds(self):
        if not Path(datasets_loader.DATASETS["kth"]["default_dir"]).is_dir():
            self.skipTest("KTH-TIPS2b indisponível")
        _, num_classes, folds = datasets_loader.get_dataset_and_loaders("kth")
        self.assertEqual((num_classes, len(folds)), (11, 4))
        self.assertTrue(all(len(val.dataset) == 1188 for _, _, val in folds))


if __name__ == "__main__":
    unittest.main()
