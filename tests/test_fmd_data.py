import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from fmd_data import FMDDataset, image_transform, make_fold_loaders


class FMDDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        for class_name in ("wood", "fabric"):
            class_dir = self.root / class_name
            class_dir.mkdir()
            for index in range(10):
                Image.new("RGB", (8, 12), (255, 0, 0)).save(class_dir / f"{index:02}.png")
            (class_dir / "metadata.txt").write_text("not an image")

    def test_dataset_and_transform(self):
        dataset = FMDDataset(self.root, image_transform())
        self.assertEqual(dataset.class_to_idx, {"fabric": 0, "wood": 1})
        self.assertEqual(len(dataset), 20)
        image, target = dataset[0]
        self.assertEqual(target, 0)
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        expected = torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
        torch.testing.assert_close(image[:, 0, 0], expected)

    def test_stratified_reproducible_folds_and_batches(self):
        dataset = FMDDataset(self.root, image_transform())
        folds = list(make_fold_loaders(dataset, batch_size=4, seed=42))
        repeated = list(make_fold_loaders(dataset, batch_size=4, seed=42))
        self.assertEqual(len(folds), 10)
        validation_indices = []
        for (fold, train, val), (repeated_fold, repeated_train, repeated_val) in zip(folds, repeated):
            self.assertEqual(fold, repeated_fold)
            self.assertEqual(train.dataset.indices, repeated_train.dataset.indices)
            self.assertEqual(val.dataset.indices, repeated_val.dataset.indices)
            self.assertFalse(set(train.dataset.indices) & set(val.dataset.indices))
            self.assertEqual(len(train.dataset), 18)
            self.assertEqual(len(val.dataset), 2)
            self.assertEqual(sorted(dataset.targets[i] for i in val.dataset.indices), [0, 1])
            validation_indices.extend(val.dataset.indices)

        self.assertEqual(sorted(validation_indices), list(range(len(dataset))))
        images, labels = next(iter(folds[0][1]))
        self.assertEqual(tuple(images.shape), (4, 3, 224, 224))
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertTrue(torch.equal(labels, next(iter(repeated[0][1]))[1]))
        val_images, val_labels = next(iter(folds[0][2]))
        self.assertEqual(tuple(val_images.shape), (2, 3, 224, 224))
        self.assertEqual(tuple(val_labels.shape), (2,))

    def test_rejects_class_with_too_few_images(self):
        (self.root / "wood" / "00.png").unlink()
        dataset = FMDDataset(self.root)
        with self.assertRaisesRegex(ValueError, "wood.*pelo menos 10"):
            list(make_fold_loaders(dataset))

    def test_rejects_missing_or_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "não encontrado"):
            FMDDataset(self.root / "missing")
        (self.root / "empty").mkdir()
        with self.assertRaisesRegex(ValueError, "Nenhuma imagem"):
            FMDDataset(self.root)


if __name__ == "__main__":
    unittest.main()
