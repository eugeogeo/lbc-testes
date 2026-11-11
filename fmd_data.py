"""Carregamento da FMD e preparação dos folds do experimento."""

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
N_SPLITS = 10
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "FMD_dataset" / "image"


class FMDDataset(Dataset):
    """Lê imagens em diretórios imediatos, um por classe."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Diretório de imagens não encontrado: {self.root}")

        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not self.classes:
            raise ValueError(f"Nenhuma pasta de classe encontrada em: {self.root}")

        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            images = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise ValueError(f"Nenhuma imagem encontrada na classe: {class_dir}")
            self.samples.extend((path, self.class_to_idx[class_name]) for path in images)

        self.targets = [target for _, target in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def make_fold_loaders(dataset, batch_size=32, seed=42, num_workers=0, n_splits=N_SPLITS):
    """Produz, para cada fold, os DataLoaders de treino e validação."""
    if batch_size <= 0:
        raise ValueError("O tamanho do batch deve ser maior que zero.")
    if num_workers < 0:
        raise ValueError("O número de workers não pode ser negativo.")
    if n_splits < 2:
        raise ValueError("A validação cruzada requer pelo menos 2 folds.")

    counts = Counter(dataset.targets)
    for class_name, class_index in dataset.class_to_idx.items():
        if counts[class_index] < n_splits:
            raise ValueError(
                f"A classe '{class_name}' precisa de pelo menos {n_splits} imagens "
                f"para a validação estratificada; encontradas: {counts[class_index]}."
            )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(range(len(dataset)), dataset.targets), start=1
    ):
        generator = torch.Generator().manual_seed(seed + fold)
        train_loader = DataLoader(
            Subset(dataset, train_indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        yield fold, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Preparar validação cruzada estratificada da FMD.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Diretório que contém uma pasta de imagens por classe.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = FMDDataset(args.data_dir, transform=image_transform())
    print(f"Imagens: {len(dataset)} | Classes: {dataset.class_to_idx}")
    for fold, train_loader, val_loader in make_fold_loaders(
        dataset, batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers
    ):
        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"treino={len(train_loader.dataset)}, validação={len(val_loader.dataset)}"
        )
        # O treinamento e a avaliação serão adicionados aqui em uma etapa posterior.


if __name__ == "__main__":
    main()
