"""Carregamento unificado de FMD, DTD e KTH-TIPS2b, cada uma com o seu protocolo de CV.

- FMD: 10-fold estratificado aleatório (semente 42); a base não tem splits oficiais.
- DTD: 10 splits oficiais (labels/trainN.txt + valN.txt para treino, testN.txt para validação).
- KTH-TIPS2b: 4 folds por amostra física (a, b, c, d): uma amostra testa, as outras 3 treinam.
"""

import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fmd_data import DEFAULT_DATA_DIR, N_SPLITS, FMDDataset, image_transform, make_fold_loaders


TEXTURE_ROOT = Path(__file__).resolve().parent / "dataset_texture" / "test"
KTH_SAMPLES = "abcd"
KTH_SAMPLE_PATTERN = re.compile(r"^\d+([a-d])-")

DATASETS = {
    "fmd": {
        "default_dir": DEFAULT_DATA_DIR,
        "num_classes": 10,
        "num_images": 1000,
        "num_folds": N_SPLITS,
    },
    "dtd": {
        "default_dir": TEXTURE_ROOT / "DTD" / "dtd" / "images",
        "num_classes": 47,
        "num_images": 5640,
        "num_folds": 10,
    },
    "kth": {
        "default_dir": TEXTURE_ROOT / "KTH-TIPS2b" / "KTH-TIPS2-b" / "images",
        "num_classes": 11,
        "num_images": 4752,
        "num_folds": len(KTH_SAMPLES),
    },
}


def _make_loaders(dataset, splits, batch_size, seed, num_workers):
    loaders = []
    for fold, (train_indices, val_indices) in enumerate(splits, start=1):
        generator = torch.Generator().manual_seed(seed + fold)
        train_loader = DataLoader(
            Subset(dataset, list(train_indices)), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, generator=generator,
        )
        val_loader = DataLoader(
            Subset(dataset, list(val_indices)), batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )
        loaders.append((fold, train_loader, val_loader))
    return loaders


def kth_sample_splits(dataset):
    """Um fold por amostra física: testa em uma letra e treina nas outras três."""
    by_sample = {letter: [] for letter in KTH_SAMPLES}
    for index, (path, _) in enumerate(dataset.samples):
        match = KTH_SAMPLE_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Não foi possível identificar a amostra da imagem: {path}")
        by_sample[match.group(1)].append(index)

    class_count = len(dataset.classes)
    for letter, indices in by_sample.items():
        classes_present = {dataset.targets[i] for i in indices}
        if len(classes_present) != class_count:
            raise ValueError(f"A amostra '{letter}' não contém as {class_count} classes.")

    return [
        ([i for other in KTH_SAMPLES if other != letter for i in by_sample[other]],
         by_sample[letter])
        for letter in KTH_SAMPLES
    ]


def _read_label_file(path, relative_to_index):
    if not path.is_file():
        raise FileNotFoundError(
            f"Arquivo de split oficial da DTD não encontrado: {path}. Copie a pasta "
            "'labels' do pacote oficial (dtd-r1.0.1) para ao lado de 'images' ou use labels_dir."
        )
    indices = []
    for line in path.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        if name not in relative_to_index:
            raise ValueError(f"Imagem listada em {path.name} não existe na base: {name}")
        indices.append(relative_to_index[name])
    return indices


def dtd_official_splits(dataset, labels_dir, num_splits=10):
    """Lê train/val/test{1..10}.txt; treino = train + val, validação = test."""
    labels_dir = Path(labels_dir)
    relative_to_index = {
        f"{path.parent.name}/{path.name}": index for index, (path, _) in enumerate(dataset.samples)
    }
    splits = []
    for number in range(1, num_splits + 1):
        train = _read_label_file(labels_dir / f"train{number}.txt", relative_to_index)
        val = _read_label_file(labels_dir / f"val{number}.txt", relative_to_index)
        test = _read_label_file(labels_dir / f"test{number}.txt", relative_to_index)
        train_val = train + val
        if len(set(train_val)) != len(train_val) or len(set(test)) != len(test):
            raise ValueError(f"O split {number} da DTD contém imagens repetidas.")
        if set(train_val) & set(test):
            raise ValueError(f"O split {number} da DTD tem imagens em treino e teste.")
        if len(train_val) + len(test) != len(dataset):
            raise ValueError(f"O split {number} da DTD não cobre toda a base.")
        splits.append((train_val, test))
    return splits


def get_dataset_and_loaders(dataset_name, data_dir=None, batch_size=32, num_folds=None,
                            seed=42, num_workers=0, labels_dir=None):
    """Retorna (dataset, num_classes, fold_loaders) para a base escolhida.

    `fold_loaders` é uma lista de (fold, train_loader, val_loader), com folds a partir de 1.
    O número de folds é definido pelo protocolo da base (FMD=10, DTD=10, KTH=4); passar
    `num_folds` incompatível gera erro. A semente só afeta a FMD e a ordem do treino.
    """
    key = str(dataset_name).lower()
    if key not in DATASETS:
        raise ValueError(f"Base desconhecida: {dataset_name!r}. Use: {', '.join(DATASETS)}.")
    if batch_size <= 0:
        raise ValueError("O tamanho do batch deve ser maior que zero.")
    spec = DATASETS[key]
    if num_folds is not None and num_folds != spec["num_folds"]:
        raise ValueError(
            f"A base '{key}' usa {spec['num_folds']} folds pelo protocolo; recebido: {num_folds}."
        )
    root = Path(data_dir) if data_dir is not None else spec["default_dir"]

    dataset = FMDDataset(root, transform=image_transform())
    if len(dataset.classes) != spec["num_classes"] or len(dataset) != spec["num_images"]:
        raise ValueError(
            f"A base '{key}' em {root} tem {len(dataset.classes)} classes e {len(dataset)} "
            f"imagens; esperado: {spec['num_classes']} classes e {spec['num_images']} imagens."
        )

    if key == "fmd":
        fold_loaders = list(make_fold_loaders(
            dataset, batch_size=batch_size, seed=seed, num_workers=num_workers,
            n_splits=spec["num_folds"],
        ))
    else:
        if key == "kth":
            splits = kth_sample_splits(dataset)
        else:
            splits = dtd_official_splits(dataset, labels_dir or root.parent / "labels",
                                         spec["num_folds"])
        fold_loaders = _make_loaders(dataset, splits, batch_size, seed, num_workers)
    return dataset, len(dataset.classes), fold_loaders
