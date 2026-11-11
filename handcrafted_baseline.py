"""Baseline de texturas LBP + GLCM com SVM linear."""

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import transforms

from fmd_data import DEFAULT_DATA_DIR, FMDDataset, make_fold_loaders


LBP_POINTS = 8
LBP_RADIUS = 1
GLCM_LEVELS = 32
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPERTIES = ("contrast", "energy", "homogeneity", "correlation")


def extract_features(image):
    """Calcula o histograma LBP e as propriedades GLCM de uma imagem PIL."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    histogram, _ = np.histogram(lbp, bins=LBP_POINTS + 2, range=(0, LBP_POINTS + 2))
    histogram = histogram.astype(np.float64) / gray.size

    quantized = gray // (256 // GLCM_LEVELS)
    glcm = graycomatrix(
        quantized, distances=[1], angles=GLCM_ANGLES, levels=GLCM_LEVELS,
        symmetric=True, normed=True,
    )
    properties = [graycoprops(glcm, name).mean() for name in GLCM_PROPERTIES]
    return np.concatenate((histogram, properties))


def evaluate_fold(train_subset, val_subset, feature_cache=None):
    """Extrai features das imagens do fold, treina o SVM e avalia a validação.

    O cache opcional armazena somente descritores calculados por imagem.
    """
    dataset = train_subset.dataset
    if val_subset.dataset is not dataset:
        raise ValueError("Os subsets de treino e validação devem pertencer ao mesmo dataset.")
    if feature_cache is None:
        feature_cache = {}

    def features_and_targets(subset):
        features, targets = [], []
        for index in subset.indices:
            key = (dataset, index)
            if key not in feature_cache:
                image, target = dataset[index]
                feature_cache[key] = (extract_features(image), target)
            feature, target = feature_cache[key]
            features.append(feature)
            targets.append(target)
        return np.asarray(features), np.asarray(targets)

    train_features, train_targets = features_and_targets(train_subset)
    val_features, val_targets = features_and_targets(val_subset)
    classifier = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(val_features)
    return {
        "accuracy": float(accuracy_score(val_targets, predictions)),
        "f1_macro": float(f1_score(
            val_targets, predictions, labels=list(range(len(dataset.classes))),
            average="macro", zero_division=0,
        )),
    }


def run_experiment(data_dir=DEFAULT_DATA_DIR, seed=42):
    dataset = FMDDataset(data_dir, transform=transforms.Resize((224, 224)))
    if len(dataset.classes) < 2:
        raise ValueError("O SVM requer pelo menos duas classes.")

    cache = {}
    fold_results = []
    for fold, train_loader, val_loader in make_fold_loaders(dataset, seed=seed):
        scores = evaluate_fold(train_loader.dataset, val_loader.dataset, cache)
        fold_results.append({"fold": fold, **scores})
        print(f"Fold {fold:2d}: acurácia={scores['accuracy']:.4f} | F1 macro={scores['f1_macro']:.4f}")

    summary = {}
    for metric in ("accuracy", "f1_macro"):
        values = [result[metric] for result in fold_results]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
        print(f"{metric}: média={summary[metric]['mean']:.4f} | desvio padrão={summary[metric]['std']:.4f}")

    return {
        "experiment": "LBP + GLCM + linear SVM",
        "seed": seed,
        "class_to_idx": dataset.class_to_idx,
        "num_images": len(dataset),
        "folds": fold_results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Avaliar o baseline handcrafted na FMD.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Diretório que contém uma pasta de imagens por classe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/handcrafted.json"),
                        help="Arquivo JSON para os resultados.")
    args = parser.parse_args()

    results = run_experiment(args.data_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
