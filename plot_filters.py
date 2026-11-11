"""Visualiza conv1 inicial e treinada do melhor fold integrado from scratch."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from integrated_resnet import create_integrated_model_scratch


DEFAULT_RESULTS = Path("results/integrated_scratch_run.json")
DATASET_CHOICES = ["fmd", "dtd", "kth"]


def dataset_results_path(dataset):
    return Path(f"results/integrated_scratch_{dataset}.json")


def load_experiment(path):
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    if results["experiment"] != "Integrated ResNet-18 Gabor and LBP-inspired from scratch":
        raise ValueError("O JSON não corresponde ao experimento integrado from scratch.")
    if results["weights"] is not None:
        raise ValueError("O experimento deve ter sido iniciado sem pesos pré-treinados.")
    folds = results["folds"]
    expected_folds = results.get("num_folds", 10)
    if len(folds) != expected_folds or {entry["fold"] for entry in folds} != set(range(1, expected_folds + 1)):
        raise ValueError(f"O JSON deve conter exatamente os {expected_folds} folds esperados.")
    if any(not math.isfinite(entry["accuracy"]) or not 0 <= entry["accuracy"] <= 1 for entry in folds):
        raise ValueError("Acurácia inválida no JSON.")
    classes = results["class_to_idx"]
    if sorted(classes.values()) != list(range(len(classes))) or len(classes) < 2:
        raise ValueError("Mapeamento de classes inválido no JSON.")
    return results


def best_fold(results):
    """Em caso de empate, escolhe o fold com menor número."""
    return max(results["folds"], key=lambda entry: (entry["accuracy"], -entry["fold"]))


def load_best_model(results, model_dir=None):
    selected = best_fold(results)
    directory = Path(model_dir) if model_dir is not None else Path(results["model_dir"])
    checkpoint = directory / f"integrated_resnet_fold_{selected['fold']}.pth"
    model = create_integrated_model_scratch(len(results["class_to_idx"]))
    initial = model.conv1.weight.detach().cpu().clone()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, initial, selected, checkpoint


def normalize_filter_pair(initial, trained):
    """Usa a mesma escala simétrica para todas as imagens de ambos os estados."""
    initial = initial.detach().cpu().numpy()
    trained = trained.detach().cpu().numpy()
    if initial.shape != (64, 3, 7, 7) or trained.shape != initial.shape:
        raise ValueError("Os pesos de conv1 devem ter o formato (64, 3, 7, 7).")
    if not np.isfinite(initial).all() or not np.isfinite(trained).all():
        raise ValueError("Os pesos dos filtros devem ser finitos.")
    scale = max(float(np.max(np.abs(initial))), float(np.max(np.abs(trained))))
    if scale == 0:
        raise ValueError("Não é possível visualizar filtros todos nulos.")
    return tuple(np.transpose(weights / (2 * scale) + 0.5, (0, 2, 3, 1))
                 for weights in (initial, trained))


def save_grid(filters, title, output):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        ax.imshow(filters[index], interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(str(index + 1), fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualizar filtros iniciais e treinados da conv1.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None,
                        help="Usa results/integrated_scratch_<dataset>.json e sufixa as imagens.")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Substitui o diretório de checkpoints registrado no JSON.")
    parser.add_argument("--initial-output", type=Path, default=None)
    parser.add_argument("--trained-output", type=Path, default=None)
    args = parser.parse_args()
    suffix = f"_{args.dataset}" if args.dataset else ""
    args.results = args.results or (dataset_results_path(args.dataset) if args.dataset
                                    else DEFAULT_RESULTS)
    args.initial_output = args.initial_output or Path(f"results/filters_initial{suffix}.png")
    args.trained_output = args.trained_output or Path(f"results/filters_trained{suffix}.png")
    if args.initial_output.resolve() == args.trained_output.resolve():
        raise ValueError("Os arquivos de saída inicial e treinado devem ser distintos.")

    results = load_experiment(args.results)
    model, initial, selected, checkpoint = load_best_model(results, args.model_dir)
    initial_rgb, trained_rgb = normalize_filter_pair(initial, model.conv1.weight)
    fold = selected["fold"]
    save_grid(initial_rgb, "Filtros iniciais de conv1", args.initial_output)
    save_grid(trained_rgb, f"Filtros treinados de conv1 — fold {fold}", args.trained_output)
    print(f"Fold {fold}: acurácia={selected['accuracy']:.4f}; checkpoint: {checkpoint}")
    print(f"Imagens salvas em: {args.initial_output} e {args.trained_output}")


if __name__ == "__main__":
    main()
