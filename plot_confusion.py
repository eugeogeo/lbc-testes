"""Matriz de confusão do melhor fold integrado from scratch."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import torch

from datasets_loader import get_dataset_and_loaders
from plot_filters import (
    DATASET_CHOICES, DEFAULT_RESULTS, dataset_results_path, load_best_model, load_experiment,
)


def validation_predictions(model, loader, device):
    model.to(device)
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Os logits de validação não são finitos.")
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return targets, predictions


def main():
    parser = argparse.ArgumentParser(description="Matriz de confusão do melhor fold integrado from scratch.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None,
                        help="Usa results/integrated_scratch_<dataset>.json e sufixa a imagem.")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Padrão: diretório da base registrada no JSON.")
    parser.add_argument("--device", default=None, help="Dispositivo PyTorch, por exemplo cpu ou cuda:0.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    suffix = f"_{args.dataset}" if args.dataset else ""
    args.results = args.results or (dataset_results_path(args.dataset) if args.dataset
                                    else DEFAULT_RESULTS)
    args.output = args.output or Path(f"results/confusion_matrix{suffix}.png")

    results = load_experiment(args.results)
    # Os mesmos folds do treino: o protocolo e a semente vêm do JSON (JSONs antigos são da FMD).
    dataset, _, loaders = get_dataset_and_loaders(
        results.get("dataset", "fmd"), args.data_dir, batch_size=results["batch_size"],
        seed=results["seed"], num_workers=args.num_workers,
    )
    if dataset.class_to_idx != results["class_to_idx"] or len(dataset) != results["num_images"]:
        raise ValueError("O dataset atual não corresponde ao mapeamento ou tamanho registrado no JSON.")
    model, _, selected, checkpoint = load_best_model(results, args.model_dir)
    val_loader = next(val for fold, _, val in loaders if fold == selected["fold"])
    device = torch.device(args.device if args.device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    targets, predictions = validation_predictions(model, val_loader, device)
    accuracy = accuracy_score(targets, predictions)
    if abs(accuracy - selected["accuracy"]) > 1e-8:
        raise ValueError(
            f"A acurácia reproduzida ({accuracy:.6f}) difere da registrada "
            f"({selected['accuracy']:.6f}); verifique dataset, checkpoint e ambiente."
        )

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(dataset.classes))))
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=dataset.classes, yticklabels=dataset.classes, ax=ax)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Matriz de confusão — fold {selected['fold']} (acurácia {accuracy:.2%})")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Fold {selected['fold']}: acurácia={accuracy:.4f}; checkpoint: {checkpoint}")
    print(f"Matriz de confusão salva em: {args.output}")


if __name__ == "__main__":
    main()
