"""Baseline ResNet-18 com fine-tuning e validação cruzada estratificada."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from datasets_loader import DATASETS, get_dataset_and_loaders


def create_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(model, loader, optimizer, criterion, device, return_accuracy=False):
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("A perda de treino não é finita.")
        if return_accuracy:
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    mean_loss = total_loss / len(loader.dataset)
    if return_accuracy:
        return mean_loss, correct / len(loader.dataset)
    return mean_loss


def evaluate(model, loader, device, num_classes):
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Os logits de validação não são finitos.")
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(
            targets, predictions, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        )),
    }


def run_experiment(data_dir=None, seed=42, epochs=10, batch_size=32,
                   learning_rate=0.0001, num_workers=0, device=None,
                   model_factory=None, on_fold_end=None, record_history=False,
                   dataset_name="fmd", num_folds=None):
    if epochs <= 0:
        raise ValueError("O número de épocas deve ser maior que zero.")
    if batch_size <= 0:
        raise ValueError("O tamanho do batch deve ser maior que zero.")
    if num_workers < 0:
        raise ValueError("O número de workers não pode ser negativo.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("A taxa de aprendizado deve ser finita e maior que zero.")

    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, num_classes, fold_loaders = get_dataset_and_loaders(
        dataset_name, data_dir, batch_size=batch_size, num_folds=num_folds, seed=seed,
        num_workers=num_workers,
    )
    if num_classes < 2:
        raise ValueError("São necessárias pelo menos duas classes.")
    if model_factory is None:
        model_factory = create_model

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    fold_results = []
    for fold, train_loader, val_loader in fold_loaders:
        fold_seed = seed + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed % (2 ** 32))
        torch.manual_seed(fold_seed)
        model = model_factory(num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        history = []
        for epoch in range(1, epochs + 1):
            if record_history:
                loss, accuracy = train_epoch(
                    model, train_loader, optimizer, criterion, device, return_accuracy=True
                )
                history.append({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy})
                print(f"Fold {fold:2d} | época {epoch}/{epochs} | perda treino={loss:.4f} "
                      f"| acurácia treino={accuracy:.4f}", flush=True)
            else:
                loss = train_epoch(model, train_loader, optimizer, criterion, device)
                print(f"Fold {fold:2d} | época {epoch}/{epochs} | perda treino={loss:.4f}", flush=True)

        scores = evaluate(model, val_loader, device, num_classes)
        if on_fold_end is not None:
            on_fold_end(fold, model)
        fold_result = {"fold": fold, **scores}
        if record_history:
            fold_result["history"] = history
        fold_results.append(fold_result)
        print(f"Fold {fold:2d}: acurácia={scores['accuracy']:.4f} | F1 macro={scores['f1_macro']:.4f}",
              flush=True)

    summary = {}
    for metric in ("accuracy", "f1_macro"):
        values = [result[metric] for result in fold_results]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
        print(f"{metric}: média={summary[metric]['mean']:.4f} | desvio padrão={summary[metric]['std']:.4f}")

    return {
        "experiment": "ImageNet ResNet-18 fine-tuning",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "dataset": dataset_name,
        "num_folds": len(fold_loaders),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "class_to_idx": dataset.class_to_idx,
        "num_images": len(dataset),
        "folds": fold_results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning da ResNet-18 com CV na base escolhida.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="fmd")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Padrão: diretório da base escolhida.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Dispositivo PyTorch, por exemplo cpu ou cuda:0.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Padrão: results/resnet18_<dataset>.json.")
    args = parser.parse_args()
    output = args.output or Path(f"results/resnet18_{args.dataset}.json")

    results = run_experiment(
        data_dir=args.data_dir, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, num_workers=args.num_workers, device=args.device,
        dataset_name=args.dataset,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Resultados salvos em: {output}")


if __name__ == "__main__":
    main()
