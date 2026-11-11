"""Treino de ResNet-18 sem pré-treinamento com o protocolo de CV da base escolhida."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from datasets_loader import DATASETS
from resnet_baseline import run_experiment


def create_scratch_model(num_classes):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_and_evaluate_cv(
    data_dir=None, seed=42, epochs=10, batch_size=32,
    learning_rate=1e-4, num_workers=0, device=None,
    model_dir=None, output=None, dataset_name="fmd",
):
    model_dir = Path(model_dir or f"saved_models/resnet_scratch_{dataset_name}")
    output = Path(output or f"results/resnet_scratch_{dataset_name}.json")
    model_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save_final_model(fold, model):
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        path = model_dir / f"resnet_scratch_fold_{fold}.pth"
        torch.save(state, path)
        print(f"Pesos finais do fold {fold} salvos em: {path}", flush=True)

    results = run_experiment(
        data_dir=data_dir, seed=seed, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, num_workers=num_workers, device=device,
        model_factory=create_scratch_model, on_fold_end=save_final_model,
        dataset_name=dataset_name,
    )
    results["experiment"] = "ResNet-18 from scratch"
    results["weights"] = None
    results["evaluation"] = "final epoch"
    results["model_dir"] = str(model_dir)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Resultados salvos em: {output}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Treino da ResNet-18 from scratch com CV na base escolhida.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="fmd")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Padrão: diretório da base escolhida.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Dispositivo PyTorch, por exemplo cpu ou cuda:0.")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Padrão: saved_models/resnet_scratch_<dataset>.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Padrão: results/resnet_scratch_<dataset>.json.")
    args = parser.parse_args()

    train_and_evaluate_cv(
        data_dir=args.data_dir, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, num_workers=args.num_workers, device=args.device,
        model_dir=args.model_dir, output=args.output, dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
