"""Treino e avaliação da ResNet integrada com o protocolo de CV da base escolhida."""

import argparse
import json
from pathlib import Path

import torch

from datasets_loader import DATASETS
from integrated_resnet import create_integrated_model
from plot_learning_curves import plot_learning_curves
from resnet_baseline import run_experiment


def train_and_evaluate_cv(
    data_dir=None, num_folds=None, num_epochs=10, batch_size=32, lr=1e-4,
    seed=42, num_workers=0, device=None, model_dir=None, output=None, plot_output=None,
    *, dataset_name="fmd", run_name="integrated", model_factory=None, experiment_name="Integrated ResNet-18 Gabor and LBP-inspired fine-tuning",
    weights_name="ResNet18_Weights.IMAGENET1K_V1",
):
    """Treina um modelo independente por fold e retorna as métricas agregadas."""
    model_dir = Path(model_dir or f"saved_models/{run_name}_{dataset_name}")
    output = Path(output or f"results/{run_name}_{dataset_name}.json")
    plot_output = Path(plot_output) if plot_output is not None else output.with_name(output.stem + "_curves.png")
    if output.resolve() == plot_output.resolve():
        raise ValueError("Os arquivos de resultados e de curvas devem ser distintos.")
    model_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if model_factory is None:
        model_factory = create_integrated_model

    def save_final_model(fold, model):
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        path = model_dir / f"integrated_resnet_fold_{fold}.pth"
        torch.save(state, path)
        print(f"Pesos finais do fold {fold} salvos em: {path}", flush=True)

    results = run_experiment(
        data_dir=data_dir, seed=seed, epochs=num_epochs, batch_size=batch_size,
        learning_rate=lr, num_workers=num_workers, device=device,
        model_factory=model_factory, on_fold_end=save_final_model, record_history=True,
        dataset_name=dataset_name, num_folds=num_folds,
    )
    results["experiment"] = experiment_name
    results["weights"] = weights_name
    results["conv1_initialization"] = "32 Gabor + 32 LBP-inspired filters"
    results["evaluation"] = "final epoch"
    results["model_dir"] = str(model_dir)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Resultados salvos em: {output}")
    plot_learning_curves(results, plot_output)
    print(f"Curvas salvas em: {plot_output}")

    accuracy = results["summary"]["accuracy"]
    f1 = results["summary"]["f1_macro"]
    return accuracy["mean"], accuracy["std"], f1["mean"], f1["std"]


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning da ResNet integrada com CV na base escolhida.")
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
                        help="Padrão: saved_models/integrated_<dataset>.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Padrão: results/integrated_<dataset>.json.")
    parser.add_argument("--plot-output", type=Path, default=None)
    args = parser.parse_args()

    train_and_evaluate_cv(
        data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.learning_rate, seed=args.seed, num_workers=args.num_workers,
        device=args.device, model_dir=args.model_dir, output=args.output, plot_output=args.plot_output,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
