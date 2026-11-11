"""Experimento integrado sem pré-treino, com histórico por época."""

import argparse
from pathlib import Path

from datasets_loader import DATASETS
from integrated_resnet import create_integrated_model_scratch
from train_integrated_cv import train_and_evaluate_cv


def main():
    parser = argparse.ArgumentParser(description="ResNet integrada from scratch com CV na base escolhida.")
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
                        help="Padrão: saved_models/integrated_scratch_<dataset>.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Padrão: results/integrated_scratch_<dataset>.json.")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="Padrão: results/integrated_scratch_<dataset>_curves.png.")
    args = parser.parse_args()

    train_and_evaluate_cv(
        data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.learning_rate, seed=args.seed, num_workers=args.num_workers,
        device=args.device, model_dir=args.model_dir, output=args.output, plot_output=args.plot_output,
        dataset_name=args.dataset, run_name="integrated_scratch",
        model_factory=create_integrated_model_scratch,
        experiment_name="Integrated ResNet-18 Gabor and LBP-inspired from scratch",
        weights_name=None,
    )


if __name__ == "__main__":
    main()
