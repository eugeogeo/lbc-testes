"""Visualização de históricos de treino de experimentos em validação cruzada."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(results, output):
    folds = results["folds"]
    epochs = results["epochs"]
    if not folds or epochs < 1:
        raise ValueError("O histórico deve conter folds e ao menos uma época.")
    x = list(range(1, epochs + 1))
    for fold in folds:
        if [entry["epoch"] for entry in fold["history"]] != x:
            raise ValueError(f"Histórico incompleto ou fora de ordem no fold {fold['fold']}.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0], "train_loss", "Perda de treino", "Cross-entropy"),
        (axes[1], "train_accuracy", "Acurácia de treino", "Acurácia"),
    ):
        values = []
        for fold in folds:
            series = [entry[metric] for entry in fold["history"]]
            if not np.isfinite(series).all():
                raise ValueError(f"Métricas não finitas no fold {fold['fold']}.")
            values.append(series)
            ax.plot(x, series, alpha=0.55, linewidth=1.2, label=f"Fold {fold['fold']}")
        ax.plot(x, np.mean(values, axis=0), color="black", linewidth=2.5, label="Média")
        ax.set(title=title, xlabel="Época", ylabel=ylabel)
        ax.set_xticks(x if epochs <= 20 else x[::max(1, epochs // 10)])
        ax.grid(alpha=0.25)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right", ncol=2, fontsize="small")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gerar curvas a partir de resultados JSON.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = json.loads(args.input.read_text(encoding="utf-8"))
    plot_learning_curves(results, args.output)
    print(f"Curvas salvas em: {args.output}")


if __name__ == "__main__":
    main()
