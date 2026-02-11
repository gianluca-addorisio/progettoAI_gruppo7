import matplotlib.pyplot as plt
from pathlib import Path


def plot_metric_summary(results: dict, metric_name: str, outdir: Path):
    """
    Plot del valore medio (con std se presente).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if "mean" in results[metric_name]:
        mean = results[metric_name]["mean"]
        std = results[metric_name]["std"]

        plt.figure()
        plt.bar([metric_name], [mean], yerr=[std])
        plt.title(f"{metric_name} (mean ± std)")
        plt.ylabel(metric_name)
    else:
        # holdout: singolo valore
        value = results[metric_name]
        plt.figure()
        plt.bar([metric_name], [value])
        plt.title(f"{metric_name} (holdout)")
        plt.ylabel(metric_name)

    plt.tight_layout()
    plt.savefig(outdir / f"{metric_name}_summary.png")
    plt.close()


def plot_metric_distribution(results: dict, metric_name: str, outdir: Path):
    """
    Plot della distribuzione delle performance sulle fold/ripetizioni.
    """
    if "values" not in results[metric_name]:
        return  # holdout non ha distribuzione

    values = results[metric_name]["values"]

    plt.figure()
    plt.boxplot(values)
    plt.title(f"Distribuzione {metric_name} sulle fold")
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(outdir / f"{metric_name}_distribution.png")
    plt.close()
