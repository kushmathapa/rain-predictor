from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from rainfall_prediction.config import DATE_COLUMN, FIGURES_DIR, TARGET_COLUMN


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_eda_reports(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    numeric_df = df.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[DATE_COLUMN], df[TARGET_COLUMN], color="#1f77b4", linewidth=1.5)
    ax.set_title("Rainfall Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Precipitation")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "rainfall_time_series.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=numeric_df, orient="h", ax=ax)
    ax.set_title("Weather Variable Box Plot")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot.png", dpi=200)
    plt.close(fig)

    hist_axes = numeric_df.hist(figsize=(14, 10), bins=20, edgecolor="black")
    hist_fig = hist_axes[0][0].figure
    hist_fig.suptitle("Feature Histograms", y=1.02)
    hist_fig.tight_layout()
    hist_fig.savefig(output_dir / "histograms.png", dpi=200)
    plt.close(hist_fig)

    kde_columns = list(numeric_df.columns)
    rows = (len(kde_columns) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    axes = axes.flatten()
    for idx, column in enumerate(kde_columns):
        sns.kdeplot(df[column], fill=True, ax=axes[idx], color="#2ca02c")
        axes[idx].set_title(f"KDE - {column}")
    for idx in range(len(kde_columns), len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "kde_plots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    pairplot = sns.pairplot(
        df[[TARGET_COLUMN, *[col for col in numeric_df.columns if col != TARGET_COLUMN]]],
        corner=True,
        diag_kind="hist",
    )
    pairplot.fig.suptitle("Pair Plot", y=1.02)
    pairplot.savefig(output_dir / "pairplot.png", dpi=200)
    plt.close(pairplot.fig)
