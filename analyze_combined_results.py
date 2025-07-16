import os
import argparse
import typing
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def _add_manual_error_bars(g: sns.FacetGrid, x_var: str, stat_col: str, err_col: str):
    """
    Manually add error bars to a seaborn FacetGrid where data is pre-aggregated.
    Mirrors implementation from analyze_experiment2_results.py.
    """
    # Mapping of method → x-position within each bar cluster
    df_all = typing.cast(pd.DataFrame, g.data)
    methods = df_all[x_var].unique()
    x_pos_map = {m: i for i, m in enumerate(methods)}

    # Seaborn stores the facet variable names in private attrs
    row_var = g._row_var  # type: ignore[attr-defined]
    col_var = g._col_var  # type: ignore[attr-defined]

    n_cols = g.axes.shape[1]
    for i, ax in enumerate(g.axes.flat):
        row_idx = i // n_cols
        col_idx = i % n_cols

        row_val = g.row_names[row_idx] if g.row_names else None
        col_val = g.col_names[col_idx] if g.col_names else None

        sub = typing.cast(pd.DataFrame, df_all.copy())
        if row_val is not None and row_var:
            sub = sub[sub[row_var] == row_val]
        if col_val is not None and col_var:
            sub = sub[sub[col_var] == col_val]

        for _, r in sub.iterrows():  # type: ignore[attr-defined]
            xpos = x_pos_map[r[x_var]]
            ax.errorbar(x=xpos, y=r[stat_col], yerr=r[err_col], fmt='none', capsize=3, color='black', zorder=10)


def load_and_merge(csv_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def plot_stats(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df_val = typing.cast(pd.DataFrame, df[df["data_type"] == "VALIDATION"].copy())
    if df_val.empty:
        print("❌ No VALIDATION rows found – nothing to plot.")
        return

    sns.set_context("paper", rc={k: 18 for k in [
        "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "figure.titlesize"]})

    for metric, err in [("corr_mean", "corr_std"), ("rmse_mean", "rmse_std")]:
        palette = "viridis" if metric == "corr_mean" else "plasma"
        bar = sns.catplot(
            data=df_val,
            x="method",
            y=metric,
            row="mass",
            col="friction",
            kind="bar",
            height=2.5,
            aspect=1.2,
            palette=palette,
            hue="method",
            width=1.0,
            errorbar=None,
            legend=False,
            sharey=True,
        )
        title = "Validation Pearson Correlation" if metric == "corr_mean" else "Validation RMSE"
        bar.fig.suptitle(title, y=1.08)
        _add_manual_error_bars(bar, "method", metric, err)
        bar.set_axis_labels("", "")
        bar.set_titles("")
        bar.set_xticklabels([])

        # Label rows / cols
        for i, ax in enumerate(bar.axes.flat):
            if i >= len(bar.axes.flat) - len(bar.col_names):
                ax.set_xlabel(bar.col_names[i % len(bar.col_names)])
            if i % len(bar.col_names) == 0:
                ax.set_ylabel(bar.row_names[i // len(bar.col_names)])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        df_val_df = typing.cast(pd.DataFrame, df_val)
        methods = df_val_df["method"].unique()
        pal = sns.color_palette(palette, len(methods))
        legend_elems = [Patch(facecolor=pal[i], label=methods[i]) for i in range(len(methods))]
        bar.fig.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(methods), fontsize=18)

        bar.fig.subplots_adjust(wspace=0.05, hspace=0.15)
        plt.tight_layout(rect=(0.03, 0.03, 1, 0.93))

        fname = "validation_corr_mean.png" if metric == "corr_mean" else "validation_rmse_mean.png"
        out_path = os.path.join(output_dir, fname)
        bar.savefig(out_path, dpi=300)
        plt.close(bar.fig)
        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple experiment summary CSVs and plot comparison bar-plots.")
    parser.add_argument("csvs", nargs="+", help="Paths to summary CSV files (experiment2 & experiment5)")
    parser.add_argument("--out", default="combined_plots", help="Output directory")
    args = parser.parse_args()

    df_combined = load_and_merge(args.csvs)
    plot_stats(df_combined, args.out) 