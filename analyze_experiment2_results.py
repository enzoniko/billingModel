import os
import argparse
import typing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def _add_manual_error_bars(g: sns.FacetGrid, x_var: str, stat_col: str, err_col: str):
    """
    Manually add error bars to a seaborn FacetGrid where data is pre-aggregated.
    `errorbar="sd"` will not work if there's only one row per bar.
    This implementation uses public APIs of FacetGrid for better robustness.
    """
    # Get unique methods and create a mapping to positions
    all_methods = g.data[x_var].unique()
    method_to_pos = {method: i for i, method in enumerate(all_methods)}
    
    # Get the row and column variables
    row_var = g._row_var
    col_var = g._col_var
    
    # Iterate through each axis in the grid
    for i, ax in enumerate(g.axes.flat):
        # Calculate which row and column this axis represents
        n_cols = g.axes.shape[1]
        row_idx = i // n_cols
        col_idx = i % n_cols
        
        # Get the actual row and column values
        row_val = g.row_names[row_idx] if g.row_names else None
        col_val = g.col_names[col_idx] if g.col_names else None
        
        # Filter data for this specific subplot
        subplot_data = g.data.copy()
        if row_val is not None and row_var:
            subplot_data = subplot_data[subplot_data[row_var] == row_val]
        if col_val is not None and col_var:
            subplot_data = subplot_data[subplot_data[col_var] == col_val]
        
        # Add error bars for each method in this subplot
        for _, row_data in subplot_data.iterrows():
            method = row_data[x_var]
            mean_val = row_data[stat_col]
            std_val = row_data[err_col]
            
            # Get the x position for this method
            x_pos = method_to_pos[method]
            
            # Add error bar
            ax.errorbar(x=x_pos, y=mean_val, yerr=std_val,
                       fmt='none', capsize=3, color='black', zorder=10)


def main(csv_path: str, output_dir: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df_val: pd.DataFrame = typing.cast(pd.DataFrame, df[df["data_type"] == "VALIDATION"].copy())
    if df_val.empty:
        print("❌ No VALIDATION rows found in the CSV – nothing to plot.")
        return

    # Set plot style for a compact, single-column paper format
    sns.set_context("paper", rc={
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.titlesize": 18,
    })

    print("\n--- VALIDATION performance pivot (corr_mean) ---")
    try:
        pivot_corr = df_val.pivot_table(index=["mass", "friction"],
                                        columns="method",
                                        values="corr_mean")
        print(pivot_corr.to_string(float_format="%.3f"))
    except Exception as e:
        print(f"Could not create correlation pivot: {e}")

    # ─── Correlation plot ────────────────────────────────────────────────────
    print("\n📊 Generating correlation bar-plot …")
    bar_width = 1.0  # Set to 1.0 to make bars touch, removing space between them

    g_corr = sns.catplot(
        data=df_val,
        x="method",
        y="corr_mean",
        row="mass",
        col="friction",
        kind="bar",
        height=2.5,
        aspect=1.2,
        palette="viridis",
        hue="method",  # Use hue to address the FutureWarning and assign colors
        width=bar_width,
        errorbar=None,
        legend=False,
        sharey=True,
    )
    g_corr.fig.suptitle("Validation Pearson Correlation", y=1.08)

    # Manually add error bars BEFORE modifying the axis labels or ticks
    _add_manual_error_bars(g_corr, "method", "corr_mean", "corr_std")

    # Now that error bars are drawn, we can safely modify/hide the labels
    g_corr.set_axis_labels("", "")
    g_corr.set_titles("")
    g_corr.set_xticklabels([])

    # Add axis values for rows and columns
    for i, ax in enumerate(g_corr.axes.flat):
        # Set column titles (friction) on the bottom row of subplots
        if i >= len(g_corr.axes.flat) - len(g_corr.col_names):
            ax.set_xlabel(g_corr.col_names[i % len(g_corr.col_names)])
        # Set row titles (mass) on the first column of subplots
        if i % len(g_corr.col_names) == 0:
            ax.set_ylabel(g_corr.row_names[i // len(g_corr.col_names)])

    # Create and add a global legend above the plot
    methods = df_val["method"].unique()
    palette = sns.color_palette("viridis", len(methods))
    legend_elements = [Patch(facecolor=palette[i], label=methods[i]) for i in range(len(methods))]
    g_corr.fig.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, 1.0), ncol=len(methods), fontsize=18)

    for ax in g_corr.axes.flat:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    g_corr.fig.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.tight_layout(rect=(0.03, 0.03, 1, 0.93))

    corr_path = os.path.join(output_dir, "validation_corr_mean.png")
    g_corr.savefig(corr_path, dpi=300)
    plt.close(g_corr.fig)
    print(f"✅ Saved: {corr_path}")

    # ─── RMSE plot ───────────────────────────────────────────────────────────
    print("📊 Generating RMSE bar-plot …")
    g_rmse = sns.catplot(
        data=df_val,
        x="method",
        y="rmse_mean",
        row="mass",
        col="friction",
        kind="bar",
        height=2.5,
        aspect=1.2,
        palette="plasma",
        hue="method",
        width=bar_width,
        errorbar=None,
        legend=False,
        sharey=True,
    )
    g_rmse.fig.suptitle("Validation RMSE", y=1.08)

    # Manually add error bars
    _add_manual_error_bars(g_rmse, "method", "rmse_mean", "rmse_std")

    # Now hide the tick labels
    g_rmse.set_axis_labels("", "")
    g_rmse.set_titles("")
    g_rmse.set_xticklabels([])

    # Add axis values for rows and columns
    for i, ax in enumerate(g_rmse.axes.flat):
        if i >= len(g_rmse.axes.flat) - len(g_rmse.col_names):
            ax.set_xlabel(g_rmse.col_names[i % len(g_rmse.col_names)])
        if i % len(g_rmse.col_names) == 0:
            ax.set_ylabel(g_rmse.row_names[i // len(g_rmse.col_names)])

    # Create and add a global legend above the plot
    palette_rmse = sns.color_palette("plasma", len(methods))
    legend_elements_rmse = [Patch(facecolor=palette_rmse[i], label=methods[i]) for i in range(len(methods))]
    g_rmse.fig.legend(handles=legend_elements_rmse, loc='upper center',
                      bbox_to_anchor=(0.5, 1.0), ncol=len(methods), fontsize=18)

    for ax in g_rmse.axes.flat:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    g_rmse.fig.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.tight_layout(rect=(0.03, 0.03, 1, 0.93))

    rmse_path = os.path.join(output_dir, "validation_rmse_mean.png")
    g_rmse.savefig(rmse_path, dpi=300)
    plt.close(g_rmse.fig)
    print(f"✅ Saved: {rmse_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot correlation & RMSE bar-plots for an Experiment-2 summary CSV.")
    parser.add_argument("summary_csv", nargs="?", default="results_experiment2_old/summary_evaluation_by_group_experiment2.csv",
                        help="Path to summary_evaluation_by_group_experiment2.csv")
    parser.add_argument("--out", default=None, help="Output directory for plots (defaults to CSV's directory)")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.summary_csv)) or "."
    main(args.summary_csv, out_dir)