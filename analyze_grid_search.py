import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def analyze_single_run(summary_file_path, output_dir):
    """
    Analyzes and visualizes the results from a single experiment run.
    """
    df = pd.read_csv(summary_file_path)
    df_val = df[df['data_type'] == 'VALIDATION'].copy()

    if not isinstance(df_val, pd.DataFrame) or df_val.empty:
        print("No VALIDATION data found in the summary file or data is not a DataFrame.")
        return

    print("\n--- Performance on VALIDATION data ---")

    # Create pivot tables for clarity
    try:
        pivot_corr = df_val.pivot_table(
            index=['mass', 'friction'],
            columns='method',
            values='corr_mean'
        )
        print("\n--- Mean Correlation (corr_mean) ---")
        print(pivot_corr.to_string(float_format="%.3f"))
    except Exception as e:
        print(f"Could not create correlation pivot table: {e}")

    try:
        pivot_rmse = df_val.pivot_table(
            index=['mass', 'friction'],
            columns='method',
            values='rmse_mean'
        )
        print("\n--- Mean RMSE (rmse_mean) ---")
        print(pivot_rmse.to_string(float_format="%.3f"))
    except Exception as e:
        print(f"Could not create RMSE pivot table: {e}")

    # --- Generate Correlation Plot ---
    print("\n--- Generating Correlation Performance Plot ---")
    try:
        g_corr = sns.catplot(
            data=df_val, x='method', y='corr_mean',
            col='friction', row='mass', kind='bar',
            height=4, aspect=1.2, palette='viridis'
        )
        g_corr.fig.suptitle('Validation Correlation by Mass and Friction', y=1.03, fontsize=16)
        g_corr.set_axis_labels("Method", "Mean Pearson Correlation")
        g_corr.set_titles("Mass: {row_name} | Friction: {col_name}")

        # Add error bars for standard deviation
        for i, mass in enumerate(g_corr.row_names):
            for j, friction in enumerate(g_corr.col_names):
                ax = g_corr.axes[i, j]
                sub_df = df_val[(df_val['mass'] == mass) & (df_val['friction'] == friction)]
                methods = [tick.get_text() for tick in ax.get_xticklabels()]
                for bar_idx, method in enumerate(methods):
                    bar_data = sub_df[sub_df['method'] == method]
                    if isinstance(bar_data, pd.DataFrame) and not bar_data.empty:
                        mean_val = bar_data['corr_mean'].iloc[0]
                        std_val = bar_data['corr_std'].iloc[0]
                        ax.errorbar(x=bar_idx, y=mean_val, yerr=std_val, fmt='none', c='black', capsize=4)

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plot_path = os.path.join(output_dir, 'single_run_correlation_performance.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(g_corr.fig)
        print(f"Correlation plot saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate correlation plot: {e}")

    # --- Generate RMSE Plot ---
    print("\n--- Generating RMSE Performance Plot ---")
    try:
        g_rmse = sns.catplot(
            data=df_val, x='method', y='rmse_mean',
            col='friction', row='mass', kind='bar',
            height=4, aspect=1.2, palette='plasma'
        )
        g_rmse.fig.suptitle('Validation RMSE by Mass and Friction', y=1.03, fontsize=16)
        g_rmse.set_axis_labels("Method", "Mean RMSE")
        g_rmse.set_titles("Mass: {row_name} | Friction: {col_name}")

        # Add error bars for standard deviation
        for i, mass in enumerate(g_rmse.row_names):
            for j, friction in enumerate(g_rmse.col_names):
                ax = g_rmse.axes[i, j]
                sub_df = df_val[(df_val['mass'] == mass) & (df_val['friction'] == friction)]
                methods = [tick.get_text() for tick in ax.get_xticklabels()]
                for bar_idx, method in enumerate(methods):
                    bar_data = sub_df[sub_df['method'] == method]
                    if isinstance(bar_data, pd.DataFrame) and not bar_data.empty:
                        mean_val = bar_data['rmse_mean'].iloc[0]
                        std_val = bar_data['rmse_std'].iloc[0]
                        ax.errorbar(x=bar_idx, y=mean_val, yerr=std_val, fmt='none', c='black', capsize=4)

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plot_path = os.path.join(output_dir, 'single_run_rmse_performance.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(g_rmse.fig)
        print(f"RMSE plot saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate RMSE plot: {e}")


def analyze_grid_search_results(base_results_dir="grid_search_results"):
    """
    Analyzes and visualizes the results from a completed grid search.
    Switches to single-run analysis if no grid search subdirectories are found.
    """
    print(f"--- Analyzing results from: {base_results_dir} ---")

    # --- Detect analysis mode: grid search vs. single run ---
    if not os.path.exists(base_results_dir):
        print(f"Error: Base results directory '{base_results_dir}' not found.")
        return
        
    run_dirs = [d for d in os.listdir(base_results_dir) if d.startswith('run_') and os.path.isdir(os.path.join(base_results_dir, d))]

    if not run_dirs:
        # --- SINGLE RUN ANALYSIS ---
        print("--- Mode: Single Run Analysis ---")
        summary_file = os.path.join(base_results_dir, 'summary_evaluation_by_group_experiment5.csv')
        if os.path.exists(summary_file):
            analyze_single_run(summary_file, base_results_dir)
        else:
            print(f"Error: Neither grid search runs nor 'summary_evaluation_by_group_experiment5.csv' found in '{base_results_dir}'.")
        return

    # --- GRID SEARCH ANALYSIS ---
    print(f"--- Mode: Grid Search Analysis ({len(run_dirs)} runs found) ---")
    all_results = []

    for run_name in run_dirs:
        run_dir = os.path.join(base_results_dir, run_name)
        params_file = os.path.join(run_dir, 'params.json')
        summary_file = os.path.join(run_dir, 'summary_evaluation_by_group_experiment5.csv')

        if os.path.exists(params_file) and os.path.exists(summary_file):
            try:
                # Load parameters
                with open(params_file, 'r') as f:
                    params = json.load(f)
                
                # Load summary metrics
                df_summary = pd.read_csv(summary_file)
                
                # Add parameters and run name to the summary data
                for key, val in params.items():
                    df_summary[key] = val
                df_summary['run_name'] = run_name
                
                all_results.append(df_summary)
            except Exception as e:
                print(f"Warning: Could not process run '{run_name}'. Error: {e}")

    if not all_results:
        print("No valid results found to analyze.")
        return

    # Combine all results into a single DataFrame
    df_full = pd.concat(all_results, ignore_index=True)

    # --- 2. Define Scoring and Rank Results ---
    df_val = df_full[df_full['data_type'] == 'VALIDATION'].copy()

    # Aggregate metrics across all physics groups for each run
    df_agg = df_val.groupby('run_name').agg(
        overall_corr_mean=('corr_mean', 'mean'),
        overall_rmse_mean=('rmse_mean', 'mean'),
        overall_corr_std=('corr_mean', 'std'),
        overall_rmse_std=('rmse_mean', 'std'),
    ).reset_index()

    # Define weights for the scoring function
    w_corr = 0.6
    w_rmse = 0.4
    w_corr_std = -0.3
    w_rmse_std = -0.2

    # Calculate score. Normalize columns to be on a similar scale before weighting.
    # Add a small epsilon to std to avoid division by zero
    epsilon = 1e-9
    df_agg['score'] = (
        w_corr * (df_agg['overall_corr_mean'] - df_agg['overall_corr_mean'].mean()) / (df_agg['overall_corr_mean'].std() + epsilon) +
        w_rmse * (df_agg['overall_rmse_mean'] - df_agg['overall_rmse_mean'].mean()) / (df_agg['overall_rmse_mean'].std() + epsilon) * -1 + # Invert RMSE
        w_corr_std * (df_agg['overall_corr_std'] - df_agg['overall_corr_std'].mean()) / (df_agg['overall_corr_std'].std() + epsilon) +
        w_rmse_std * (df_agg['overall_rmse_std'] - df_agg['overall_rmse_std'].mean()) / (df_agg['overall_rmse_std'].std() + epsilon)
    )

    # Merge scores back with parameters
    run_params = df_full.drop_duplicates(subset=['run_name']).set_index('run_name')
    param_cols = [col for col in run_params.columns if col not in df_agg.columns and col != 'run_name']
    df_ranked = pd.merge(df_agg, run_params[param_cols], on='run_name').sort_values('score', ascending=False)
    
    print("\n--- Top 10 Hyperparameter Configurations (Validation Data) ---")
    
    # Define columns to display
    metric_cols = ['score', 'overall_corr_mean', 'overall_rmse_mean', 'overall_corr_std']
    display_cols = ['run_name'] + param_cols + metric_cols
    
    # Ensure all columns exist before trying to display them
    display_cols = [col for col in display_cols if col in df_ranked.columns]
    
    print(df_ranked[display_cols].head(10).to_string())

    # Save the full ranked list
    ranked_path = os.path.join(base_results_dir, "ranked_grid_search_results.csv")
    df_ranked.to_csv(ranked_path, index=False)
    print(f"\nFull ranked results saved to: {ranked_path}")

    # --- 3. Visualize Hyperparameter Impact ---
    varied_params = [col for col in param_cols if df_full[col].nunique() > 1]
    
    if not varied_params:
        print("\nNo varied parameters to plot. Exiting visualization.")
        return

    print("\n--- Generating Performance Plots ---")
    analysis_plots_dir = os.path.join(base_results_dir, "analysis_plots")
    os.makedirs(analysis_plots_dir, exist_ok=True)

    for param in varied_params:
        plt.figure(figsize=(14, 6))

        # Plot Correlation vs. Parameter
        plt.subplot(1, 2, 1)
        sns.lineplot(data=df_ranked, x=param, y='overall_corr_mean', marker='o', errorbar='sd', legend=False)
        sns.pointplot(data=df_ranked, x=param, y='overall_corr_mean', color='black', join=False)
        plt.title(f'Mean Correlation vs. {param}')
        plt.ylabel('Mean Pearson Correlation (Validation)')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Plot RMSE vs. Parameter
        plt.subplot(1, 2, 2)
        sns.lineplot(data=df_ranked, x=param, y='overall_rmse_mean', marker='o', errorbar='sd', legend=False)
        sns.pointplot(data=df_ranked, x=param, y='overall_rmse_mean', color='black', join=False)
        plt.title(f'Mean RMSE vs. {param}')
        plt.ylabel('Mean RMSE (Validation)')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_path = os.path.join(analysis_plots_dir, f'performance_vs_{param}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  Saved plot: {plot_path}")
        
    print("--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from a single run or a grid search.")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing the experiment results."
    )
    args = parser.parse_args()
    analyze_grid_search_results(args.results_dir) 