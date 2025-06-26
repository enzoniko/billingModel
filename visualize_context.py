import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def _group_context(context_str: str) -> str:
    """Removes the trailing '_<int>' from a context string for grouping."""
    return re.sub(r'_\d+$', '', context_str)

def plot_context_sequence(csv_path: str, output_dir: str = "."):
    """
    Reads a processed simulation CSV, groups similar contexts, and plots the
    context sequence over time using colored dots.

    Args:
        csv_path (str): The path to the input CSV file.
        output_dir (str): The directory to save the output plot.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        return

    # Extract simulation ID for the plot title and filename
    sim_id = os.path.basename(csv_path).split('_')[1]
    
    # Read data
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Group contexts by removing unique identifiers
    df['context_grouped'] = df['context'].apply(_group_context)

    # Map grouped context strings to integers for plotting on the y-axis
    unique_contexts = sorted(df['context_grouped'].unique())
    context_map = {context: i for i, context in enumerate(unique_contexts)}
    df['context_int'] = df['context_grouped'].map(context_map)
    
    # Create a color map for the unique grouped contexts
    # Using a high-quality colormap like 'tab20' for distinct colors
    colors = plt.cm.get_cmap('tab20', len(unique_contexts))
    color_map = {context: colors(i) for i, context in enumerate(unique_contexts)}

    # Create the plot
    print("Generating plot...")
    plt.figure(figsize=(18, 8)) # Increased height for better legend spacing
    
    # Plot each context group as a separate scatter plot to create a proper legend
    for context, context_int in context_map.items():
        subset = df[df['context_int'] == context_int]
        plt.scatter(subset.index, subset['context_int'], color=color_map[context], label=context, marker='s', s=15)

    # Configure plot aesthetics
    plt.yticks(ticks=list(context_map.values()), labels=list(context_map.keys()))
    plt.xlabel("Time (row index)")
    plt.ylabel("Map Context")
    plt.title(f"Grouped Context Sequence for Simulation {sim_id}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Place legend outside the plot area for clarity
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Map Features")
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=(0, 0, 0.85, 1))

    # Save the plot with a new name
    output_filename = f"context_sequence_sim_{sim_id}_grouped.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    
    print(f"Successfully saved plot to {output_path}")

if __name__ == "__main__":
    # Define the target CSV file
    CSV_FILE = "processed_data/simulation_100_mass_10900_friction_1.0.csv"
    
    # Run the plotting function
    plot_context_sequence(CSV_FILE) 