import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
import re
from matplotlib.colors import ListedColormap

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_data"
DIAGNOSTICS_DIR = "diagnostics"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

# Key sensors to plot for comparison - using actual column names from process_simulations.py
SENSORS_TO_PLOT = ['PITCH', 'IMU_ACC_Z', 'ENGINE_RPM', 'SPEED', 'YAW', 'THROTTLE']

def get_generic_context(specific_context: str) -> str:
    """Strips unique IDs from context strings (e.g., ramp_asc_2.5_90.5_1 -> ramp_asc)."""
    if not isinstance(specific_context, str): 
        return "unknown"
    
    # Handle different context types
    if specific_context == "road":
        return "road"
    
    # For ramps, keep direction info but remove unique numbers
    if specific_context.startswith("ramp_"):
        parts = specific_context.split('_')
        if len(parts) >= 2:
            return f"ramp_{parts[1]}"  # e.g., ramp_asc or ramp_desc
        return "ramp"
    
    # For other features, just keep the main type
    for feature_type in ["pothole", "speedbump", "elevated_crosswalk", "cut"]:
        if specific_context.startswith(feature_type):
            return feature_type
    
    return "unknown"

def find_sim_file(sim_id: int) -> str:
    """Finds the CSV file for a given simulation ID."""
    pattern = os.path.join(PROCESSED_DATA_DIR, f"simulation_{sim_id}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No simulation file found for ID {sim_id} in '{PROCESSED_DATA_DIR}'")
    return files[0]

def get_context_colors(unique_contexts):
    """Returns a color mapping for different contexts."""
    # Define specific colors for common road features
    color_map = {
        'road': '#2E8B57',           # SeaGreen
        'ramp_asc': '#FF6347',       # Tomato  
        'ramp_desc': '#FF4500',      # OrangeRed
        'ramp_two': '#DC143C',       # Crimson (distinct red for ramp_two)
        'pothole': '#8B0000',        # DarkRed
        'speedbump': '#FFD700',      # Gold
        'elevated_crosswalk': '#9370DB',  # MediumPurple
        'cut': '#CD853F',            # Peru
        'unknown': '#708090'         # SlateGray
    }
    
    # Use predefined colors where available, otherwise generate from colormap
    colors = []
    cmap = plt.cm.get_cmap('Set3')
    color_idx = 0
    
    for ctx in unique_contexts:
        if ctx in color_map:
            colors.append(color_map[ctx])
        else:
            colors.append(cmap(color_idx / max(1, len(unique_contexts) - len(color_map))))
            color_idx += 1
    
    return dict(zip(unique_contexts, colors))

def plot_sensor_data_with_context(df: pd.DataFrame, sim_id: int):
    """
    Plots sensor data over time, with line segments colored by road context.
    """
    # Add generic context column
    df['generic_context'] = df['context'].apply(get_generic_context)
    
    # Get available sensors (some might not exist in this simulation)
    available_sensors = [sensor for sensor in SENSORS_TO_PLOT if sensor in df.columns]
    
    if not available_sensors:
        print(f"Warning: None of the target sensors {SENSORS_TO_PLOT} found in simulation {sim_id}")
        return
    
    print(f"Plotting {len(available_sensors)} sensors: {available_sensors}")
    
    # Get unique contexts and their colors
    unique_contexts = sorted(df['generic_context'].unique())
    context_colors = get_context_colors(unique_contexts)
    
    print(f"Found {len(unique_contexts)} contexts: {unique_contexts}")
    
    # Create subplots
    num_sensors = len(available_sensors)
    fig, axes = plt.subplots(num_sensors, 1, figsize=(20, 4 * num_sensors))
    if num_sensors == 1:
        axes = [axes]
    
    fig.suptitle(f'Sensor Data Colored by Road Context - Simulation {sim_id}', fontsize=16)
    
    for i, sensor in enumerate(available_sensors):
        ax = axes[i]
        
        # Plot sensor data with context coloring
        time_points = np.arange(len(df))
        sensor_values = df[sensor].values
        
        # Group consecutive points with same context for efficient plotting
        current_context = None
        start_idx = 0
        
        for j in range(len(df) + 1):
            # Check if context changed or we reached the end
            if j == len(df) or df['generic_context'].iloc[j] != current_context:
                # Plot the previous segment if it exists
                if current_context is not None and start_idx < j:
                    segment_time = time_points[start_idx:j]
                    segment_values = sensor_values[start_idx:j]
                    
                    # Make "road" context more transparent to highlight other contexts
                    alpha_value = 0.4 if current_context == 'road' else 1.0
                    
                    ax.plot(segment_time, segment_values, 
                           color=context_colors[current_context], 
                           linewidth=1.5,
                           alpha=alpha_value,
                           label=current_context if current_context not in [line.get_label() for line in ax.lines] else "")
                
                # Update for next segment
                if j < len(df):
                    current_context = df['generic_context'].iloc[j]
                    start_idx = j
        
        ax.set_title(f'{sensor} (colored by road context)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sensor Value')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first subplot to avoid clutter
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(DIAGNOSTICS_DIR, f"sim_{sim_id}_context_colored_sensors.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plot saved to: {output_path}")

def plot_context_distribution(df: pd.DataFrame, sim_id: int):
    """
    Creates a secondary plot showing the distribution of contexts over time.
    """
    df['generic_context'] = df['context'].apply(get_generic_context)
    unique_contexts = sorted(df['generic_context'].unique())
    context_colors = get_context_colors(unique_contexts)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 3))
    
    # Create a context timeline
    time_points = np.arange(len(df))
    context_numeric = pd.Categorical(df['generic_context'], categories=unique_contexts).codes
    
    # Plot as colored segments
    for i, ctx in enumerate(unique_contexts):
        mask = df['generic_context'] == ctx
        if mask.any():
            ax.scatter(time_points[mask], [i] * mask.sum(), 
                      c=context_colors[ctx], s=10, alpha=0.7, label=ctx)
    
    ax.set_title(f'Road Context Timeline - Simulation {sim_id}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Context Type')
    ax.set_yticks(range(len(unique_contexts)))
    ax.set_yticklabels(unique_contexts)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the timeline plot
    output_path = os.path.join(DIAGNOSTICS_DIR, f"sim_{sim_id}_context_timeline.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Context timeline plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots showing sensor data colored by road context.")
    parser.add_argument(
        "--sim_id",
        type=int,
        required=True,
        help="The simulation ID to analyze (e.g., 1)."
    )
    args = parser.parse_args()

    try:
        file_path = find_sim_file(args.sim_id)
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Data shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        print(f"Context column present: {'context' in df.columns}")
        
        if 'context' not in df.columns:
            print("Error: 'context' column not found in the data.")
            return
        
        # Check which sensors are available
        available_sensors = [sensor for sensor in SENSORS_TO_PLOT if sensor in df.columns]
        missing_sensors = [sensor for sensor in SENSORS_TO_PLOT if sensor not in df.columns]
        
        if missing_sensors:
            print(f"Warning: Missing sensors: {missing_sensors}")
        
        if not available_sensors:
            print(f"Error: None of the target sensors {SENSORS_TO_PLOT} are present in the data.")
            return
        
        # Generate both plots
        plot_sensor_data_with_context(df, args.sim_id)
        plot_context_distribution(df, args.sim_id)
        
        # Print some context statistics
        context_counts = df['context'].value_counts()
        print(f"\nContext distribution:")
        for ctx, count in context_counts.items():
            print(f"  {ctx}: {count} time steps ({count/len(df)*100:.1f}%)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 