import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
import glob
from ssqueezepy import ssq_cwt

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_data"
DIAGNOSTICS_DIR = "diagnostics"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
SAMPLING_FREQUENCY = 10  # Hz (from 100ms per measurement)

# Key sensors to plot for comparison
SENSORS_TO_PLOT = [
    'IMU_ACC_Z_DYNAMIC', 'IMU_ACC_X', 'IMU_ACC_Y', 
    'ENGINE_RPM', 'SPEED', 'THROTTLE'
]

# Crash detection parameters
CRASH_THRESHOLD = 10.0  # m/s² threshold for crash detection
CRASH_WINDOW_SECONDS = 1.0  # ±1 second window around crash points

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
    
    if specific_context == 'crash':
        return 'crash'
    
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
    color_map = {
        'road': '#2E8B57',           # SeaGreen
        'ramp_asc': '#FF6347',       # Tomato  
        'ramp_desc': '#FF4500',      # OrangeRed
        'ramp_two': '#DC143C',       # Crimson
        'pothole': '#8B0000',        # DarkRed
        'speedbump': '#FFD700',      # Gold
        'elevated_crosswalk': '#9370DB',  # MediumPurple
        'cut': '#CD853F',            # Peru
        'crash': '#4B0082',          # Indigo
        'unknown': '#708090'         # SlateGray
    }
    
    # Use predefined colors where available, otherwise use matplotlib defaults
    colors = []
    default_cmap = plt.colormaps.get_cmap('tab10')
    color_idx = 0
    
    for ctx in unique_contexts:
        if ctx in color_map:
            colors.append(color_map[ctx])
        else:
            colors.append(default_cmap(color_idx % 10))
            color_idx += 1
    
    return dict(zip(unique_contexts, colors))

def find_continuous_segments(mask):
    """Find continuous segments where mask is True."""
    if len(mask) == 0:
        return []
    
    # Find transitions
    diff = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    return list(zip(starts, ends))

def process_data_for_diagnostics(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    Applies gravity correction and identifies crash events while preserving existing contexts.
    """
    print("Processing data: Applying gravity correction and detecting crashes...")
    
    # Standardize column names to uppercase for consistency
    df.columns = [col.upper() for col in df.columns]

    # Step 1: Correct for Gravity
    if 'IMU_ACC_Z' in df.columns:
        df['IMU_ACC_Z_DYNAMIC'] = df['IMU_ACC_Z'] - 9.81
    else:
        df['IMU_ACC_Z_DYNAMIC'] = 0
        print("Warning: 'IMU_ACC_Z' not found. 'IMU_ACC_Z_DYNAMIC' set to 0.")

    # Step 2: Detect and Label "Crash" Context (PRESERVE existing contexts)
    if 'IMU_ACC_X' in df.columns and 'IMU_ACC_Y' in df.columns:
        df['ACC_HORIZONTAL'] = np.sqrt(df['IMU_ACC_X']**2 + df['IMU_ACC_Y']**2)
        crash_indices = df.index[df['ACC_HORIZONTAL'] > CRASH_THRESHOLD].tolist()
        
        if crash_indices:
            print(f"Found {len(crash_indices)} potential crash points. Applying context window...")
            window_size = int(CRASH_WINDOW_SECONDS * fs)
            
            # Initialize context column ONLY if it doesn't exist
            if 'CONTEXT' not in df.columns:
                df['CONTEXT'] = 'unknown'
            
            # Convert to object type to allow string assignment
            df['CONTEXT'] = df['CONTEXT'].astype(object)

            # Mark crash contexts while preserving existing non-crash contexts
            for idx in crash_indices:
                start = max(0, idx - window_size)
                end = min(len(df), idx + window_size + 1)
                df.loc[start:end, 'CONTEXT'] = 'crash'
        else:
            print("No crash events detected.")
    else:
        print("Warning: Cannot detect crashes - missing IMU_ACC_X or IMU_ACC_Y")
        
    # Ensure we have a context column for further processing
    if 'CONTEXT' not in df.columns:
        df['CONTEXT'] = 'unknown'
        
    return df

def plot_sensor_data_with_context(df: pd.DataFrame, sim_id: int, available_sensors: list, context_colors: dict):
    """
    Creates time-series plots with context-colored line segments.
    """
    if not available_sensors:
        print(f"Warning: No target sensors found in simulation {sim_id}")
        return
    
    print(f"Plotting time-series for {len(available_sensors)} sensors...")
    
    num_sensors = len(available_sensors)
    fig, axes = plt.subplots(num_sensors, 1, figsize=(50, 4 * num_sensors))
    if num_sensors == 1:
        axes = [axes]
    
    fig.suptitle(f'Sensor Data Colored by Road Context - Simulation {sim_id}', fontsize=16)
    
    for i, sensor in enumerate(available_sensors):
        ax = axes[i]
        time_points = np.arange(len(df))
        sensor_values = df[sensor].values
        
        # Plot segments grouped by context
        current_context = None
        start_idx = 0
        
        for j in range(len(df) + 1):
            if j == len(df) or df['generic_context'].iloc[j] != current_context:
                # Plot the previous segment
                if current_context is not None and start_idx < j:
                    segment_time = time_points[start_idx:j]
                    segment_values = sensor_values[start_idx:j]
                    
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
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(DIAGNOSTICS_DIR, f"sim_{sim_id}_context_colored_sensors_timeseries.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time-series plot saved to: {output_path}")

def plot_sswt_spectrograms_by_context(df: pd.DataFrame, sim_id: int, available_sensors: list, context_colors: dict, fs: int):
    """
    Generates SSWT spectrograms grouped by context - each context gets its own column with concatenated segments.
    """
    print(f"Generating context-grouped SSWT spectrograms for {len(available_sensors)} sensors...")
    
    unique_contexts = sorted(df['generic_context'].unique())
    # Rearrange contexts to put 'crash' and 'road' at the end for enhanced visibility
    priority_contexts = ['crash', 'road']
    other_contexts = [ctx for ctx in unique_contexts if ctx not in priority_contexts]
    available_priority = [ctx for ctx in priority_contexts if ctx in unique_contexts]
    analysis_contexts = other_contexts + available_priority
    
    if not analysis_contexts:
        print("No contexts found for analysis.")
        return
        
    print(f"Analyzing contexts: {analysis_contexts}")
    
    num_sensors = len(available_sensors)
    num_contexts = len(analysis_contexts)
    
    # First pass: calculate width ratios based on concatenated data length for each context
    context_lengths = []
    for context in analysis_contexts:
        context_mask = (df['generic_context'] == context).values
        segments = find_continuous_segments(context_mask)
        total_length = sum(end - start for start, end in segments) if segments else 1
        context_lengths.append(total_length)
    
    # Calculate width ratios with enhanced stretching for crash and road contexts
    min_width = min(context_lengths) if context_lengths else 1
    raw_ratios = [length / min_width for length in context_lengths]
    max_ratio = max(raw_ratios) if raw_ratios else 1
    
    width_ratios = []
    for i, (context, ratio) in enumerate(zip(analysis_contexts, raw_ratios)):
        if context in ['crash', 'road']:
            # Enhanced stretching: minimum 2x + dynamic scaling, capped at 6x
            enhanced_ratio = max(2, min(6, ratio * 4 / max_ratio + 2))
            width_ratios.append(enhanced_ratio)
        else:
            # Regular scaling for other contexts, capped at 3x
            regular_ratio = max(1, min(3, ratio * 2 / max_ratio))
            width_ratios.append(regular_ratio)
    
    print(f"Context lengths: {dict(zip(analysis_contexts, context_lengths))}")
    print(f"Width ratios: {dict(zip(analysis_contexts, width_ratios))}")
    
    # Create figure with dynamic width based on context lengths
    total_width = sum(width_ratios) * 4  # Base width per unit (reduced since crash/road get enhanced width)
    fig = plt.figure(figsize=(total_width, 5 * num_sensors))
    
    # Create GridSpec with custom width ratios
    gs = GridSpec(num_sensors, num_contexts, figure=fig, width_ratios=width_ratios)
    
    # Create axes array manually
    axes = []
    for i in range(num_sensors):
        row_axes = []
        for j in range(num_contexts):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Handle single cases for consistency
    if num_sensors == 1 and num_contexts == 1:
        axes = [[axes[0][0]]]
    elif num_sensors == 1:
        axes = [axes[0]]
    
    fig.suptitle(f'SSWT Spectrograms Grouped by Context - Simulation {sim_id}', fontsize=18)
    
    # Process each sensor
    for sensor_idx, sensor in enumerate(available_sensors):
        signal = df[sensor].fillna(0).values
        
        try:
            print(f"Processing SSWT for {sensor}...")
            
            # First pass: collect all magnitudes for this sensor to determine consistent color scale
            all_magnitudes = []
            context_data = {}  # Store data for second pass
            
            for context_idx, context in enumerate(analysis_contexts):
                # Find all segments for this context
                context_mask = (df['generic_context'] == context).values
                segments = find_continuous_segments(context_mask)
                
                if not segments:
                    context_data[context] = None
                    continue
                
                # Concatenate signal segments for this context
                concatenated_signal = []
                segment_boundaries = [0]
                
                for start_idx, end_idx in segments:
                    segment_signal = signal[start_idx:end_idx]
                    concatenated_signal.append(segment_signal)
                    segment_boundaries.append(segment_boundaries[-1] + (end_idx - start_idx))
                
                # Combine all segments horizontally
                combined_signal = np.hstack(concatenated_signal)
                
                # Generate SSWT with automatic parameter optimization
                # Let ssqueezepy automatically determine optimal parameters based on signal characteristics
                Tx, _, ssq_freqs, *_ = ssq_cwt(combined_signal, wavelet='morlet', fs=fs)
                
                magnitude = np.abs(np.asarray(Tx, dtype=complex))
                all_magnitudes.append(magnitude)
                
                # Store data for plotting
                context_data[context] = {
                    'magnitude': magnitude,
                    'freq_axis': np.array(ssq_freqs) if hasattr(ssq_freqs, '__iter__') else np.array([ssq_freqs]),
                    'segment_boundaries': segment_boundaries,
                    'num_segments': len(segments)
                }
            
            # Determine consistent color scale for this sensor
            if all_magnitudes:
                all_mags_combined = np.hstack([mag.flatten() for mag in all_magnitudes])
                vmin, vmax = np.percentile(all_mags_combined, [1, 99])  # Use 1-99 percentile for better contrast
            else:
                vmin, vmax = 0, 1
            
            # Second pass: plot with consistent color scale
            for context_idx, context in enumerate(analysis_contexts):
                ax = axes[sensor_idx][context_idx]
                
                if context_data[context] is None:
                    ax.text(0.5, 0.5, f'No {context}\nsegments found', 
                           ha='center', va='center', transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                    ax.set_title(f'{sensor} - {context}')
                    continue
                
                data = context_data[context]
                magnitude = data['magnitude']
                freq_axis = data['freq_axis']
                segment_boundaries = data['segment_boundaries']
                
                # Create time axis for concatenated data
                concat_time_axis = np.arange(magnitude.shape[1])
                T, F = np.meshgrid(concat_time_axis, freq_axis)
                
                # Plot concatenated spectrogram with consistent color scale
                im = ax.pcolormesh(T, F, magnitude, shading='gouraud', cmap='hot_r', 
                                 vmin=vmin, vmax=vmax)
                ax.set_title(f'{sensor} - {context} ({data["num_segments"]} segments)')
                ax.set_ylabel('Frequency [Hz]')
                
                # Add segment separator lines
                for boundary in segment_boundaries[1:-1]:  # Skip first (0) and last boundary
                    ax.axvline(x=boundary, color='white', linewidth=2, alpha=0.8, linestyle='--')
                
                # Add colorbar with consistent scale
                plt.colorbar(im, ax=ax, label='Magnitude')
                
                # Only show x-label on bottom row
                if sensor_idx == num_sensors - 1:
                    ax.set_xlabel('Concatenated Time Steps')
                    
        except Exception as e:
            print(f"Could not generate context-grouped SSWT for sensor '{sensor}': {e}")
            for context_idx in range(num_contexts):
                ax = axes[sensor_idx][context_idx]
                ax.text(0.5, 0.5, f"SSWT failed for {sensor}\nError: {str(e)}", 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    plt.tight_layout()
    
    output_path = os.path.join(DIAGNOSTICS_DIR, f"sim_{sim_id}_sswt_spectrograms_by_context.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Context-grouped SSWT Spectrogram plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots showing sensor data colored by road context.")
    parser.add_argument("--sim_id", type=int, required=True, help="The simulation ID to analyze")
    args = parser.parse_args()

    try:
        # Load data
        file_path = find_sim_file(args.sim_id)
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Process data
        df = process_data_for_diagnostics(df, SAMPLING_FREQUENCY)
        
        print(f"Data shape after processing: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # Check for context column (should be CONTEXT after processing)
        context_col = 'CONTEXT' if 'CONTEXT' in df.columns else 'context'
        if context_col not in df.columns:
            print("Error: No context column found in the data.")
            return
        
        # Create generic contexts and colors
        df['generic_context'] = df[context_col].apply(get_generic_context)
        unique_contexts = sorted(df['generic_context'].unique())
        context_colors = get_context_colors(unique_contexts)
        
        print(f"Found contexts: {unique_contexts}")
        
        # Check available sensors
        available_sensors = [sensor for sensor in SENSORS_TO_PLOT if sensor in df.columns]
        missing_sensors = [sensor for sensor in SENSORS_TO_PLOT if sensor not in df.columns]
        
        if missing_sensors:
            print(f"Warning: Missing sensors: {missing_sensors}")
        
        if not available_sensors:
            print(f"Error: None of the target sensors {SENSORS_TO_PLOT} are present in the data.")
            return
        
        print(f"Available sensors: {available_sensors}")
        
        # Generate plots
        plot_sensor_data_with_context(df, args.sim_id, available_sensors, context_colors)
        plot_sswt_spectrograms_by_context(df, args.sim_id, available_sensors, context_colors, SAMPLING_FREQUENCY)
        
        # Print context statistics
        context_counts = df[context_col].value_counts()
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