import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
import glob
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import the autoencoder model
from recurrent_autoencoder_anomaly_detection import VehicleAutoencoder, SENSORS_FOR_AUTOENCODER, WINDOW_SIZE

# Import SSWT for wavelet analysis
from ssqueezepy import ssq_cwt

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_data"
DIAGNOSTICS_DIR = "enhanced_diagnostics"
MODELS_DIR = "autoencoder_models"
RESULTS_DIR = "autoencoder_results"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

SAMPLING_FREQUENCY = 10  # Hz

# Use all sensors from autoencoder (now 17 sensors instead of 6)
SENSORS_TO_PLOT = SENSORS_FOR_AUTOENCODER

def get_generic_context(specific_context: str) -> str:
    """Strips unique IDs from context strings."""
    if not isinstance(specific_context, str): 
        return "unknown"
    
    if specific_context == "road":
        return "road"
    
    if specific_context.startswith("ramp_"):
        parts = specific_context.split('_')
        if len(parts) >= 2:
            return f"ramp_{parts[1]}"
        return "ramp"
    
    if specific_context == 'crash':
        return 'crash'
    
    for feature_type in ["pothole", "speedbump", "elevated_crosswalk", "cut"]:
        if specific_context.startswith(feature_type):
            return feature_type
    
    return "unknown"

def find_sim_file(sim_id: int) -> str:
    """Finds the CSV file for a given simulation ID."""
    pattern = os.path.join(PROCESSED_DATA_DIR, f"simulation_{sim_id}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No simulation file found for ID {sim_id}")
    return files[0]

def process_data_for_diagnostics(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    """Apply same preprocessing as original diagnostic_plot.py"""
    print("Processing data: Applying gravity correction and detecting crashes...")
    
    df.columns = [col.upper() for col in df.columns]

    if 'IMU_ACC_Z' in df.columns:
        df['IMU_ACC_Z_DYNAMIC'] = df['IMU_ACC_Z'] - 9.81
    else:
        df['IMU_ACC_Z_DYNAMIC'] = 0
        print("Warning: 'IMU_ACC_Z' not found. 'IMU_ACC_Z_DYNAMIC' set to 0.")

    if 'IMU_ACC_X' in df.columns and 'IMU_ACC_Y' in df.columns:
        df['ACC_HORIZONTAL'] = np.sqrt(df['IMU_ACC_X']**2 + df['IMU_ACC_Y']**2)
        crash_threshold = 10.0
        crash_window_seconds = 1.0
        window_size = int(crash_window_seconds * fs)
        
        crash_indices = df.index[df['ACC_HORIZONTAL'] > crash_threshold].tolist()
        
        if crash_indices:
            print(f"Found {len(crash_indices)} potential crash points. Applying context window...")
            
            if 'CONTEXT' not in df.columns:
                df['CONTEXT'] = 'unknown'
            
            df['CONTEXT'] = df['CONTEXT'].astype(object)

            for idx in crash_indices:
                start = max(0, idx - window_size)
                end = min(len(df), idx + window_size + 1)
                df.loc[start:end, 'CONTEXT'] = 'crash'
        else:
            print("No crash events detected.")
    else:
        print("Warning: Cannot detect crashes - missing IMU_ACC_X or IMU_ACC_Y")
        
    if 'CONTEXT' not in df.columns:
        df['CONTEXT'] = 'unknown'
        
    return df

def determine_vehicle_group(sim_id: int) -> str | None:
    """Determine which vehicle group a simulation belongs to."""
    # Based on controle.txt mapping
    group_ranges = {
        'group_1': range(1, 21),
        'group_2': range(21, 41),
        'group_3': range(41, 61),
        'group_4': range(61, 81),
        'group_5': range(81, 101),
        'group_6': range(101, 121),
        'group_7': range(121, 141),
        'group_8': range(141, 161),
        'group_9': range(161, 181),
        'group_10': range(181, 201),
        'group_11': range(201, 221),
    }
    
    for group_name, sim_range in group_ranges.items():
        if sim_id in sim_range:
            return group_name
    
    return None

def load_trained_model(group_name: str) -> tuple[VehicleAutoencoder, MinMaxScaler]:
    """Load the trained autoencoder model and scaler for a group."""
    # Load model
    model_path = os.path.join(MODELS_DIR, f"{group_name}_best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {group_name} at {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VehicleAutoencoder(input_size=len(SENSORS_FOR_AUTOENCODER))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load scaler
    results_path = os.path.join(RESULTS_DIR, f"{group_name}_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results file found for {group_name} at {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    scaler = results['scaler']
    return model, scaler

def calculate_reconstruction_errors_for_simulation(df: pd.DataFrame, model: VehicleAutoencoder, 
                                                 scaler: MinMaxScaler) -> pd.DataFrame:
    """Calculate reconstruction errors for the entire simulation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if all required sensors are available
    available_sensors = [sensor for sensor in SENSORS_FOR_AUTOENCODER if sensor in df.columns]
    if len(available_sensors) < len(SENSORS_FOR_AUTOENCODER):
        missing = set(SENSORS_FOR_AUTOENCODER) - set(available_sensors)
        raise ValueError(f"Missing required sensors: {missing}")
    
    # Extract sensor data
    sensor_data = df[SENSORS_FOR_AUTOENCODER].values
    
    # Initialize reconstruction error arrays
    reconstruction_errors = np.full((len(df), len(SENSORS_FOR_AUTOENCODER)), np.nan)
    
    # Process data in sliding windows
    for i in range(len(df) - WINDOW_SIZE + 1):
        window_data = sensor_data[i:i + WINDOW_SIZE]
        
        # Normalize window
        window_normalized = scaler.transform(window_data)
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0).to(device)  # [1, window_size, sensors]
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed = model(window_tensor)
        
        # Calculate errors for this window (MAE per sensor per timestep)
        original_tensor = window_tensor
        errors = torch.abs(reconstructed - original_tensor).squeeze(0).cpu().numpy()  # [window_size, sensors]
        
        # Assign errors to the corresponding positions (using the middle of the window)
        middle_idx = i + WINDOW_SIZE // 2
        if middle_idx < len(reconstruction_errors):
            reconstruction_errors[middle_idx] = errors[WINDOW_SIZE // 2]  # Use middle timestep of window
    
    # Add reconstruction error columns to dataframe
    df_with_errors = df.copy()
    for j, sensor in enumerate(SENSORS_FOR_AUTOENCODER):
        df_with_errors[f"{sensor}_reconstruction_error"] = reconstruction_errors[:, j]
    
    return df_with_errors

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

def plot_reconstruction_error_sswt_spectrograms_by_context(df: pd.DataFrame, sim_id: int, context_colors: dict, fs: int):
    """
    Generates SSWT spectrograms for reconstruction error signals grouped by context.
    Similar to the original diagnostic_plot.py but for reconstruction error signals.
    """
    
    # Find available reconstruction error signals
    available_error_signals = [col for col in df.columns if col.endswith('_reconstruction_error')]
    
    if not available_error_signals:
        print("No reconstruction error signals found for SSWT analysis")
        return
    
    print(f"Generating SSWT spectrograms for {len(available_error_signals)} reconstruction error signals...")
    
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
    
    num_signals = len(available_error_signals)
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
    total_width = sum(width_ratios) * 4  # Base width per unit
    fig = plt.figure(figsize=(total_width, 5 * num_signals))
    
    # Create GridSpec with custom width ratios
    gs = GridSpec(num_signals, num_contexts, figure=fig, width_ratios=width_ratios)
    
    # Create axes array manually
    axes = []
    for i in range(num_signals):
        row_axes = []
        for j in range(num_contexts):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Handle single cases for consistency
    if num_signals == 1 and num_contexts == 1:
        axes = [[axes[0][0]]]
    elif num_signals == 1:
        axes = [axes[0]]
    
    fig.suptitle(f'Reconstruction Error SSWT Spectrograms Grouped by Context - Simulation {sim_id}', fontsize=18)
    
    # Process each reconstruction error signal
    for signal_idx, error_signal in enumerate(available_error_signals):
        # Extract the base sensor name for display
        sensor_name = error_signal.replace('_reconstruction_error', '')
        
        # Get the reconstruction error values, filtering out NaN
        signal_data = df[error_signal].fillna(0).values
        
        try:
            print(f"Processing SSWT for {error_signal}...")
            
            # First pass: collect all magnitudes for this signal to determine consistent color scale
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
                    segment_signal = signal_data[start_idx:end_idx]
                    # Filter out any remaining NaN or inf values
                    segment_signal = segment_signal[np.isfinite(segment_signal)]
                    if len(segment_signal) > 0:
                        concatenated_signal.append(segment_signal)
                        segment_boundaries.append(segment_boundaries[-1] + len(segment_signal))
                
                if not concatenated_signal:
                    context_data[context] = None
                    continue
                
                # Combine all segments horizontally
                combined_signal = np.hstack(concatenated_signal)
                
                # Skip if signal is too short or all zeros
                if len(combined_signal) < 10 or np.all(combined_signal == 0):
                    context_data[context] = None
                    continue
                
                # Generate SSWT with automatic parameter optimization
                try:
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
                except Exception as e:
                    print(f"  SSWT failed for {error_signal} in context {context}: {e}")
                    context_data[context] = None
                    continue
            
            # Determine consistent color scale for this signal
            if all_magnitudes:
                all_mags_combined = np.hstack([mag.flatten() for mag in all_magnitudes])
                # Use more conservative percentiles for reconstruction errors (which are typically small)
                vmin, vmax = np.percentile(all_mags_combined, [5, 95])
                if vmax == vmin:  # Handle case where all values are the same
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = 0, 1
            
            # Second pass: plot with consistent color scale
            for context_idx, context in enumerate(analysis_contexts):
                ax = axes[signal_idx][context_idx]
                
                if context_data[context] is None:
                    ax.text(0.5, 0.5, f'No valid {context}\ndata for\n{sensor_name}', 
                           ha='center', va='center', transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                    ax.set_title(f'{sensor_name} Recon. Error - {context}')
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
                ax.set_title(f'{sensor_name} Recon. Error - {context}\n({data["num_segments"]} segments)')
                ax.set_ylabel('Frequency [Hz]')
                
                # Add segment separator lines
                for boundary in segment_boundaries[1:-1]:  # Skip first (0) and last boundary
                    ax.axvline(x=boundary, color='white', linewidth=2, alpha=0.8, linestyle='--')
                
                # Add colorbar with consistent scale
                plt.colorbar(im, ax=ax, label='Reconstruction Error Magnitude')
                
                # Only show x-label on bottom row
                if signal_idx == num_signals - 1:
                    ax.set_xlabel('Concatenated Time Steps')
                    
        except Exception as e:
            print(f"Could not generate SSWT for reconstruction error signal '{error_signal}': {e}")
            for context_idx in range(num_contexts):
                ax = axes[signal_idx][context_idx]
                ax.text(0.5, 0.5, f"SSWT failed for\n{sensor_name}\nRecon. Error\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    plt.tight_layout()
    
    output_path = os.path.join(DIAGNOSTICS_DIR, f"sim_{sim_id}_reconstruction_error_sswt_spectrograms_by_context.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction Error SSWT Spectrogram plot saved to: {output_path}")

def plot_enhanced_sensor_data_with_reconstruction_errors(df: pd.DataFrame, sim_id: int, context_colors: dict):
    """Plot both original sensors and reconstruction errors."""
    
    # Find available sensors and their reconstruction errors
    available_sensors = [sensor for sensor in SENSORS_TO_PLOT if sensor in df.columns]
    available_error_signals = [f"{sensor}_reconstruction_error" for sensor in available_sensors 
                              if f"{sensor}_reconstruction_error" in df.columns]
    
    if not available_sensors:
        print(f"Warning: No target sensors found in simulation {sim_id}")
        return
    
    print(f"Plotting enhanced time-series for {len(available_sensors)} sensors + {len(available_error_signals)} reconstruction errors...")
    
    # Create subplots: sensors + reconstruction errors
    total_signals = len(available_sensors) + len(available_error_signals)
    fig, axes = plt.subplots(total_signals, 1, figsize=(20, 3 * total_signals))
    if total_signals == 1:
        axes = [axes]
    
    fig.suptitle(f'Enhanced Sensor Data with Reconstruction Errors - Simulation {sim_id}', fontsize=16)
    
    time_points = np.arange(len(df))
    plot_idx = 0
    
    # Plot original sensors
    for sensor in available_sensors:
        ax = axes[plot_idx]
        sensor_values = df[sensor].values
        
        # Plot segments grouped by context
        current_context = None
        start_idx = 0
        
        for j in range(len(df) + 1):
            if j == len(df) or df['generic_context'].iloc[j] != current_context:
                if current_context is not None and start_idx < j:
                    segment_time = time_points[start_idx:j]
                    segment_values = sensor_values[start_idx:j]
                    
                    alpha_value = 0.4 if current_context == 'road' else 1.0
                    
                    ax.plot(segment_time, segment_values, 
                           color=context_colors[current_context], 
                           linewidth=1.5,
                           alpha=alpha_value,
                           label=current_context if current_context not in [line.get_label() for line in ax.lines] else "")
                
                if j < len(df):
                    current_context = df['generic_context'].iloc[j]
                    start_idx = j
        
        ax.set_title(f'{sensor} (Original Signal)')
        ax.set_ylabel('Sensor Value')
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plot_idx += 1
    
    # Plot reconstruction errors
    for error_signal in available_error_signals:
        ax = axes[plot_idx]
        error_values = df[error_signal].values
        
        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(error_values)
        valid_time = time_points[valid_mask]
        valid_errors = error_values[valid_mask]
        
        if len(valid_errors) == 0:
            ax.text(0.5, 0.5, 'No valid reconstruction error data', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # Plot error segments grouped by context
            current_context = None
            start_idx = 0
            
            for j in range(len(df) + 1):
                if j == len(df) or df['generic_context'].iloc[j] != current_context:
                    if current_context is not None and start_idx < j:
                        segment_mask = valid_mask[start_idx:j]
                        if np.any(segment_mask):
                            segment_indices = np.where(valid_mask[start_idx:j])[0] + start_idx
                            segment_time = time_points[segment_indices]
                            segment_errors = error_values[segment_indices]
                            
                            # Use different styling for reconstruction errors
                            ax.plot(segment_time, segment_errors, 
                                   color=context_colors[current_context], 
                                   linewidth=2,
                                   alpha=0.8,
                                   linestyle='--' if current_context == 'road' else '-')
                    
                    if j < len(df):
                        current_context = df['generic_context'].iloc[j]
                        start_idx = j
        
        sensor_name = error_signal.replace('_reconstruction_error', '')
        ax.set_title(f'{sensor_name} (Reconstruction Error - MAE)')
        ax.set_ylabel('Reconstruction Error')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Only show x-label on bottom plot
    axes[-1].set_xlabel('Time Step')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(DIAGNOSTICS_DIR, f"sim_{sim_id}_enhanced_sensors_with_reconstruction_errors.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced diagnostic plots with reconstruction errors")
    parser.add_argument("--sim_id", type=int, required=True, help="The simulation ID to analyze")
    args = parser.parse_args()

    try:
        # Determine which vehicle group this simulation belongs to
        group_name = determine_vehicle_group(args.sim_id)
        if group_name is None:
            print(f"Error: Simulation {args.sim_id} does not belong to any known vehicle group")
            return
        
        print(f"Simulation {args.sim_id} belongs to {group_name}")
        
        # Load trained model and scaler
        print(f"Loading trained autoencoder model for {group_name}...")
        try:
            model, scaler = load_trained_model(group_name)
            print("Model loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please train the autoencoder for {group_name} first using:")
            print(f"python recurrent_autoencoder_anomaly_detection.py --group {group_name}")
            return
        
        # Load simulation data
        file_path = find_sim_file(args.sim_id)
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Process data (same as original diagnostic_plot.py)
        df = process_data_for_diagnostics(df, SAMPLING_FREQUENCY)
        
        print(f"Data shape after processing: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # Calculate reconstruction errors
        print("Calculating reconstruction errors...")
        df_with_errors = calculate_reconstruction_errors_for_simulation(df, model, scaler)
        
        # Create generic contexts and colors
        context_col = 'CONTEXT' if 'CONTEXT' in df_with_errors.columns else 'context'
        df_with_errors['generic_context'] = df_with_errors[context_col].apply(get_generic_context)
        unique_contexts = sorted(df_with_errors['generic_context'].unique())
        context_colors = get_context_colors(unique_contexts)
        
        print(f"Found contexts: {unique_contexts}")
        
        # Generate enhanced plots
        plot_enhanced_sensor_data_with_reconstruction_errors(df_with_errors, args.sim_id, context_colors)
        plot_reconstruction_error_sswt_spectrograms_by_context(df_with_errors, args.sim_id, context_colors, SAMPLING_FREQUENCY)
        
        # Print reconstruction error statistics
        print(f"\nReconstruction Error Statistics:")
        error_columns = [col for col in df_with_errors.columns if col.endswith('_reconstruction_error')]
        for col in error_columns:
            errors = df_with_errors[col].dropna()
            if len(errors) > 0:
                print(f"  {col}:")
                print(f"    Mean: {errors.mean():.6f}")
                print(f"    Std:  {errors.std():.6f}")
                print(f"    Min:  {errors.min():.6f}")
                print(f"    Max:  {errors.max():.6f}")
                
                # Context-specific statistics
                for context in unique_contexts:
                    context_mask = df_with_errors['generic_context'] == context
                    context_errors = df_with_errors.loc[context_mask, col].dropna()
                    if len(context_errors) > 0:
                        print(f"    {context}: Mean={context_errors.mean():.6f}, Std={context_errors.std():.6f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 