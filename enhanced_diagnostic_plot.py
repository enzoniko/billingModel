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

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_data"
DIAGNOSTICS_DIR = "enhanced_diagnostics"
MODELS_DIR = "autoencoder_models"
RESULTS_DIR = "autoencoder_results"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

SAMPLING_FREQUENCY = 10  # Hz

# Key sensors to plot for comparison (from original diagnostic_plot.py)
SENSORS_TO_PLOT = [
    'IMU_ACC_Z_DYNAMIC', 'IMU_ACC_X', 'IMU_ACC_Y', 
    'ENGINE_RPM', 'SPEED', 'THROTTLE'
]

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
        
        # Generate enhanced plot
        plot_enhanced_sensor_data_with_reconstruction_errors(df_with_errors, args.sim_id, context_colors)
        
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