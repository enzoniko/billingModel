import os
import re
import glob
import random
import argparse
from collections import defaultdict
from typing import Tuple, List, Dict, Union
import pandas as pd
import numpy as np
import torch
from scipy.signal import butter, sosfilt
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.utils import to_time_series_dataset

# --- Configuration ---

# Paths
PROCESSED_DATA_DIR = "processed_data"
RESULTS_DIR = "results_experiment1"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment Parameters
FIXED_WINDOW_SIZE = 30
USE_PCA_PREPROCESSING = True
PREPROCESSING_PCA_COMPONENTS = 1
FEATURE_EXTRACTION_DOMAIN = 'time' # 'time' or 'frequency'

# Train/Validation Split Configuration
TRAIN_VALIDATION_SPLIT = 0.7  # 70% for training, 30% for validation

# Method Selection Configuration
METHODS_TO_RUN = ['kmeans']  # Options: 'dtw', 'kmeans', 'kmeans_dtw'
# METHODS_TO_RUN = ['dtw', 'kmeans', 'kmeans_dtw']  # Run all methods

# Define the mapping from simulation sequence to physics parameters
PHYSICS_GROUPS = {
    'group_1': {'mass': 8300, 'friction': 1.0, 'seq_range': range(1, 21)},
    'group_2': {'mass': 10900, 'friction': 1.0, 'seq_range': range(21, 41)},
    'group_3': {'mass': 13500, 'friction': 1.0, 'seq_range': range(41, 61)},
    'group_4': {'mass': 13500, 'friction': 1.0, 'seq_range': range(61, 81)},
    'group_5': {'mass': 10900, 'friction': 1.0, 'seq_range': range(81, 101)},
    'group_6': {'mass': 8300, 'friction': 0.75, 'seq_range': range(101, 121)},
    'group_7': {'mass': 10900, 'friction': 0.75, 'seq_range': range(121, 141)},
    'group_8': {'mass': 13500, 'friction': 0.75, 'seq_range': range(141, 161)},
    'group_9': {'mass': 13500, 'friction': 0.5, 'seq_range': range(161, 181)},
    'group_10': {'mass': 10900, 'friction': 0.5, 'seq_range': range(181, 201)},
    'group_11': {'mass': 8300, 'friction': 0.5, 'seq_range': range(201, 221)},
}

# Define the columns to be used as sensor data for clustering
# Excluding GPS (x,y,z), identifiers, and the target 'context' itself.
SENSOR_COLUMNS = [
    'IMU_ACC_x', 'IMU_ACC_Y', 'IMU_ACC_Z', 'SPEED', 'DRAG', 'GEAR',
    'ENGINE_RPM', 'YAW', 'PITCH', 'STEER', 'THROTTLE', 'BRAKE',
    'REVERSE', 'YAW_RATE', 'PITCH_RATE', 'ROLL_RATE'
]


# --- Step 1 (Helper): Ground-Truth Price Mapping ---

def get_price_for_context(context: str) -> float:
    """
    Calculates a heuristic price based on the context string.
    """
    if not isinstance(context, str):
        return 1.0  # Default for non-string contexts

    context = context.lower()
    if 'road' in context:
        return 1.0
    if 'ramp' in context:
        parts = context.split('_')
        try:
            # e.g., ramp_asc_2.5_90.5_1
            steepness = float(parts[2])
            if 'asc' in parts[1]:
                return 2.0 + steepness * 1.5
            elif 'desc' in parts[1]:
                return max(1.0, 2.0 - steepness * 1.0)
        except (IndexError, ValueError):
            return 3.0  # Default for malformed ramp
    if 'pothole' in context: return 15.0
    if 'speedbump' in context: return 8.0
    if 'elevated_crosswalk' in context: return 5.0
    if 'cut' in context: return 3.0
    return 3.0 # Default for any other unknown context


# --- Core Functions (Adapted from source) ---

def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Creates non-overlapping fixed-size windows from time series data."""
    num_timestamps = data.shape[0]
    num_windows = num_timestamps // window_size
    if num_windows == 0:
        return np.array([])
    return np.array([data[i * window_size:(i + 1) * window_size] for i in range(num_windows)])

def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes data to [0, 1] range based on global min/max."""
    min_values = np.min(data, axis=(0, 1), keepdims=True)
    max_values = np.max(data, axis=(0, 1), keepdims=True)
    # Avoid division by zero if a feature is constant
    range_values = max_values - min_values
    range_values[range_values == 0] = 1
    return (data - min_values) / range_values, min_values, max_values

def extract_features(signal: torch.Tensor, domain='frequency') -> torch.Tensor:
    """Extracts statistical features from signals."""
    if domain == 'frequency':
        # Apply FFT
        fft_signal = torch.fft.rfft(signal, dim=1).abs()
    else: # domain == 'time'
        fft_signal = signal

    features = torch.zeros((signal.shape[0], 11, signal.shape[2]), dtype=torch.float32, device=signal.device)
    features[:, 0, :] = torch.min(fft_signal, dim=1)[0]
    features[:, 1, :] = torch.max(fft_signal, dim=1)[0]
    features[:, 2, :] = torch.mean(fft_signal, dim=1)
    features[:, 3, :] = torch.sqrt(torch.mean(fft_signal**2, dim=1))
    features[:, 4, :] = torch.var(fft_signal, dim=1)
    m = torch.mean(fft_signal, dim=1, keepdim=True)
    std = torch.std(fft_signal, dim=1, keepdim=False)
    features[:, 5, :] = torch.mean((fft_signal - m)**3, dim=1) / (std**3 + 1e-8)
    features[:, 6, :] = torch.mean((fft_signal - m)**4, dim=1) / (std**4 + 1e-8)
    # Clean up potential NaNs and Infs
    features[torch.isnan(features)] = 0
    features[torch.isinf(features)] = 0
    return features

def preprocess_and_extract(signal: torch.Tensor, domain='time', scaler=None, pca=None) -> torch.Tensor:
    """Applies feature extraction and optional PCA."""
    features = extract_features(signal, domain)
    features_reshaped = features.view(features.size(0), -1).cpu().numpy()

    if scaler and pca:
        # Use existing scaler and PCA to transform data
        features_normalized = scaler.transform(features_reshaped)
        features_pca = pca.transform(features_normalized)
        return torch.tensor(features_pca, dtype=torch.float32).to(signal.device)

    elif USE_PCA_PREPROCESSING:
        # Fit new scaler and PCA
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_reshaped)
        pca = PCA(n_components=PREPROCESSING_PCA_COMPONENTS)
        features_pca = pca.fit_transform(features_normalized)
        return torch.tensor(features_pca, dtype=torch.float32).to(signal.device)
    else:
        # Return reshaped features without PCA
        return features.view(features.size(0), -1)

def preprocess_data(windowed_data: np.ndarray, domain='time', fit_preprocessors=True, scaler=None, pca=None) -> np.ndarray:
    """Converts windowed data to feature vectors."""
    if windowed_data.ndim == 2: # If a single window is passed
        windowed_data = windowed_data[np.newaxis, :, :]

    # Ensure data is in the correct format for PyTorch (B, T, F)
    data_tensor = torch.tensor(windowed_data, dtype=torch.float32)
    
    if fit_preprocessors:
        preprocessed_tensor = preprocess_and_extract(data_tensor, domain)
    else:
        preprocessed_tensor = extract_features(data_tensor, domain).view(data_tensor.size(0), -1)

    return preprocessed_tensor.cpu().numpy()

def compute_dtw_distances(data: Union[np.ndarray, list], centroids: Union[np.ndarray, list]) -> np.ndarray:
    """Computes DTW distance from each data sample to each centroid."""
    # data can be a 3D numpy array of fixed-length windows,
    # or a list of 2D numpy arrays for variable-length windows (centroids).
    distances = np.zeros((len(data), len(centroids)))
    for i, sample in enumerate(data):
        sample_arr = np.ascontiguousarray(sample, dtype=np.float64)
        for j, centroid in enumerate(centroids):
            centroid_arr = np.ascontiguousarray(centroid, dtype=np.float64)
            distances[i, j] = dtw(sample_arr, centroid_arr)
    return distances

def normalize_and_compute_weighted_average(distances: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Calculates a weighted average price based on normalized inverse distances."""
    # Handle case where a distance is zero to avoid division by zero
    if np.any(distances == 0):
        # If a window is identical to a centroid, assign that centroid's price
        avg = np.zeros(distances.shape[0])
        for i in range(distances.shape[0]):
            zero_dist_indices = np.where(distances[i] == 0)[0]
            if len(zero_dist_indices) > 0:
                avg[i] = np.mean(prices[zero_dist_indices])
            else: # Should not happen if the check passed, but as a fallback
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_distances = 1.0 / distances[i]
                normalized_inv_dist = inv_distances / np.sum(inv_distances)
                avg[i] = np.sum(normalized_inv_dist * prices)
        return avg
    
    with np.errstate(divide='ignore'):
        inv_distances = 1.0 / distances
    
    # Normalize the inverse distances to get weights
    normalized_weights = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)
    
    # Compute the weighted average of prices
    return np.sum(normalized_weights * prices, axis=1)


# --- Step 2 (Helper): Centroid Identification ---

def create_windows_from_context(df: pd.DataFrame, sensor_cols: list) -> Tuple[list, list]:
    """Extracts variable-length windows based on contiguous context blocks."""
    windows = []
    contexts = []
    if df.empty or 'context' not in df.columns:
        return windows, contexts

    current_context = df['context'].iloc[0]
    start_idx = 0
    for i in range(1, len(df)):
        if df['context'].iloc[i] != current_context:
            window_df = df.iloc[start_idx:i]
            if not window_df.empty:
                windows.append(window_df[sensor_cols].values)
                contexts.append(current_context)
            current_context = df['context'].iloc[i]
            start_idx = i

    # Add the last block
    last_window_df = df.iloc[start_idx:]
    if not last_window_df.empty:
        windows.append(last_window_df[sensor_cols].values)
        contexts.append(current_context)

    return windows, contexts

def get_generic_context(specific_context: str) -> str:
    """Strips unique IDs from context strings (e.g., ramp_..._1 -> ramp_...)."""
    if not isinstance(specific_context, str): return "unknown"
    # Find the last underscore followed by a number
    match = re.search(r'_\d+$', specific_context)
    if match:
        return specific_context[:match.start()]
    return specific_context

# --- Steps 3 & 4: Billing and Ground Truth Calculation ---

def calculate_prices(
    fixed_windows: np.ndarray,
    initial_centroids_raw: list,
    centroid_prices: np.ndarray,
    final_dtw_centroids: np.ndarray = None,
    final_kmeans_centroids: np.ndarray = None,
    final_hybrid_centroids: np.ndarray = None,
    scaler_kmeans: StandardScaler = None,
    pca_kmeans: PCA = None,
    scaler_hybrid: StandardScaler = None,
) -> dict:
    """Calculates prices for fixed-size windows using selected methods."""
    if fixed_windows.shape[0] == 0:
        return {method: np.array([]) for method in METHODS_TO_RUN}

    results = {}
    
    # Method 1: DTW (distance to final centroids from fitted model)
    if 'dtw' in METHODS_TO_RUN and final_dtw_centroids is not None:
        dtw_distances = compute_dtw_distances(fixed_windows, final_dtw_centroids)
        dtw_prices = normalize_and_compute_weighted_average(dtw_distances, centroid_prices)
        results["dtw"] = dtw_prices

    # Method 2: KMeans (on preprocessed features)
    if 'kmeans' in METHODS_TO_RUN and final_kmeans_centroids is not None and scaler_kmeans is not None and pca_kmeans is not None:
        # Preprocess incoming windows with the group-fitted scaler and PCA
        fixed_windows_features = preprocess_data(fixed_windows, domain=FEATURE_EXTRACTION_DOMAIN, fit_preprocessors=False)
        fixed_windows_transformed = pca_kmeans.transform(scaler_kmeans.transform(fixed_windows_features))
        kmeans_distances = np.linalg.norm(fixed_windows_transformed[:, np.newaxis, :] - final_kmeans_centroids, axis=2)
        kmeans_prices = normalize_and_compute_weighted_average(kmeans_distances, centroid_prices)
        results["kmeans"] = kmeans_prices
    
    # Method 3: KMeans with DTW
    if 'kmeans_dtw' in METHODS_TO_RUN and final_hybrid_centroids is not None and scaler_hybrid is not None:
        # Preprocess incoming windows into the hybrid feature space
        fixed_windows_features = preprocess_data(fixed_windows, domain=FEATURE_EXTRACTION_DOMAIN, fit_preprocessors=False)
        dtw_distances_to_initial = compute_dtw_distances(fixed_windows, initial_centroids_raw)
        hybrid_features_fixed_windows = np.hstack([fixed_windows_features, dtw_distances_to_initial])
        
        # Transform using the group-fitted hybrid scaler
        hybrid_windows_transformed = scaler_hybrid.transform(hybrid_features_fixed_windows)
        
        # Calculate distance to the final hybrid centroids
        hybrid_distances = np.linalg.norm(hybrid_windows_transformed[:, np.newaxis, :] - final_hybrid_centroids, axis=2)
        kmeans_dtw_prices = normalize_and_compute_weighted_average(hybrid_distances, centroid_prices)
        results["kmeans_dtw"] = kmeans_dtw_prices
    
    return results

def calculate_ground_truth_price(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """Calculates the ground-truth price for each fixed-size window."""
    num_windows = len(df) // window_size
    if num_windows == 0:
        return np.array([])
        
    context_prices = df['context'].apply(get_price_for_context).to_numpy()
    
    ground_truth_prices = [
        np.mean(context_prices[i * window_size : (i + 1) * window_size])
        for i in range(num_windows)
    ]
    return np.array(ground_truth_prices)


# --- Step 5: Evaluation and Analysis ---

def perform_evaluation_and_plot(
    results: dict, 
    ground_truth: np.ndarray, 
    group_name: str, 
    sim_id: int
) -> dict:
    """Calculates metrics and generates plots for a single run."""
    eval_results = {}
    for method, calculated in results.items():
        if len(calculated) != len(ground_truth) or len(calculated) == 0:
            print(f"  Skipping evaluation for {method} due to length mismatch or empty data.")
            continue

        # Numerical Evaluation
        corr, _ = pearsonr(calculated, ground_truth)
        mae = mean_absolute_error(ground_truth, calculated)
        rmse = np.sqrt(mean_squared_error(ground_truth, calculated))
        eval_results[method] = {"pearson_corr": corr, "mae": mae, "rmse": rmse}
        
        # Plotting
        plot_dir = os.path.join(RESULTS_DIR, group_name, method)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Scatter Plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=ground_truth, y=calculated, alpha=0.6)
        plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], 'r--', lw=2)
        plt.title(f'Scatter: {method.upper()} - Sim {sim_id}\nCorr: {corr:.3f}, MAE: {mae:.3f}')
        plt.xlabel("Ground-Truth Price")
        plt.ylabel("Calculated Price")
        plt.savefig(os.path.join(plot_dir, f"sim_{sim_id}_scatter.png"))
        plt.close()
        
        # Time-Series Tracking Plot
        plt.figure(figsize=(15, 6))
        plt.plot(ground_truth, label='Ground-Truth Price', color='black', lw=2)
        plt.plot(calculated, label=f'Calculated Price ({method.upper()})', color='blue', alpha=0.8)
        plt.title(f'Price Tracking: {method.upper()} - Sim {sim_id}')
        plt.xlabel("Window Index")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(plot_dir, f"sim_{sim_id}_tracking.png"))
        plt.close()
        
    return eval_results

def plot_group_summary_tracking(group_name: str, all_run_results: list):
    """
    Plots the concatenated ground truth vs. calculated prices for an entire physics group,
    ordered by ground-truth price.
    """
    if not all_run_results:
        return

    for method in METHODS_TO_RUN:
        # Concatenate results from all runs in the group
        ground_truth_all = np.concatenate([r['ground_truth'] for r in all_run_results if r])
        calculated_all = np.concatenate([r['results'][method] for r in all_run_results if r and method in r['results']])
        
        if len(ground_truth_all) == 0 or len(calculated_all) == 0:
            continue

        # Sort data by ground-truth price
        sort_indices = np.argsort(ground_truth_all)
        ground_truth_sorted = ground_truth_all[sort_indices]
        calculated_sorted = calculated_all[sort_indices]

        plt.figure(figsize=(20, 8))
        plt.plot(ground_truth_sorted, label='Ground-Truth Price', color='black', lw=2, alpha=0.7)
        plt.plot(calculated_sorted, label=f'Calculated Price ({method.upper()})', color='red', alpha=0.7, linestyle='--')
        
        # Add a diagonal reference line for perfect prediction
        plt.plot([ground_truth_sorted.min(), ground_truth_sorted.max()], 
                [ground_truth_sorted.min(), ground_truth_sorted.max()], 
                'g--', alpha=0.5, label='Perfect Prediction')

        plt.title(f'Group Summary Price Tracking: {group_name} - {method.upper()} (Sorted by Ground Truth)')
        plt.xlabel("Data Points (Sorted by Ground-Truth Price)")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = os.path.join(RESULTS_DIR, group_name, f"summary_tracking_{method}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  Group summary plot saved to: {output_path}")


# --- Main Execution ---

def main():
    """Main function to run the entire experiment."""
    parser = argparse.ArgumentParser(description="Run the billing validation experiment.")
    parser.add_argument(
        "--group",
        type=str,
        help="Run the experiment for a single specified physics group (e.g., 'group_1')."
    )
    args = parser.parse_args()

    groups_to_process = PHYSICS_GROUPS
    if args.group:
        if args.group in PHYSICS_GROUPS:
            groups_to_process = {args.group: PHYSICS_GROUPS[args.group]}
            print(f"--- Running experiment for single group: {args.group} ---")
        else:
            print(f"Error: Group '{args.group}' not found in PHYSICS_GROUPS. Available groups are: {list(PHYSICS_GROUPS.keys())}")
            return

    all_results_list = []

    for group_name, params in groups_to_process.items():
        print(f"\n--- Processing {group_name} (Mass: {params['mass']}, Friction: {params['friction']}) ---")
        
        # === Step 2: Centroid Identification ===
        
        # 1. Load Group Data
        group_csv_files = []
        for sim_num in params['seq_range']:
            # Correct glob pattern for filenames like 'simulation_1_...'
            pattern = os.path.join(PROCESSED_DATA_DIR, f"simulation_{sim_num}_*.csv")
            group_csv_files.extend(glob.glob(pattern))

        if not group_csv_files:
            print(f"  No CSV files found for {group_name}. Skipping.")
            continue
        
        # PROPER TRAIN/VALIDATION SPLIT
        group_csv_files.sort()  # Ensure consistent ordering
        n_train_files = int(len(group_csv_files) * TRAIN_VALIDATION_SPLIT)
        train_files = group_csv_files[:n_train_files]
        validation_files = group_csv_files[n_train_files:]
        
        print(f"  Split: {len(train_files)} training files, {len(validation_files)} validation files")
        
        if not train_files or not validation_files:
            print(f"  Insufficient files for train/validation split. Skipping group.")
            continue
        
        # Determine the actual sensor columns available in this group's data
        try:
            df_sample = pd.read_csv(train_files[0])
        except Exception as e:
            print(f"  Could not read sample file {train_files[0]}. Skipping group. Error: {e}")
            continue
            
        actual_sensor_columns = [col for col in SENSOR_COLUMNS if col in df_sample.columns]
        missing_cols = set(SENSOR_COLUMNS) - set(actual_sensor_columns)
        if missing_cols:
            print(f"  Info: The following sensors are not available in this group: {sorted(list(missing_cols))}")
        if not actual_sensor_columns:
            print(f"  No valid sensor columns found in this group's data. Skipping group.")
            continue

        # 2. Extract "Tailored" Windows and Normalize (TRAINING DATA ONLY)
        train_dfs = [pd.read_csv(f) for f in train_files]
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Use only the columns that are actually available for this group
        if train_df.empty or train_df[actual_sensor_columns].isnull().to_numpy().any():
            print(f"  Training data for {group_name} is empty or contains NaNs after concat. Skipping.")
            continue
        
        # Normalize ONLY the training data (crucial for proper validation)
        train_numeric_data = train_df[actual_sensor_columns].to_numpy()
        normalized_train_data, min_vals, max_vals = normalize_data(train_numeric_data)
        normalized_train_df = train_df.copy()
        normalized_train_df[actual_sensor_columns] = normalized_train_data
        
        tailored_windows, specific_contexts = create_windows_from_context(normalized_train_df, actual_sensor_columns)

        if not tailored_windows:
            print(f"  Could not extract any context-based windows for {group_name}. Skipping.")
            continue

        # 3. Group by Context Type
        grouped_windows = defaultdict(list)
        for window, context in zip(tailored_windows, specific_contexts):
            generic_ctx = get_generic_context(context)
            grouped_windows[generic_ctx].append(window)

        # 4. Select Initial Centroids
        initial_centroids = []
        centroid_contexts = []
        for ctx, w_list in grouped_windows.items():
            # Filter for windows that are long enough for statistical analysis
            valid_windows = [w for w in w_list if w.shape[0] > 1]
            if not valid_windows:
                print(f"  Warning: No valid windows (length > 1) for context '{ctx}'. Skipping this context for centroid selection.")
                continue

            selected_window = random.choice(valid_windows)
            initial_centroids.append(selected_window)
            centroid_contexts.append(ctx)
        
        if not initial_centroids:
            print(f"  No valid centroids could be found for {group_name}. Skipping group.")
            continue

        print(f"  Identified {len(initial_centroids)} unique contexts to use as centroids.")

        # 5. Set Up Pricing
        centroid_prices = np.array([get_price_for_context(c) for c in centroid_contexts])

        # === New for Exp2: Group-level Model and Preprocessor Fitting ===
        
        n_clusters = len(initial_centroids)
        
        # Initialize variables for models/preprocessors that might not be used
        final_dtw_centroids = None
        final_kmeans_centroids = None
        final_hybrid_centroids = None
        scaler_kmeans = None
        pca_kmeans = None
        scaler_hybrid = None
        
        # --- 1. DTW Model Fitting ---
        if 'dtw' in METHODS_TO_RUN:
            print("  Fitting DTW model for the group...")
            # tslearn expects a 3D array of shape (n_samples, n_timestamps, n_features)
            tslearn_formatted_tailored = to_time_series_dataset(tailored_windows)
            tslearn_formatted_initial = to_time_series_dataset(initial_centroids)
            dtw_model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', n_init=5, random_state=42, n_jobs=-1, init=tslearn_formatted_initial).fit(tslearn_formatted_tailored)
            final_dtw_centroids = dtw_model.cluster_centers_

        # --- 2. KMeans Model and Preprocessor Fitting ---
        if 'kmeans' in METHODS_TO_RUN or 'kmeans_dtw' in METHODS_TO_RUN:
            print("  Fitting KMeans preprocessors and model for the group...")
            # A. Extract features from all tailored windows
            # np.concatenate can't handle ragged arrays, so we must iterate
            all_windows_features_list = [preprocess_data(w, domain=FEATURE_EXTRACTION_DOMAIN, fit_preprocessors=False) for w in tailored_windows]
            all_windows_features = np.vstack(all_windows_features_list)
            
            # B. Fit Scaler and PCA on these features
            scaler_kmeans = StandardScaler().fit(all_windows_features)
            pca_kmeans = PCA(n_components=PREPROCESSING_PCA_COMPONENTS).fit(scaler_kmeans.transform(all_windows_features))

            if 'kmeans' in METHODS_TO_RUN:
                # C. Transform all windows and initial centroids into the new feature space
                transformed_windows_kmeans = pca_kmeans.transform(scaler_kmeans.transform(all_windows_features))
                initial_centroids_features = np.vstack([preprocess_data(c, domain=FEATURE_EXTRACTION_DOMAIN, fit_preprocessors=False) for c in initial_centroids])
                transformed_initial_centroids_kmeans = pca_kmeans.transform(scaler_kmeans.transform(initial_centroids_features))

                # D. Fit KMeans model
                kmeans_model = KMeans(n_clusters=n_clusters, init=transformed_initial_centroids_kmeans, n_init='auto', random_state=42).fit(transformed_windows_kmeans)
                final_kmeans_centroids = kmeans_model.cluster_centers_
        
        # --- 3. Hybrid KMeans Model and Preprocessor Fitting ---
        if 'kmeans_dtw' in METHODS_TO_RUN:
            print("  Fitting Hybrid KMeans preprocessors and model for the group...")
            # A. Get DTW distances from each tailored window to each raw initial centroid
            all_windows_dtw_distances = compute_dtw_distances(tailored_windows, initial_centroids)

            # B. Create the hybrid feature set
            all_windows_hybrid_features = np.hstack([all_windows_features, all_windows_dtw_distances])
            
            # C. Fit a new scaler for the hybrid features (no PCA for simplicity)
            scaler_hybrid = StandardScaler().fit(all_windows_hybrid_features)

            # D. Transform all hybrid features
            transformed_windows_hybrid = scaler_hybrid.transform(all_windows_hybrid_features)
            
            # E. Create and transform initial centroids for the hybrid model
            initial_centroids_dtw_distances = compute_dtw_distances(initial_centroids, initial_centroids)
            initial_centroids_hybrid_features = np.hstack([initial_centroids_features, initial_centroids_dtw_distances])
            transformed_initial_centroids_hybrid = scaler_hybrid.transform(initial_centroids_hybrid_features)

            # F. Fit the Hybrid KMeans model
            hybrid_model = KMeans(n_clusters=n_clusters, random_state=42).fit(transformed_windows_hybrid)
            final_hybrid_centroids = hybrid_model.cluster_centers_

        # === Steps 3, 4, 5: Runtime Simulation, Billing, and Evaluation ===
        
        # TRAINING EVALUATION: Check performance on training data
        print("  Evaluating performance on TRAINING data...")
        train_run_results_for_plotting = []
        for csv_file in train_files:
            sim_id_match = re.search(r'simulation_(\d+)_', os.path.basename(csv_file))
            if not sim_id_match: continue
            sim_id = int(sim_id_match.group(1))
            
            print(f"    Evaluating training simulation {sim_id}...")
            
            run_df = pd.read_csv(csv_file)
            if run_df.empty: continue

            # Create fixed-size windows from NON-normalized data first
            run_numeric_data = run_df[actual_sensor_columns].to_numpy()
            fixed_windows_raw = create_windows(run_numeric_data, FIXED_WINDOW_SIZE)
            if fixed_windows_raw.shape[0] == 0: continue

            # Normalize the windows using the group's min/max values
            fixed_windows_norm = (fixed_windows_raw - min_vals) / (max_vals - min_vals)

            # Calculate Price using trained models
            calculated_prices = calculate_prices(
                fixed_windows_norm,
                initial_centroids,
                centroid_prices,
                final_dtw_centroids,
                final_kmeans_centroids,
                final_hybrid_centroids,
                scaler_kmeans,
                pca_kmeans,
                scaler_hybrid
            )
            
            # Ground-Truth Price Calculation
            ground_truth_prices = calculate_ground_truth_price(run_df, FIXED_WINDOW_SIZE)

            # Evaluation and Analysis (Training)
            run_evals = perform_evaluation_and_plot(calculated_prices, ground_truth_prices, f"{group_name}_TRAIN", sim_id)
            
            # Store results for the group-level summary plot
            if calculated_prices and ground_truth_prices.size > 0:
                train_run_results_for_plotting.append({
                    'results': calculated_prices,
                    'ground_truth': ground_truth_prices
                })

            for method, metrics in run_evals.items():
                result_row = {
                    "group": group_name,
                    "mass": params['mass'],
                    "friction": params['friction'],
                    "sim_id": sim_id,
                    "method": method,
                    "data_type": "TRAINING",  # New column to distinguish training vs validation
                    **metrics
                }
                all_results_list.append(result_row)
        
        # VALIDATION EVALUATION: Test on validation files (never seen during training)
        print("  Evaluating performance on VALIDATION data...")
        validation_run_results_for_plotting = []
        for csv_file in validation_files:
            sim_id_match = re.search(r'simulation_(\d+)_', os.path.basename(csv_file))
            if not sim_id_match: continue
            sim_id = int(sim_id_match.group(1))
            
            print(f"    Evaluating validation simulation {sim_id}...")
            
            run_df = pd.read_csv(csv_file)
            if run_df.empty: continue

            # Create fixed-size windows from NON-normalized data first
            run_numeric_data = run_df[actual_sensor_columns].to_numpy()
            fixed_windows_raw = create_windows(run_numeric_data, FIXED_WINDOW_SIZE)
            if fixed_windows_raw.shape[0] == 0: continue

            # Normalize the windows using the group's min/max values
            fixed_windows_norm = (fixed_windows_raw - min_vals) / (max_vals - min_vals)

            # Calculate Price using trained models
            calculated_prices = calculate_prices(
                fixed_windows_norm,
                initial_centroids,
                centroid_prices,
                final_dtw_centroids,
                final_kmeans_centroids,
                final_hybrid_centroids,
                scaler_kmeans,
                pca_kmeans,
                scaler_hybrid
            )
            
            # Ground-Truth Price Calculation
            ground_truth_prices = calculate_ground_truth_price(run_df, FIXED_WINDOW_SIZE)

            # Evaluation and Analysis (Validation)
            run_evals = perform_evaluation_and_plot(calculated_prices, ground_truth_prices, f"{group_name}_VAL", sim_id)
            
            # Store results for the group-level summary plot
            if calculated_prices and ground_truth_prices.size > 0:
                validation_run_results_for_plotting.append({
                    'results': calculated_prices,
                    'ground_truth': ground_truth_prices
                })

            for method, metrics in run_evals.items():
                result_row = {
                    "group": group_name,
                    "mass": params['mass'],
                    "friction": params['friction'],
                    "sim_id": sim_id,
                    "method": method,
                    "data_type": "VALIDATION",  # New column to distinguish training vs validation
                    **metrics
                }
                all_results_list.append(result_row)
        
        # After processing all runs in a group, create the summary plots
        plot_group_summary_tracking(f"{group_name}_TRAIN", train_run_results_for_plotting)
        plot_group_summary_tracking(f"{group_name}_VAL", validation_run_results_for_plotting)
    
    # === Step 6: Final Comparison Across Physics Groups ===
    if not all_results_list:
        print("\nNo results were generated. Exiting.")
        return
        
    summary_df = pd.DataFrame(all_results_list)
    
    # Save the full results table
    summary_path = os.path.join(RESULTS_DIR, "full_evaluation_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nFull evaluation results saved to {summary_path}")

    # Create and print the summary table with separate training/validation metrics
    agg_metrics = summary_df.groupby(['mass', 'friction', 'method', 'data_type']).agg(
        mean_corr=('pearson_corr', 'mean'),
        std_corr=('pearson_corr', 'std'),
        mean_mae=('mae', 'mean'),
        std_mae=('mae', 'std')
    ).reset_index()
    
    agg_path = os.path.join(RESULTS_DIR, "summary_evaluation_by_group.csv")
    agg_metrics.to_csv(agg_path, index=False)
    print("\n--- Final Summary Table ---")
    print(agg_metrics.to_string())
    print(f"\nSummary table saved to {agg_path}")

if __name__ == "__main__":
    main() 