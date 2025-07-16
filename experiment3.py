import os
import re
import glob
import random
import argparse
from collections import defaultdict, Counter
from typing import Tuple, List, Dict, Union, Optional
import pandas as pd
import numpy as np
import torch
import pickle
from scipy.signal import butter, sosfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")
try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    print("Warning: KMedoids not available. Install with: pip install scikit-learn-extra")
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy.fft import fft

# Import autoencoder components
from recurrent_autoencoder_anomaly_detection import VehicleAutoencoder, SENSORS_FOR_AUTOENCODER, WINDOW_SIZE

# --- Configuration ---

# Paths
PROCESSED_DATA_DIR = "processed_data"
RESULTS_DIR = "results_experiment3" # New directory for this experiment
MODELS_DIR = "autoencoder_models"
AUTOENCODER_RESULTS_DIR = "autoencoder_results"
ERROR_CACHE_DIR = "reconstruction_error_cache" # Reuse cache
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ERROR_CACHE_DIR, exist_ok=True)

# Experiment Parameters
FIXED_WINDOW_SIZE = 30

# --- NEW: Data Augmentation and Balancing Configuration ---
# This is the core of the new, valid methodology.
# We create a perfectly balanced dataset by augmenting "pure" context samples.
AUGMENTATION_CONFIG = {
    'enabled': True,
    'samples_per_class': 100, # Generate this many augmented samples for each unique context
    'noise_level': 0.02       # Add Gaussian noise with std = 2% of the feature's std
}

# Enhanced Dimensionality Reduction Configuration
DIMENSIONALITY_REDUCTION = {
    'method': 'intelligent_pca',
    'target_variance': 0.95,
    'max_components': 100,
    'min_components': 10
}

# Enhanced Pricing Strategy Configuration
PRICING_STRATEGY = {
    'method': 'exponential_weighting', # More robust than thresholding
    'exponential_power': 3.0,
    'min_weight_threshold': 0.01
}

# Train/Validation Split Configuration
TRAIN_VALIDATION_SPLIT = 0.7

# Fast Validation Configuration
FAST_VALIDATION = {
    'enabled': True,
    'max_train_files': 5,
    'max_val_files': 3,
    'max_windows_per_file': 100,
    'skip_plots': False,
}

# Method Selection Configuration
METHODS_TO_RUN = ['kmedoids', 'kmeans', 'baseline']

# Enhanced Feature Parameters
WAVELET_TYPE = 'morl'
WAVELET_SCALES = np.arange(1, 32)

# Physics Groups
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

RECONSTRUCTION_ERROR_COLUMNS = [f"{sensor}_reconstruction_error" for sensor in SENSORS_FOR_AUTOENCODER]

# --- Step 1 (Helper): Ground-Truth Price Mapping ---

def get_price_for_context(context: str) -> float:
    """Calculates a heuristic price based on the context string."""
    if not isinstance(context, str): return 1.0
    context = context.lower()
    if 'road' in context: return 1.0
    if 'ramp' in context:
        parts = context.split('_')
        try:
            steepness = float(parts[2])
            return (2.0 + steepness * 1.5) if 'asc' in parts[1] else max(1.0, 2.0 - steepness * 1.0)
        except (IndexError, ValueError): return 3.0
    if 'crash' in context: return 50.0
    if 'pothole' in context: return 15.0
    if 'speedbump' in context: return 8.0
    if 'elevated_crosswalk' in context: return 5.0
    if 'cut' in context: return 3.0
    return 3.0

# --- Autoencoder Integration & Caching (from experiment2.py) ---

def load_trained_autoencoder(group_name: str) -> Tuple[VehicleAutoencoder, MinMaxScaler]:
    """Load the trained autoencoder model and scaler for a group."""
    model_path = os.path.join(MODELS_DIR, f"{group_name}_best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {group_name} at {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VehicleAutoencoder(input_size=len(SENSORS_FOR_AUTOENCODER))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    results_path = os.path.join(AUTOENCODER_RESULTS_DIR, f"{group_name}_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results file found for {group_name} at {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return model, results['scaler']

def process_data_for_autoencoder(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as autoencoder training."""
    df.columns = [col.upper() for col in df.columns]
    if 'IMU_ACC_Z' in df.columns:
        df['IMU_ACC_Z_DYNAMIC'] = df['IMU_ACC_Z'] - 9.81
    else:
        df['IMU_ACC_Z_DYNAMIC'] = 0
    if 'IMU_ACC_X' in df.columns and 'IMU_ACC_Y' in df.columns:
        df['ACC_HORIZONTAL'] = np.sqrt(df['IMU_ACC_X']**2 + df['IMU_ACC_Y']**2)
        crash_threshold = 10.0
        crash_window_size = int(1.0 * 10) # 1 second at 10Hz
        crash_indices = df.index[df['ACC_HORIZONTAL'] > crash_threshold].tolist()
        if crash_indices:
            if 'CONTEXT' not in df.columns: df['CONTEXT'] = df.get('context', 'unknown')
            df['CONTEXT'] = df['CONTEXT'].astype(object)
            for idx in crash_indices:
                start, end = max(0, idx - crash_window_size), min(len(df), idx + crash_window_size + 1)
                df.loc[start:end, 'CONTEXT'] = 'crash'
    if 'CONTEXT' not in df.columns: df['CONTEXT'] = df.get('context', 'unknown')
    return df

def get_or_generate_error_windows(
    csv_file: str, model: VehicleAutoencoder, scaler: MinMaxScaler, group_name: str
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Generates or loads cached reconstruction error windows for a given simulation file."""
    sim_id_match = re.search(r'simulation_(\d+)_', os.path.basename(csv_file))
    if not sim_id_match: return None, None
    sim_id = sim_id_match.group(1)
    cache_path = os.path.join(ERROR_CACHE_DIR, f"{group_name}_sim_{sim_id}_error_windows.pkl")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_path}. Regenerating. Error: {e}")

    try:
        df = pd.read_csv(csv_file)
        df = process_data_for_autoencoder(df)
        if not all(sensor in df.columns for sensor in SENSORS_FOR_AUTOENCODER): return None, None
        
        # Use sliding windows with stride 1 (same behaviour as reference script)
        sensor_data = df[SENSORS_FOR_AUTOENCODER].to_numpy(dtype=float, copy=False)

        # Total number of possible windows when sliding one sample at a time
        num_windows = len(df) - WINDOW_SIZE + 1
        if num_windows <= 0:
            return None, None

        # Build a (num_windows, WINDOW_SIZE, n_signals) view. Prefer stride trick for speed, fall back to list comp.
        try:
            raw_windows = np.lib.stride_tricks.sliding_window_view(
                sensor_data, (WINDOW_SIZE, sensor_data.shape[1])
            )
            raw_windows = raw_windows.reshape(num_windows, WINDOW_SIZE, sensor_data.shape[1])
        except AttributeError:
            # Older NumPy versions may not have sliding_window_view
            raw_windows = np.array([sensor_data[i:i + WINDOW_SIZE] for i in range(num_windows)])

        # Determine context for each sliding window using the same boundaries
        context_col = 'CONTEXT' if 'CONTEXT' in df.columns else 'context'
        window_contexts = [
            assign_context_to_fixed_window(
                df.iloc[i:i + WINDOW_SIZE][context_col].values,
                'majority_expensive'
            )
            for i in range(num_windows)
        ]

        error_windows = np.zeros_like(raw_windows, dtype=np.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(num_windows):
            window_normalized = scaler.transform(raw_windows[i])
            window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstructed_tensor = model(window_tensor)
            error_windows[i] = torch.abs(reconstructed_tensor - window_tensor).squeeze(0).cpu().numpy()

        with open(cache_path, 'wb') as f: pickle.dump((error_windows, window_contexts), f)
        return error_windows, window_contexts
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None, None

# --- Feature Extraction (from experiment2.py) ---

def extract_time_domain_features(signal: np.ndarray) -> np.ndarray:
    """Extract time-domain statistical features."""
    features = [np.min(signal), np.max(signal), np.mean(signal), np.std(signal), np.var(signal)]
    mean_val, std_val = np.mean(signal), np.std(signal)
    if std_val > 1e-8:
        features.append(np.mean(((signal - mean_val) / std_val) ** 3)) # Skewness
        features.append(np.mean(((signal - mean_val) / std_val) ** 4)) # Kurtosis
    else:
        features.extend([0.0, 0.0])
    features.append(np.sum(signal ** 2)) # Energy
    features.append(np.sqrt(np.mean(signal ** 2))) # RMS
    features.append(np.sum(np.diff(np.sign(signal)) != 0) / len(signal)) # Zero crossing rate
    features.append(np.max(signal) - np.min(signal)) # Peak to peak
    return np.array(features)

def extract_enhanced_fft_features(signal: np.ndarray, fs: float = 10.0) -> np.ndarray:
    """Extract enhanced FFT features."""
    if len(signal) < 2: return np.zeros(7)
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    pos_freqs, pos_fft = freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2]
    if len(pos_freqs) == 0: return np.zeros(7)
    
    features = []
    sum_fft = np.sum(pos_fft)
    spec_centroid = np.sum(pos_freqs * pos_fft) / sum_fft if sum_fft > 0 else 0.0
    features.append(spec_centroid)
    
    cum_sum = np.cumsum(pos_fft)
    total_energy = cum_sum[-1] if len(cum_sum) > 0 else 0
    if total_energy > 0:
        rolloff_idx = np.where(cum_sum >= 0.95 * total_energy)[0]
        spec_rolloff = pos_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else pos_freqs[-1]
    else: spec_rolloff = 0.0
    features.append(spec_rolloff)

    spec_spread = np.sqrt(np.sum(((pos_freqs - spec_centroid) ** 2) * pos_fft) / sum_fft) if sum_fft > 0 else 0.0
    features.append(spec_spread)
    features.append(np.sum(np.diff(pos_fft) ** 2)) # Spectral flux
    features.append(pos_freqs[np.argmax(pos_fft)] if len(pos_fft) > 0 else 0.0) # Dominant freq
    features.append(np.sum(pos_fft ** 2)) # Spectral energy
    
    norm_fft = pos_fft / sum_fft if sum_fft > 0 else pos_fft
    spec_entropy = -np.sum(norm_fft * np.log2(norm_fft + 1e-12)) if sum_fft > 0 else 0.0
    features.append(spec_entropy)
    return np.array(features)

def extract_morlet_wavelet_features(signal: np.ndarray, scales: np.ndarray = WAVELET_SCALES, sampling_period: float = 0.1) -> np.ndarray:
    """Extract Morlet wavelet features."""
    expected_len = len(WAVELET_SCALES) + 8
    if len(signal) < 2: return np.zeros(expected_len)
    try:
        cwt_result = pywt.cwt(signal, scales, WAVELET_TYPE, sampling_period=sampling_period)
        coeffs, freqs = cwt_result[0], cwt_result[1]
        scale_energies = np.sum(np.abs(coeffs) ** 2, axis=1)
        features = scale_energies.tolist()
        features.append(np.sum(scale_energies))
        features.append(scales[np.argmax(scale_energies)])
        
        sum_energies = np.sum(scale_energies)
        mean_freq = np.sum(freqs.reshape(-1, 1) * np.abs(coeffs)**2) / sum_energies if sum_energies > 0 else 0.0
        features.append(mean_freq)
        
        norm_energies = scale_energies / sum_energies if sum_energies > 0 else scale_energies
        wavelet_entropy = -np.sum(norm_energies * np.log2(norm_energies + 1e-12)) if sum_energies > 0 else 0.0
        features.append(wavelet_entropy)
        
        features.extend(np.percentile(scale_energies, [25, 50, 75]) if len(scale_energies) > 0 else [0.0, 0.0, 0.0])
        features.append(np.sum(scale_energies > np.mean(scale_energies)))
        return np.array(features)
    except Exception as e:
        print(f"Warning: Wavelet feature extraction failed: {e}")
        return np.zeros(expected_len)

def extract_enhanced_features_from_reconstruction_errors(windowed_data: np.ndarray) -> np.ndarray:
    """Extracts a comprehensive feature set from windowed reconstruction error signals."""
    if windowed_data.ndim == 2: windowed_data = windowed_data[np.newaxis, :, :]
    num_windows, _, num_signals = windowed_data.shape
    
    time_f, fft_f, wave_f = 11, 7, len(WAVELET_SCALES) + 8
    total_f_per_signal = time_f + fft_f + wave_f
    all_features = np.zeros((num_windows, num_signals * total_f_per_signal))

    for win_idx in range(num_windows):
        feature_offset = 0
        for sig_idx in range(num_signals):
            signal = windowed_data[win_idx, :, sig_idx]
            if np.any(np.isnan(signal)): signal = np.nan_to_num(signal, nan=0.0)
            
            if len(signal) > 0:
                time_features = extract_time_domain_features(signal)
                fft_features = extract_enhanced_fft_features(signal)
                wavelet_features = extract_morlet_wavelet_features(signal)
                signal_features = np.concatenate([time_features, fft_features, wavelet_features])
            else:
                signal_features = np.zeros(total_f_per_signal)
            
            end_idx = feature_offset + len(signal_features)
            all_features[win_idx, feature_offset:end_idx] = signal_features
            feature_offset = end_idx
            
    return np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)

# --- Dimensionality Reduction (from experiment2.py) ---

def intelligent_pca_reduction(features: np.ndarray, target_variance: float, max_components: int, min_components: int) -> Tuple[PCA, int]:
    """Intelligently determines the optimal number of PCA components."""
    n_samples, n_features = features.shape
    max_possible = min(n_samples - 1, n_features, max_components)
    pca_full = PCA(n_components=max_possible).fit(features)
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumsum_variance >= target_variance)) + 1
    n_components = max(min_components, min(n_components, max_components))
    print(f"  Intelligent PCA: {n_components} components retain {cumsum_variance[n_components-1]:.3f} variance")
    pca_final = PCA(n_components=n_components).fit(features)
    return pca_final, n_components

def apply_dimensionality_reduction(features: np.ndarray, config: dict) -> Tuple[np.ndarray, PCA]:
    """Applies the configured dimensionality reduction method."""
    method = config['method']
    print(f"🔧 Applying dimensionality reduction: {method} on features of shape {features.shape}")
    if method == 'intelligent_pca':
        reducer, _ = intelligent_pca_reduction(
            features, config['target_variance'], config['max_components'], config['min_components']
        )
        reduced_features = reducer.transform(features)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    print(f"✅ Final reduced feature space: {reduced_features.shape}")
    return reduced_features, reducer

def transform_new_features(features: np.ndarray, reducer: PCA) -> np.ndarray:
    """Transforms new features using a fitted dimensionality reducer."""
    return reducer.transform(features)

# --- Core Data Handling (from experiment2_old.py) ---

def assign_context_to_fixed_window(contexts: Union[np.ndarray, list], strategy: str = 'majority_expensive') -> str:
    """Assigns a single context label to a fixed window."""
    if len(contexts) == 0: return 'unknown'
    context_list = contexts.tolist() if isinstance(contexts, np.ndarray) else list(contexts)
    context_counts = Counter(context_list)
    if strategy == 'expensive':
        context_prices = [get_price_for_context(ctx) for ctx in context_list]
        return context_list[np.argmax(context_prices)]
    # 'majority_expensive'
    max_count = context_counts.most_common(1)[0][1]
    tied_contexts = [ctx for ctx, count in context_counts.items() if count == max_count]
    if len(tied_contexts) == 1: return tied_contexts[0]
    tied_prices = [get_price_for_context(ctx) for ctx in tied_contexts]
    return tied_contexts[np.argmax(tied_prices)]

def extract_contiguous_context_blocks(df: pd.DataFrame, error_cols: list, context_col: str) -> Dict[str, List[np.ndarray]]:
    """Extracts contiguous blocks of data for each context type."""
    context_blocks = defaultdict(list)
    if df.empty or context_col not in df.columns: return dict(context_blocks)
    
    current_context = df[context_col].iloc[0]
    start_idx = 0
    for i in range(1, len(df)):
        if df[context_col].iloc[i] != current_context:
            block_data = df.iloc[start_idx:i][error_cols].values
            if len(block_data) > 0: context_blocks[current_context].append(block_data)
            current_context = df[context_col].iloc[i]
            start_idx = i
    final_block_data = df.iloc[start_idx:][error_cols].values
    if len(final_block_data) > 0: context_blocks[current_context].append(final_block_data)
    return dict(context_blocks)

def create_context_representative_samples(context_blocks: Dict[str, List[np.ndarray]], target_length: int) -> Tuple[List[np.ndarray], List[str]]:
    """Creates 'pure' representative samples for each context."""
    rep_samples, context_labels = [], []
    print(f"  Creating {target_length}-measurement 'pure' representative samples for each context...")
    for context, blocks in context_blocks.items():
        if not blocks: continue
        blocks_sorted = sorted(blocks, key=len, reverse=True)
        
        # Find a suitable block
        sample_created = False
        for block in blocks_sorted:
            if len(block) >= target_length:
                rep_samples.append(block[:target_length])
                context_labels.append(context)
                sample_created = True
                break
        if not sample_created: # If no block was long enough, concatenate or repeat
            concatenated = np.vstack(blocks) if blocks else np.array([])
            if len(concatenated) >= target_length:
                rep_samples.append(concatenated[:target_length])
                context_labels.append(context)
            elif len(concatenated) > 0:
                repeats = (target_length // len(concatenated)) + 1
                rep_samples.append((concatenated.tolist() * repeats)[:target_length])
                context_labels.append(context)
    
    print(f"  Created {len(rep_samples)} unique 'pure' representative samples.")
    return rep_samples, context_labels

# --- NEW: Data Augmentation and Balancing (Core of the Fix) ---

def augment_and_balance_data(pure_samples: List[np.ndarray], contexts: List[str], config: dict) -> Tuple[List[np.ndarray], List[str]]:
    """
    Creates a large, perfectly balanced dataset by augmenting 'pure' samples with noise.
    This is a valid and robust way to handle class imbalance without SMOTE.
    """
    if not config['enabled']:
        return pure_samples, contexts

    print(f"🚀 Augmenting data to create a balanced training set...")
    print(f"   Samples per class: {config['samples_per_class']}, Noise level: {config['noise_level']}")
    
    augmented_samples, augmented_labels = [], []
    
    # Group pure samples by their generic context
    grouped_pure_samples = defaultdict(list)
    for sample, context in zip(pure_samples, contexts):
        generic_ctx = get_generic_context(context)
        grouped_pure_samples[generic_ctx].append(sample)

    for generic_ctx, sample_list in grouped_pure_samples.items():
        # Use the first sample as the base for this generic context
        base_sample = sample_list[0]
        
        # Add the original pure sample
        augmented_samples.append(base_sample)
        augmented_labels.append(generic_ctx)
        
        # Calculate noise scale once per feature
        noise_scale = np.std(base_sample, axis=0) * config['noise_level']
        noise_scale = np.where(noise_scale == 0, config['noise_level'] * 0.01, noise_scale) # Avoid zero std

        # Generate augmented samples
        for _ in range(config['samples_per_class'] - 1):
            noise = np.random.normal(0, noise_scale, size=base_sample.shape)
            augmented_sample = base_sample + noise
            augmented_samples.append(augmented_sample)
            augmented_labels.append(generic_ctx)
            
    print(f"   Original pure samples: {len(pure_samples)}")
    print(f"   Total augmented samples: {len(augmented_samples)}")
    print(f"   Balanced class distribution: {dict(Counter(augmented_labels))}")
    
    return augmented_samples, augmented_labels

def get_generic_context(specific_context: str) -> str:
    """Strips unique IDs from context strings (e.g., ramp_..._1 -> ramp_...)."""
    if not isinstance(specific_context, str): return "unknown"
    match = re.search(r'_\d+$', specific_context)
    return specific_context[:match.start()] if match else specific_context

# --- Clustering and Pricing (from experiment2.py) ---

def handle_single_element_clusters(centroids: np.ndarray, labels: np.ndarray, features: np.ndarray, min_cluster_size: int = 2, is_kmedoids: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Merges clusters with fewer than `min_cluster_size` samples."""
    cluster_counts = Counter(labels)
    small_clusters = [cid for cid, count in cluster_counts.items() if count < min_cluster_size]
    if not small_clusters: return centroids, labels

    print(f"  Found {len(small_clusters)} small clusters to merge: {small_clusters}")
    updated_labels = labels.copy()
    for small_cid in small_clusters:
        small_centroid = centroids[small_cid]
        distances = [np.linalg.norm(centroids[cid] - small_centroid) for cid in range(len(centroids)) if cid not in small_clusters]
        valid_cids = [cid for cid in range(len(centroids)) if cid not in small_clusters]
        if valid_cids:
            nearest_cid = valid_cids[np.argmin(distances)]
            updated_labels[labels == small_cid] = nearest_cid
    
    # Recalculate centroids and remap labels to be consecutive
    unique_labels = np.unique(updated_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[l] for l in updated_labels])
    
    final_centroids = np.zeros((len(unique_labels), features.shape[1]))
    for new_label, old_label in enumerate(unique_labels):
        cluster_features = features[updated_labels == old_label]
        if is_kmedoids:
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(cluster_features, metric='euclidean')
            medoid_idx = np.argmin(np.sum(dist_matrix, axis=1))
            final_centroids[new_label] = cluster_features[medoid_idx]
        else:
            final_centroids[new_label] = np.mean(cluster_features, axis=0)
            
    print(f"  Final clusters after merging: {len(final_centroids)}")
    return final_centroids, final_labels

def calculate_prices_with_strategy(distances: np.ndarray, prices: np.ndarray, strategy_config: dict) -> np.ndarray:
    """Calculates prices using the configured strategy."""
    if distances.shape[0] == 0: return np.array([])
    
    power = strategy_config['exponential_power']
    min_weight = strategy_config['min_weight_threshold']
    calculated_prices = np.zeros(distances.shape[0])
    
    for i in range(distances.shape[0]):
        weights = np.exp(-power * distances[i])
        valid_mask = weights >= min_weight
        if not np.any(valid_mask):
            calculated_prices[i] = prices[np.argmin(distances[i])]
        else:
            valid_weights = weights[valid_mask]
            valid_prices = prices[valid_mask]
            calculated_prices[i] = np.sum((valid_weights / np.sum(valid_weights)) * valid_prices)
            
    return calculated_prices

def calculate_cluster_prices(fixed_windows: np.ndarray, centroid_prices: np.ndarray, final_centroids: np.ndarray, scaler: StandardScaler, reducer: PCA) -> np.ndarray:
    """Calculates prices for fixed-size windows for a single clustering model."""
    if fixed_windows.shape[0] == 0: return np.array([])
    
    features = extract_enhanced_features_from_reconstruction_errors(fixed_windows)
    scaled_features = scaler.transform(features)
    reduced_features = transform_new_features(scaled_features, reducer)
    
    distances = np.linalg.norm(reduced_features[:, np.newaxis, :] - final_centroids, axis=2)
    return calculate_prices_with_strategy(distances, centroid_prices, PRICING_STRATEGY)

def calculate_baseline_prices(fixed_windows: np.ndarray) -> np.ndarray:
    """Baseline pricing using raw reconstruction error magnitudes (RMS)."""
    if fixed_windows.shape[0] == 0: return np.array([])
    rms = np.sqrt(np.mean(fixed_windows**2, axis=(1, 2)))
    rms = np.nan_to_num(rms)
    if np.max(rms) > np.min(rms):
        return 1.0 + 49.0 * (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    return np.full(len(rms), 25.0)

# --- Evaluation (HONEST and DIRECT) ---

def perform_evaluation_and_plot(results: dict, ground_truth: np.ndarray, group_name: str, sim_id: int) -> dict:
    """
    Calculates metrics and generates plots for a single run.
    This version uses ONLY direct evaluation, without any invalid transformations.
    """
    eval_results = {}
    for method, calculated in results.items():
        if len(calculated) != len(ground_truth) or len(calculated) == 0: continue

        # --- DIRECT EVALUATION ---
        # We calculate correlation and error on the raw, untransformed model output.
        # This is the scientifically valid way to assess performance.
        corr, _ = pearsonr(calculated, ground_truth)
        mae = mean_absolute_error(ground_truth, calculated)
        rmse = np.sqrt(mean_squared_error(ground_truth, calculated))
        
        eval_results[method] = {"pearson_corr": corr, "mae": mae, "rmse": rmse}
        
        if FAST_VALIDATION['skip_plots']: continue

        # --- PLOTTING ---
        plot_dir = os.path.join(RESULTS_DIR, group_name, method)
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(18, 7))
        plt.plot(ground_truth, label='Ground-Truth Price', color='black', lw=2.5, alpha=0.6)
        plt.plot(calculated, label=f'Calculated Price ({method.upper()})', color='purple', alpha=0.8, linestyle='-')
        plt.title(f'Direct Price Tracking: {method.upper()} - Sim {sim_id}\nCorr: {corr:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}')
        plt.xlabel("Window Index")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"sim_{sim_id}_direct_tracking.png"), dpi=150)
        plt.close()
        
    return eval_results

def plot_group_summary_tracking(group_name: str, all_run_results: list):
    """Plots concatenated ground truth vs. calculated prices for a group, sorted by ground-truth price."""
    if not all_run_results or FAST_VALIDATION['skip_plots']: return

    for method in METHODS_TO_RUN:
        ground_truth_all = np.concatenate([r['ground_truth'] for r in all_run_results if r and method in r['results']])
        calculated_all = np.concatenate([r['results'][method] for r in all_run_results if r and method in r['results']])
        if len(ground_truth_all) == 0: continue

        sort_indices = np.argsort(ground_truth_all)
        ground_truth_sorted = ground_truth_all[sort_indices]
        calculated_sorted = calculated_all[sort_indices]

        plt.figure(figsize=(20, 8))
        plt.plot(ground_truth_sorted, label='Ground-Truth Price', color='black', lw=2, alpha=0.7)
        plt.plot(calculated_sorted, label=f'Calculated Price ({method.upper()})', color='purple', alpha=0.7, linestyle='--')
        plt.plot([ground_truth_sorted.min(), ground_truth_sorted.max()], [ground_truth_sorted.min(), ground_truth_sorted.max()], 'g--', alpha=0.5, label='Perfect Prediction')
        plt.title(f'Group Summary: {group_name} - {method.upper()} (Sorted by Ground Truth)')
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
    parser = argparse.ArgumentParser(description="Run Experiment 3: A robust, valid billing model.")
    parser.add_argument("--group", type=str, help="Run for a single physics group (e.g., 'group_1').")
    args = parser.parse_args()

    groups_to_process = PHYSICS_GROUPS
    if args.group:
        if args.group in PHYSICS_GROUPS:
            groups_to_process = {args.group: PHYSICS_GROUPS[args.group]}
        else:
            print(f"Error: Group '{args.group}' not found. Exiting.")
            return

    print("🚗 EXPERIMENT 3: Robust Billing Validation with Augmented Pure Samples")
    print("=" * 80)
    print(f"METHODOLOGY: Train on augmented 'pure' samples, evaluate with direct correlation.")
    print(f"AUGMENTATION: {AUGMENTATION_CONFIG['enabled']} ({AUGMENTATION_CONFIG['samples_per_class']} samples/class)")
    print(f"METHODS: {METHODS_TO_RUN}")
    print(f"DIM REDUCTION: {DIMENSIONALITY_REDUCTION['method']}")
    print(f"PRICING: {PRICING_STRATEGY['method']}")
    print("=" * 80)

    all_results_list = []

    for group_name, params in groups_to_process.items():
        print(f"\n--- Processing {group_name} ---")
        
        try:
            autoencoder_model, autoencoder_scaler = load_trained_autoencoder(group_name)
            print(f"✅ Autoencoder for {group_name} loaded successfully.")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}. Please train the autoencoder for {group_name} first.")
            continue
        
        # === Step 1: Collect all training data to find pure samples ===
        group_csv_files = sorted([f for sim_num in params['seq_range'] for f in glob.glob(os.path.join(PROCESSED_DATA_DIR, f"simulation_{sim_num}_*.csv"))])
        if not group_csv_files:
            print(f"  No CSV files found for {group_name}. Skipping group.")
            continue

        n_train = int(len(group_csv_files) * TRAIN_VALIDATION_SPLIT)
        train_files, validation_files = group_csv_files[:n_train], group_csv_files[n_train:]
        if FAST_VALIDATION['enabled']:
            train_files = train_files[:FAST_VALIDATION['max_train_files']]
            validation_files = validation_files[:FAST_VALIDATION['max_val_files']]
        if not train_files or not validation_files:
            print(f"  Insufficient files for train/val split. Train: {len(train_files)}, Val: {len(validation_files)}. Skipping group.")
            continue
        
        print(f"  Split: {len(train_files)} training files, {len(validation_files)} validation files")

        all_training_windows, all_training_contexts = [], []
        for train_file in train_files:
            error_windows, window_contexts = get_or_generate_error_windows(train_file, autoencoder_model, autoencoder_scaler, group_name)
            if error_windows is not None and window_contexts is not None:
                all_training_windows.append(error_windows)
                all_training_contexts.extend(window_contexts)
        if not all_training_windows:
            print(f"  Could not generate any training windows for {group_name}. Skipping group.")
            continue
        
        all_training_windows = np.vstack(all_training_windows)

        # === Step 2: Create "Pure" Samples and then Augment them ===
        # This is the key step for valid training
        context_blocks = defaultdict(list)
        for window, context in zip(all_training_windows, all_training_contexts):
            context_blocks[context].append(window)

        pure_samples, pure_contexts = create_context_representative_samples(dict(context_blocks), FIXED_WINDOW_SIZE)
        if not pure_samples:
            print(f"  Could not create any 'pure' samples from the training data. Skipping group.")
            continue

        augmented_samples, augmented_contexts = augment_and_balance_data(pure_samples, pure_contexts, AUGMENTATION_CONFIG)

        # === Step 3: Feature Extraction and Model Training on Augmented Data ===
        print(f"\n  Training clustering models on {len(augmented_samples)} balanced, augmented samples...")
        
        features = extract_enhanced_features_from_reconstruction_errors(np.array(augmented_samples))
        scaler = StandardScaler().fit(features)
        scaled_features = scaler.transform(features)
        reduced_features, reducer = apply_dimensionality_reduction(scaled_features, DIMENSIONALITY_REDUCTION)

        # Determine initial centroids from the 'pure' samples in the new feature space
        pure_features = extract_enhanced_features_from_reconstruction_errors(np.array(pure_samples))
        scaled_pure_features = scaler.transform(pure_features)
        reduced_pure_features = transform_new_features(scaled_pure_features, reducer)
        
        # Group pure features by generic context to select one initial centroid per class
        initial_centroid_features = []
        centroid_contexts = []
        grouped_pure_features = defaultdict(list)
        for i, context in enumerate(pure_contexts):
            grouped_pure_features[get_generic_context(context)].append(reduced_pure_features[i])
        
        for context, feature_list in grouped_pure_features.items():
            initial_centroid_features.append(np.mean(feature_list, axis=0)) # Use mean of pure samples
            centroid_contexts.append(context)
        
        initial_centroid_features = np.array(initial_centroid_features)
        n_clusters = len(initial_centroid_features)
        centroid_prices = np.array([get_price_for_context(c) for c in centroid_contexts])

        print(f"  Identified {n_clusters} unique generic contexts for clustering.")
        print(f"  Centroid contexts: {centroid_contexts}")
        print(f"  Centroid prices: {centroid_prices}")

        # Train all specified models
        trained_models = {}
        for method in METHODS_TO_RUN:
            if method == 'baseline': continue
            if method == 'kmedoids' and not KMEDOIDS_AVAILABLE: continue
            
            print(f"\n--- Training {method.upper()} model ---")
            if method == 'kmedoids':
                model = KMedoids(n_clusters=n_clusters, init='k-medoids++', random_state=42).fit(reduced_features)
            else: # kmeans
                model = KMeans(n_clusters=n_clusters, init=initial_centroid_features, n_init=1, random_state=42).fit(reduced_features)
            
            final_centroids, final_labels = model.cluster_centers_, model.labels_
            final_centroids, final_labels = handle_single_element_clusters(
                np.array(final_centroids), np.array(final_labels), reduced_features, is_kmedoids=(method=='kmedoids')
            )
            
            # Update centroid prices to match final (post-merge) clusters
            updated_prices = []
            for i in range(len(final_centroids)):
                cluster_contexts = np.array(augmented_contexts)[final_labels == i]
                if len(cluster_contexts) > 0:
                    # Assign price based on the most common (and most expensive in case of tie) context in the final cluster
                    updated_prices.append(get_price_for_context(assign_context_to_fixed_window(cluster_contexts)))
                else: # Should not happen
                    updated_prices.append(3.0)

            trained_models[method] = {
                'final_centroids': final_centroids,
                'centroid_prices': np.array(updated_prices)
            }
            print(f"  ✅ {method.upper()} model trained successfully with {len(final_centroids)} final clusters.")

        # === Visualization: PCA scatter plot coloured by context ===
        if reduced_features.shape[1] >= 2:
            try:
                unique_ctx = sorted(set(augmented_contexts))
                palette = sns.color_palette("hls", len(unique_ctx))
                ctx_to_color = {ctx: palette[i] for i, ctx in enumerate(unique_ctx)}

                plt.figure(figsize=(10, 8))
                for ctx in unique_ctx:
                    idxs = [idx for idx, c in enumerate(augmented_contexts) if c == ctx]
                    plt.scatter(
                        reduced_features[idxs, 0],
                        reduced_features[idxs, 1],
                        label=ctx,
                        s=12,
                        color=ctx_to_color[ctx],
                        alpha=0.7,
                    )

                # Overlay centroids from KMEANS if available (else KMEDOIDS)
                centroid_source = None
                if "kmeans" in trained_models:
                    centroid_source = trained_models["kmeans"]
                elif "kmedoids" in trained_models:
                    centroid_source = trained_models["kmedoids"]

                if centroid_source is not None:
                    centroids_2d = centroid_source["final_centroids"][:, :2]
                    plt.scatter(
                        centroids_2d[:, 0],
                        centroids_2d[:, 1],
                        marker="X",
                        s=140,
                        c="black",
                        edgecolors="white",
                        linewidths=0.7,
                        label="Cluster Centroids",
                    )

                plt.title(f"PCA Scatter (first 2 components) – {group_name}")
                plt.xlabel("PC 1")
                plt.ylabel("PC 2")
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
                plt.tight_layout()

                plot_dir = os.path.join(RESULTS_DIR, group_name)
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, "pca_clusters_by_context.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"  PCA cluster plot saved to: {plot_path}")
            except Exception as e:
                print(f"  Warning: Could not generate PCA cluster plot: {e}")

        # === Step 4: Evaluation on Validation Data ===
        print("\n  Evaluating performance on VALIDATION data...")
        validation_run_results = []
        for csv_file in validation_files:
            sim_id_match = re.search(r'simulation_(\d+)_', os.path.basename(csv_file))
            if not sim_id_match: continue
            sim_id = int(sim_id_match.group(1))
            
            print(f"    Evaluating validation simulation {sim_id}...")
            fixed_windows, gt_contexts = get_or_generate_error_windows(csv_file, autoencoder_model, autoencoder_scaler, group_name)
            if fixed_windows is None or gt_contexts is None:
                print(f"    Skipping sim {sim_id} due to processing error.")
                continue

            if FAST_VALIDATION['enabled']:
                max_win = FAST_VALIDATION['max_windows_per_file']
                fixed_windows = fixed_windows[:max_win]
                gt_contexts = gt_contexts[:max_win]

            calculated_prices = {}
            if 'baseline' in METHODS_TO_RUN:
                calculated_prices['baseline'] = calculate_baseline_prices(fixed_windows)
            for method, artifacts in trained_models.items():
                calculated_prices[method] = calculate_cluster_prices(
                    fixed_windows, artifacts['centroid_prices'], artifacts['final_centroids'], scaler, reducer
                )
            
            ground_truth_prices = np.array([get_price_for_context(ctx) for ctx in gt_contexts])
            run_evals = perform_evaluation_and_plot(calculated_prices, ground_truth_prices, f"{group_name}_VAL", sim_id)
            
            if calculated_prices and ground_truth_prices.size > 0:
                validation_run_results.append({'results': calculated_prices, 'ground_truth': ground_truth_prices})

            for method, metrics in run_evals.items():
                all_results_list.append({
                    "group": group_name, "mass": params['mass'], "friction": params['friction'],
                    "sim_id": sim_id, "method": method, "data_type": "VALIDATION", **metrics
                })
        
        plot_group_summary_tracking(f"{group_name}_VAL", validation_run_results)

    # === Step 5: Final Summary ===
    if not all_results_list:
        print("\nNo results were generated. Exiting.")
        return
        
    summary_df = pd.DataFrame(all_results_list)
    summary_path = os.path.join(RESULTS_DIR, "full_evaluation_results_experiment3.csv")
    summary_df.to_csv(summary_path, index=False)

    agg_metrics = summary_df.groupby(['mass', 'friction', 'method', 'data_type']).agg(
        corr_mean=('pearson_corr', 'mean'), corr_std=('pearson_corr', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std')
    ).reset_index()
    
    agg_path = os.path.join(RESULTS_DIR, "summary_evaluation_by_group_experiment3.csv")
    agg_metrics.to_csv(agg_path, index=False)
    
    print("\n" + "="*80)
    print("🎯 EXPERIMENT 3 FINAL RESULTS (Direct Evaluation)")
    print("="*80)
    print(agg_metrics.to_string())
    print(f"\nFull evaluation results saved to {summary_path}")
    print(f"Summary table saved to {agg_path}")
    print("="*80)

if __name__ == "__main__":
    main()
