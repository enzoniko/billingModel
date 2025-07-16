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
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy.fft import fft

# Import autoencoder components
from recurrent_autoencoder_anomaly_detection import VehicleAutoencoder, SENSORS_FOR_AUTOENCODER, WINDOW_SIZE

# --- Configuration ---

# Paths
PROCESSED_DATA_DIR = "processed_data"
RESULTS_DIR = "results_experiment2_old"
MODELS_DIR = "autoencoder_models"
AUTOENCODER_RESULTS_DIR = "autoencoder_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment Parameters
FIXED_WINDOW_SIZE = 30

# Enhanced Dimensionality Reduction Configuration
DIMENSIONALITY_REDUCTION = {
    'method': 'intelligent_pca',  # Options: 'intelligent_pca', 'umap', 'incremental_pca', 'feature_selection', 'hybrid'
    'target_variance': 0.95,      # For intelligent PCA: retain 95% of variance
    'max_components': 100,        # Maximum components regardless of variance
    'min_components': 10,         # Minimum components to ensure sufficient representation
    'umap_n_components': 50,      # For UMAP reduction
    'umap_n_neighbors': 15,       # UMAP neighborhood size
    'feature_selection_k': 50,    # Top-k features for univariate selection
    'hybrid_first_stage': 200,    # First stage reduction (PCA/feature selection)
    'hybrid_second_stage': 50     # Final stage reduction
}

# Enhanced Pricing Strategy Configuration
PRICING_STRATEGY = {
    'method': 'threshold_assignment',  # Options: 'weighted_average', 'hard_assignment', 'threshold_assignment', 'exponential_weighting'
    'threshold_distance': 0.8,        # For threshold_assignment: relative closeness threshold (0.8 = 80% closer)
    'exponential_power': 3.0,         # For exponential_weighting: power for exponential decay
    'min_weight_threshold': 0.01      # Minimum weight threshold to ignore very distant centroids
}

# Train/Validation Split Configuration
TRAIN_VALIDATION_SPLIT = 0.7  # 70% for training, 30% for validation

# Method Selection Configuration - kmeans + baseline
METHODS_TO_RUN = ['kmeans', 'baseline', 'hybrid']  # kmeans + baseline method

# Hybrid combination parameters
HYBRID_PARAMS = {
    'alpha': 0.3  # incentive factor (0 = pure masking, >0 boosts peaks)
}

# Enhanced Feature Parameters
WAVELET_TYPE = 'morl'  # Morlet wavelet
WAVELET_SCALES = np.arange(1, 32)  # Scales for wavelet transform

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

# Use reconstruction error signals as the input features
RECONSTRUCTION_ERROR_COLUMNS = [f"{sensor}_reconstruction_error" for sensor in SENSORS_FOR_AUTOENCODER]

print(f"Using {len(RECONSTRUCTION_ERROR_COLUMNS)} reconstruction error signals as features:")
for i, col in enumerate(RECONSTRUCTION_ERROR_COLUMNS):
    print(f"  {i+1}. {col}")

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
    if 'crash' in context: return 50.0  # Highest priority - most severe event
    if 'pothole' in context: return 15.0
    if 'speedbump' in context: return 8.0
    if 'elevated_crosswalk' in context: return 5.0
    if 'cut' in context: return 3.0
    return 3.0 # Default for any other unknown context


# --- Autoencoder Integration Functions ---

def load_trained_autoencoder(group_name: str) -> Tuple[VehicleAutoencoder, MinMaxScaler]:
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
    results_path = os.path.join(AUTOENCODER_RESULTS_DIR, f"{group_name}_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results file found for {group_name} at {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    scaler = results['scaler']
    return model, scaler

def process_data_for_autoencoder(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as autoencoder training."""
    print("Processing data: Applying gravity correction and detecting crashes...")
    
    # Standardize column names to uppercase
    df.columns = [col.upper() for col in df.columns]
    
    # Correct for gravity
    if 'IMU_ACC_Z' in df.columns:
        df['IMU_ACC_Z_DYNAMIC'] = df['IMU_ACC_Z'] - 9.81
    else:
        df['IMU_ACC_Z_DYNAMIC'] = 0
        print("Warning: 'IMU_ACC_Z' not found. Setting IMU_ACC_Z_DYNAMIC to 0.")
    
    # Calculate horizontal acceleration for crash detection
    if 'IMU_ACC_X' in df.columns and 'IMU_ACC_Y' in df.columns:
        df['ACC_HORIZONTAL'] = np.sqrt(df['IMU_ACC_X']**2 + df['IMU_ACC_Y']**2)
        
        # Detect crashes (threshold = 10.0 m/s²)
        crash_threshold = 10.0
        crash_window_seconds = 1.0
        fs = 10  # 10Hz sampling
        window_size = int(crash_window_seconds * fs)
        
        crash_indices = df.index[df['ACC_HORIZONTAL'] > crash_threshold].tolist()
        
        if crash_indices:
            print(f"Found {len(crash_indices)} potential crash points. Applying context window...")
            
            # Initialize context column if it doesn't exist
            if 'CONTEXT' not in df.columns:
                df['CONTEXT'] = df.get('context', 'unknown')
            
            # Convert to object type to allow string assignment
            df['CONTEXT'] = df['CONTEXT'].astype(object)
            
            # Mark crash contexts
            for idx in crash_indices:
                start = max(0, idx - window_size)
                end = min(len(df), idx + window_size + 1)
                df.loc[start:end, 'CONTEXT'] = 'crash'
        else:
            print("No crash events detected.")
            df['CONTEXT'] = df.get('context', 'unknown')
    else:
        print("Warning: Cannot detect crashes - missing IMU_ACC_X or IMU_ACC_Y")
        df['CONTEXT'] = df.get('context', 'unknown')
    
    return df

def generate_reconstruction_error_signals(df: pd.DataFrame, model: VehicleAutoencoder, 
                                        scaler: MinMaxScaler) -> pd.DataFrame:
    """Generate reconstruction error signals for the entire simulation."""
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
    
    # Create new dataframe with reconstruction error signals
    df_with_errors = df.copy()
    for j, sensor in enumerate(SENSORS_FOR_AUTOENCODER):
        df_with_errors[f"{sensor}_reconstruction_error"] = reconstruction_errors[:, j]
    
    return df_with_errors 

# --- Enhanced Feature Extraction Functions ---

def extract_time_domain_features(signal: np.ndarray) -> np.ndarray:
    """Extract time-domain statistical features from a signal."""
    features = []
    
    # Basic statistics
    features.extend([
        np.min(signal),
        np.max(signal),
        np.mean(signal),
        np.std(signal),
        np.var(signal)
    ])
    
    # Higher order moments
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val > 1e-8:
        # Skewness
        features.append(np.mean(((signal - mean_val) / std_val) ** 3))
        # Kurtosis
        features.append(np.mean(((signal - mean_val) / std_val) ** 4))
    else:
        features.extend([0.0, 0.0])
    
    # Energy
    features.append(np.sum(signal ** 2))
    
    # Root Mean Square
    features.append(np.sqrt(np.mean(signal ** 2)))
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    features.append(zero_crossings / len(signal))
    
    # Peak to peak
    features.append(np.max(signal) - np.min(signal))
    
    return np.array(features)

def extract_enhanced_fft_features(signal: np.ndarray, fs: float = 10.0) -> np.ndarray:
    """Extract enhanced FFT features from a signal."""
    # Handle edge cases for very small signals
    if len(signal) < 2:
        print(f"  DEBUG: Very small signal length {len(signal)}, returning zero features")
        return np.zeros(7)  # Return 7 zero features to match expected output
    
    # Compute FFT
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Use only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_vals[:len(fft_vals)//2]
    
    # Handle case where positive frequencies array is empty
    if len(positive_freqs) == 0 or len(positive_fft) == 0:
        print(f"  DEBUG: Empty positive frequencies for signal length {len(signal)}, returning zero features")
        return np.zeros(7)
    
    features = []
    
    # Spectral centroid
    if np.sum(positive_fft) > 0:
        spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
    else:
        spectral_centroid = 0.0
    features.append(spectral_centroid)
    
    # Spectral rolloff (95%)
    cumulative_sum = np.cumsum(positive_fft)
    if len(cumulative_sum) > 0:
        total_energy = cumulative_sum[-1]
        if total_energy > 0:
            rolloff_idx = np.where(cumulative_sum >= 0.95 * total_energy)[0]
            spectral_rolloff = positive_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else positive_freqs[-1]
        else:
            spectral_rolloff = 0.0
    else:
        spectral_rolloff = 0.0
    features.append(spectral_rolloff)
    
    # Spectral spread
    if np.sum(positive_fft) > 0:
        spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * positive_fft) / np.sum(positive_fft))
    else:
        spectral_spread = 0.0
    features.append(spectral_spread)
    
    # Spectral flux
    features.append(np.sum(np.diff(positive_fft) ** 2))
    
    # Fundamental frequency (dominant frequency)
    if len(positive_fft) > 0:
        dominant_freq_idx = np.argmax(positive_fft)
        features.append(positive_freqs[dominant_freq_idx])
    else:
        features.append(0.0)
    
    # Spectral energy
    features.append(np.sum(positive_fft ** 2))
    
    # Spectral entropy
    if np.sum(positive_fft) > 0:
        normalized_fft = positive_fft / np.sum(positive_fft)
        spectral_entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-12))
    else:
        spectral_entropy = 0.0
    features.append(spectral_entropy)
    
    return np.array(features)

def extract_morlet_wavelet_features(signal: np.ndarray, scales: np.ndarray = WAVELET_SCALES, 
                                  sampling_period: float = 0.1) -> np.ndarray:
    """Extract Morlet wavelet features from a signal."""
    # Handle edge cases for very small signals
    if len(signal) < 2:
        print(f"  DEBUG: Very small signal length {len(signal)} for wavelet, returning zero features")
        expected_length = len(WAVELET_SCALES) + 8
        return np.zeros(expected_length)
        
    try:
        # Continuous Wavelet Transform with Morlet wavelet
        cwt_result = pywt.cwt(signal, scales, WAVELET_TYPE, sampling_period=sampling_period)
        coefficients = cwt_result[0]
        frequencies = cwt_result[1]
        
        features = []
        
        # Energy at each scale
        scale_energies = np.sum(np.abs(coefficients) ** 2, axis=1)
        features.extend(scale_energies.tolist())
        
        # Total energy
        features.append(np.sum(scale_energies))
        
        # Dominant scale (scale with maximum energy)
        dominant_scale_idx = np.argmax(scale_energies)
        features.append(scales[dominant_scale_idx])
        
        # Frequency content statistics
        freq_weighted_energy = np.sum(frequencies.reshape(-1, 1) * np.abs(coefficients) ** 2, axis=1)
        if np.sum(scale_energies) > 0:
            mean_frequency = np.sum(freq_weighted_energy) / np.sum(scale_energies)
        else:
            mean_frequency = 0.0
        features.append(mean_frequency)
        
        # Wavelet entropy
        if np.sum(scale_energies) > 0:
            normalized_energies = scale_energies / np.sum(scale_energies)
            wavelet_entropy = -np.sum(normalized_energies * np.log2(normalized_energies + 1e-12))
        else:
            wavelet_entropy = 0.0
        features.append(wavelet_entropy)
        
        # Energy distribution across scales (percentiles)
        if len(scale_energies) > 0:
            features.extend([
                np.percentile(scale_energies, 25),
                np.percentile(scale_energies, 50),
                np.percentile(scale_energies, 75)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Number of significant scales (above mean energy)
        mean_energy = np.mean(scale_energies)
        significant_scales = np.sum(scale_energies > mean_energy)
        features.append(significant_scales)
        
        return np.array(features)
        
    except Exception as e:
        print(f"Warning: Wavelet feature extraction failed: {e}")
        # Return zeros if wavelet transform fails
        expected_length = len(WAVELET_SCALES) + 8  # scales + additional features
        return np.zeros(expected_length)

def extract_enhanced_features_from_reconstruction_errors(windowed_data: np.ndarray) -> np.ndarray:
    """
    Extract enhanced features from windowed reconstruction error signals.
    
    Args:
        windowed_data: Shape (num_windows, window_size, num_error_signals)
    
    Returns:
        Feature matrix: Shape (num_windows, total_features)
    """
    if windowed_data.ndim == 2:  # Single window
        windowed_data = windowed_data[np.newaxis, :, :]
    
    num_windows, window_size, num_signals = windowed_data.shape
    
    # Calculate expected feature dimensions
    time_features_per_signal = 11  # From extract_time_domain_features
    fft_features_per_signal = 7    # From extract_enhanced_fft_features  
    wavelet_features_per_signal = len(WAVELET_SCALES) + 8  # From extract_morlet_wavelet_features
    
    total_features_per_signal = time_features_per_signal + fft_features_per_signal + wavelet_features_per_signal
    total_features = num_signals * total_features_per_signal
    
    print(f"Extracting enhanced features: {total_features_per_signal} per signal × {num_signals} signals = {total_features} total features")
    
    all_features = np.zeros((num_windows, total_features))
    
    for window_idx in range(num_windows):
        feature_idx = 0
        
        for signal_idx in range(num_signals):
            signal = windowed_data[window_idx, :, signal_idx]
            
            # Handle NaN values
            if np.all(np.isnan(signal)) or len(signal) == 0:
                # Fill with zeros if signal is all NaN
                signal_features = np.zeros(total_features_per_signal)
            else:
                # Replace NaN with interpolation or zero
                if np.any(np.isnan(signal)):
                    signal = np.nan_to_num(signal, nan=0.0)
                
                # Extract features
                time_features = extract_time_domain_features(signal)
                fft_features = extract_enhanced_fft_features(signal)
                wavelet_features = extract_morlet_wavelet_features(signal)
                
                # Combine features
                signal_features = np.concatenate([time_features, fft_features, wavelet_features])
            
            # Store features
            end_idx = feature_idx + len(signal_features)
            all_features[window_idx, feature_idx:end_idx] = signal_features
            feature_idx = end_idx
    
    # Clean up any remaining NaN or inf values
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return all_features

# --- Enhanced Dimensionality Reduction Functions ---

def intelligent_pca_reduction(features: np.ndarray, target_variance: float = 0.95, 
                            max_components: int = 100, min_components: int = 10) -> Tuple[PCA, int]:
    """
    Intelligently determines the optimal number of PCA components based on variance retention.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        target_variance: Target variance to retain (default: 95%)
        max_components: Maximum number of components
        min_components: Minimum number of components
        
    Returns:
        Tuple of (fitted_pca_object, n_components_selected)
    """
    n_samples, n_features = features.shape
    max_possible = min(int(n_samples - 1), int(n_features), max_components)
    
    if max_possible < 1:
        # Not enough data to reduce, return a dummy object that does nothing
        print("  Warning: Not enough samples to perform PCA. Returning dummy reducer.")
        class DummyPCAReducer:
            def fit(self, X: np.ndarray) -> 'DummyPCAReducer': return self
            def transform(self, X: np.ndarray) -> np.ndarray: return X if isinstance(X, np.ndarray) else np.array(X)
        return DummyPCAReducer(), features.shape[1] # type: ignore

    # Fit PCA with maximum possible components
    pca_full = PCA(n_components=max_possible)
    pca_full.fit(features)
    
    # Calculate cumulative explained variance
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find number of components needed for target variance - FIX TYPE ISSUES
    component_idx = np.argmax(cumsum_variance >= target_variance)
    n_components = int(component_idx) + 1  # Convert to Python int explicitly
    
    # If variance target is not met, argmax returns 0. Check if we should use all components.
    if cumsum_variance[-1] < target_variance:
        n_components = max_possible

    n_components = max(min_components, min(n_components, max_components))
    n_components = min(n_components, max_possible) # Ensure we don't exceed what's possible
    
    print(f"  Intelligent PCA: {n_components} components retain {cumsum_variance[n_components-1]:.3f} variance")
    print(f"  Reduction: {n_features} → {n_components} features ({100*(1-n_components/n_features):.1f}% reduction)")
    
    # Create final PCA with selected components
    pca_final = PCA(n_components=n_components)
    pca_final.fit(features)
    
    return pca_final, n_components

def umap_reduction(features: np.ndarray, n_components: int = 50, n_neighbors: int = 15):
    """
    Apply UMAP dimensionality reduction (non-linear).
    
    Args:
        features: Feature matrix (n_samples, n_features)  
        n_components: Target dimensionality
        n_neighbors: Number of neighbors for UMAP
        
    Returns:
        Fitted UMAP object
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    print(f"  UMAP: {features.shape[1]} → {n_components} features")
    
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    umap_reducer.fit(features)
    
    return umap_reducer

def feature_selection_reduction(features: np.ndarray, labels: np.ndarray, 
                              k: int = 50, method: str = 'mutual_info') -> SelectKBest:
    """
    Apply univariate feature selection to select top-k features.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Context labels for supervised selection
        k: Number of features to select
        method: 'mutual_info' or 'f_classif'
        
    Returns:
        Fitted SelectKBest object
    """
    print(f"  Feature Selection ({method}): {features.shape[1]} → {k} features")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
    
    selector.fit(features, labels)
    
    # Show top features if possible
    feature_scores = selector.scores_
    top_indices = selector.get_support(indices=True)
    print(f"  Selected features have scores: {feature_scores[top_indices][:5]}... (showing top 5)")
    
    return selector

def hybrid_reduction(features: np.ndarray, labels: np.ndarray,
                   first_stage: int = 200, second_stage: int = 50) -> Tuple[SelectKBest, PCA]:
    """
    Apply two-stage hybrid dimensionality reduction:
    1. Feature selection to remove irrelevant features
    2. PCA for final dimensionality reduction
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Context labels for supervised feature selection
        first_stage: Number of features after first stage
        second_stage: Final number of components
        
    Returns:
        Tuple of (feature_selector, pca_reducer)
    """
    print(f"  Hybrid Reduction: {features.shape[1]} → {first_stage} → {second_stage} features")
    
    # Stage 1: Feature selection
    selector = feature_selection_reduction(features, labels, k=first_stage, method='mutual_info')
    features_selected = selector.transform(features)
    
    # Stage 2: PCA on selected features
    pca_reducer = PCA(n_components=second_stage)
    pca_reducer.fit(features_selected)
    
    variance_retained = np.sum(pca_reducer.explained_variance_ratio_)
    print(f"  Final variance retained: {variance_retained:.3f}")
    
    return selector, pca_reducer

def apply_dimensionality_reduction(features: np.ndarray, labels: Optional[list] = None, 
                                 config: Optional[dict] = None) -> Tuple[np.ndarray, object]:
    """
    Apply the configured dimensionality reduction method.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Context labels (for supervised methods)
        config: Dimensionality reduction configuration
        
    Returns:
        Tuple of (reduced_features, fitted_reducer_object)
    """
    if config is None:
        config = DIMENSIONALITY_REDUCTION
    
    method = config['method']
    print(f"🔧 Applying dimensionality reduction: {method}")
    print(f"📊 Original feature space: {features.shape}")
    
    if method == 'intelligent_pca':
        reducer, n_components = intelligent_pca_reduction(
            features, 
            target_variance=config['target_variance'],
            max_components=config['max_components'],
            min_components=config['min_components']
        )
        reduced_features = reducer.transform(features)
        
    elif method == 'umap':
        reducer = umap_reduction(
            features,
            n_components=config['umap_n_components'],
            n_neighbors=config['umap_n_neighbors']
        )
        reduced_features = reducer.transform(features)
        
    elif method == 'incremental_pca':
        # Useful for large datasets that don't fit in memory
        n_components = min(config['max_components'], features.shape[1])
        reducer = IncrementalPCA(n_components=n_components)
        reduced_features = reducer.fit_transform(features)
        print(f"  Incremental PCA: {features.shape[1]} → {n_components} features")
        
    elif method == 'feature_selection':
        if labels is None:
            raise ValueError("Feature selection requires context labels")
        # Convert labels to numpy array for sklearn
        labels_array = np.array(labels) if isinstance(labels, list) else labels
        reducer = feature_selection_reduction(
            features, labels_array, 
            k=config['feature_selection_k']
        )
        reduced_features = reducer.transform(features)
        
    elif method == 'hybrid':
        if labels is None:
            raise ValueError("Hybrid method requires context labels")
        # Convert labels to numpy array for sklearn  
        labels_array = np.array(labels) if isinstance(labels, list) else labels
        selector, pca_reducer = hybrid_reduction(
            features, labels_array,
            first_stage=config['hybrid_first_stage'],
            second_stage=config['hybrid_second_stage']
        )
        # For hybrid, we need to store both reducers
        features_selected = selector.transform(features)
        reduced_features = pca_reducer.transform(features_selected)
        reducer = (selector, pca_reducer)  # Return tuple for hybrid
        
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Ensure reduced_features is always a numpy array
    if not isinstance(reduced_features, np.ndarray):
        reduced_features = np.array(reduced_features)
    
    print(f"✅ Final reduced feature space: {reduced_features.shape}")
    print(f"📉 Compression ratio: {features.shape[1]/reduced_features.shape[1]:.1f}:1")
    
    return reduced_features, reducer

def transform_new_features(features: np.ndarray, reducer: Union[object, Tuple], method: str) -> np.ndarray:
    """
    Transform new features using a fitted dimensionality reducer.
    
    Args:
        features: New feature matrix to transform
        reducer: Fitted reducer object or tuple for hybrid method
        method: Reduction method used
        
    Returns:
        Transformed features
    """
    if method == 'hybrid':
        # For hybrid method, reducer is a tuple of (selector, pca_reducer)
        if isinstance(reducer, tuple) and len(reducer) == 2:
            selector, pca_reducer = reducer
            features_selected = selector.transform(features)
            transformed = pca_reducer.transform(features_selected)
        else:
            raise ValueError("Hybrid method requires tuple of (selector, pca_reducer)")
    else:
        # For other methods, reducer is a single object with transform method
        if hasattr(reducer, 'transform'):
            transformed = reducer.transform(features)
        else:
            raise ValueError(f"Reducer for method '{method}' does not have transform method")
    
    # Ensure output is always numpy array
    if not isinstance(transformed, np.ndarray):
        transformed = np.array(transformed)
    
    return transformed

# --- Core Functions (Adapted from experiment1.py) ---

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

# DTW functions removed - using only kmeans for faster execution

def calculate_prices_with_strategy(distances: np.ndarray, prices: np.ndarray, 
                                  strategy: str = 'threshold_assignment',
                                  threshold_distance: float = 0.5,
                                  exponential_power: float = 3.0,
                                  min_weight_threshold: float = 0.01) -> np.ndarray:
    """
    Calculates prices using different strategies to address soft prediction issues.
    
    Args:
        distances: Distance matrix (n_windows, n_centroids)
        prices: Centroid prices array (n_centroids,)
        strategy: Pricing strategy to use
        threshold_distance: Distance threshold for pure assignment
        exponential_power: Power for exponential weighting
        min_weight_threshold: Minimum weight to consider a centroid
        
    Returns:
        Calculated prices array (n_windows,)
    """
    if distances.shape[0] == 0:
        return np.array([])
    
    if strategy == 'hard_assignment':
        # Assign each window to its closest centroid (pure prices)
        closest_indices = np.argmin(distances, axis=1)
        return prices[closest_indices]
        
    elif strategy == 'threshold_assignment':
        # If RELATIVELY much closer to one centroid, use its price; otherwise weighted average
        calculated_prices = np.zeros(distances.shape[0])
        
        for i in range(distances.shape[0]):
            sorted_distances = np.sort(distances[i])
            min_distance = sorted_distances[0]
            second_min_distance = sorted_distances[1] if len(sorted_distances) > 1 else min_distance
            min_idx = np.argmin(distances[i])
            
            # Relative threshold: if closest is X% closer than second closest
            if second_min_distance > 0:
                relative_closeness = 1.0 - (min_distance / second_min_distance)
                if relative_closeness >= threshold_distance:  # e.g., 90% closer
                    # Much closer to one centroid - use its exact price
                    calculated_prices[i] = prices[min_idx]
                else:
                    # Ambiguous - use weighted average
                    calculated_prices[i] = _weighted_average_pricing(distances[i], prices, min_weight_threshold)
            else:
                # Edge case: all distances are zero - use closest
                calculated_prices[i] = prices[min_idx]
        
        return calculated_prices
        
    elif strategy == 'exponential_weighting':
        # Use exponential decay to make closer centroids dominate much more
        calculated_prices = np.zeros(distances.shape[0])
        
        for i in range(distances.shape[0]):
            # Apply exponential decay: weight = exp(-power * distance)
            weights = np.exp(-exponential_power * distances[i])
            
            # Filter out very small weights
            valid_mask = weights >= min_weight_threshold
            if np.sum(valid_mask) == 0:
                # Fallback to closest centroid
                calculated_prices[i] = prices[np.argmin(distances[i])]
            else:
                valid_weights = weights[valid_mask]
                valid_prices = prices[valid_mask]
                normalized_weights = valid_weights / np.sum(valid_weights)
                calculated_prices[i] = np.sum(normalized_weights * valid_prices)
        
        return calculated_prices
        
    elif strategy == 'weighted_average':
        # Original weighted average method (kept for comparison)
        calculated_prices = np.zeros(distances.shape[0])
        for i in range(distances.shape[0]):
            calculated_prices[i] = _weighted_average_pricing(distances[i], prices, min_weight_threshold)
        return calculated_prices
        
    else:
        raise ValueError(f"Unknown pricing strategy: {strategy}")

def _weighted_average_pricing(distances_row: np.ndarray, prices: np.ndarray, 
                            min_weight_threshold: float = 0.01) -> float:
    """Helper function for weighted average pricing of a single window."""
    # Handle case where a distance is zero
    if np.any(distances_row == 0):
        zero_indices = np.where(distances_row == 0)[0]
        return float(np.mean(prices[zero_indices]))  # Convert to Python float
    
    # Calculate inverse distance weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_distances = 1.0 / distances_row
    
    # Filter out very small weights
    valid_mask = inv_distances >= min_weight_threshold
    if np.sum(valid_mask) == 0:
        # Fallback to closest centroid
        return float(prices[np.argmin(distances_row)])  # Convert to Python float
    
    valid_inv_distances = inv_distances[valid_mask]
    valid_prices = prices[valid_mask]
    normalized_weights = valid_inv_distances / np.sum(valid_inv_distances)
    
    return float(np.sum(normalized_weights * valid_prices))  # Convert to Python float

# Backward compatibility function
def normalize_and_compute_weighted_average(distances: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Legacy function for backward compatibility."""
    return calculate_prices_with_strategy(distances, prices, 'weighted_average')

# --- Step 2 (Helper): Fixed-Window Context Assignment ---

def assign_context_to_fixed_window(contexts: np.ndarray, strategy: str = 'majority_expensive') -> str:
    """
    Assigns a single context label to a fixed window that may contain multiple contexts.
    
    Args:
        contexts: Array of context strings for the window
        strategy: Strategy for context assignment
            - 'majority': Most frequent context
            - 'expensive': Most expensive context  
            - 'majority_expensive': Majority vote, tie-broken by expense
    
    Returns:
        Single context string for the window
    """
    if len(contexts) == 0:
        return 'unknown'
    
    # Convert to list if it's a pandas Series
    if hasattr(contexts, 'tolist'):
        contexts = contexts.tolist()
    
    if strategy == 'majority':
        # Simple majority voting
        context_counts = Counter(contexts)
        return context_counts.most_common(1)[0][0]
        
    elif strategy == 'expensive':
        # Most expensive context wins
        context_prices = [get_price_for_context(ctx) for ctx in contexts]
        max_price_idx = np.argmax(context_prices)
        return contexts[max_price_idx]
        
    elif strategy == 'majority_expensive':
        # Majority vote, with tie-breaking by most expensive
        context_counts = Counter(contexts)
        max_count = context_counts.most_common(1)[0][1]
        
        # Get all contexts with the maximum count
        tied_contexts = [ctx for ctx, count in context_counts.items() if count == max_count]
        
        if len(tied_contexts) == 1:
            return tied_contexts[0]
        else:
            # Break tie by most expensive context
            tied_prices = [get_price_for_context(ctx) for ctx in tied_contexts]
            most_expensive_idx = np.argmax(tied_prices)
            return tied_contexts[most_expensive_idx]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def extract_contiguous_context_blocks(df: pd.DataFrame, error_cols: list, 
                                    context_col: str = 'context') -> Dict[str, List[np.ndarray]]:
    """
    Extracts contiguous blocks of data for each context type.
    
    Args:
        df: DataFrame with reconstruction error signals and context information
        error_cols: List of reconstruction error column names
        context_col: Name of context column
    
    Returns:
        Dictionary mapping context names to lists of contiguous data blocks
    """
    context_blocks = defaultdict(list)
    
    if df.empty or context_col not in df.columns:
        return dict(context_blocks)
    
    current_context = df[context_col].iloc[0]
    start_idx = 0
    
    print(f"  DEBUG: Extracting contiguous blocks for contexts...")
    
    for i in range(1, len(df)):
        if df[context_col].iloc[i] != current_context:
            # End of current context block
            block_data = df.iloc[start_idx:i][error_cols].values
            if len(block_data) > 0:  # Only add non-empty blocks
                context_blocks[current_context].append(block_data)
                print(f"    Found {current_context} block: {len(block_data)} measurements")
            
            current_context = df[context_col].iloc[i]
            start_idx = i
    
    # Add the last block
    final_block_data = df.iloc[start_idx:][error_cols].values
    if len(final_block_data) > 0:
        context_blocks[current_context].append(final_block_data)
        print(f"    Found {current_context} block: {len(final_block_data)} measurements")
    
    # Summary
    for context, blocks in context_blocks.items():
        total_measurements = sum(len(block) for block in blocks)
        print(f"  Context '{context}': {len(blocks)} blocks, {total_measurements} total measurements")
    
    return dict(context_blocks)

def create_context_representative_samples(context_blocks: Dict[str, List[np.ndarray]], 
                                        target_length: int = FIXED_WINDOW_SIZE) -> Tuple[list, list]:
    """
    Creates representative samples for each context by concatenating blocks to reach target length.
    
    Args:
        context_blocks: Dictionary mapping context names to lists of contiguous data blocks
        target_length: Target length for representative samples
    
    Returns:
        Tuple of (representative_samples, context_labels)
    """
    representative_samples = []
    context_labels = []
    
    print(f"  Creating {target_length}-measurement representative samples for each context...")
    
    for context, blocks in context_blocks.items():
        if not blocks:
            continue
            
        # Sort blocks by length (descending) to prioritize longer blocks
        blocks_sorted = sorted(blocks, key=len, reverse=True)
        
        # Strategy 1: If any block is >= target_length, use it directly
        for block in blocks_sorted:
            if len(block) >= target_length:
                # Use the first target_length measurements
                sample = block[:target_length]
                representative_samples.append(sample)
                context_labels.append(context)
                print(f"    {context}: Used single block of {len(block)} measurements (trimmed to {target_length})")
                break
        else:
            # Strategy 2: Concatenate multiple blocks until reaching target_length
            concatenated_data = []
            used_blocks = 0
            
            # Keep adding blocks until we have enough measurements
            while len(concatenated_data) < target_length and used_blocks < len(blocks_sorted):
                block = blocks_sorted[used_blocks]
                concatenated_data.extend(block.tolist())
                used_blocks += 1
            
            if len(concatenated_data) >= target_length:
                # We have enough data
                sample = np.array(concatenated_data[:target_length])
                representative_samples.append(sample)
                context_labels.append(context)
                print(f"    {context}: Concatenated {used_blocks} blocks to get {len(sample)} measurements")
            else:
                # Still not enough data - repeat the concatenated data
                if len(concatenated_data) > 0:
                    repeats_needed = (target_length // len(concatenated_data)) + 1
                    repeated_data = (concatenated_data * repeats_needed)[:target_length]
                    sample = np.array(repeated_data)
                    representative_samples.append(sample)
                    context_labels.append(context)
                    print(f"    {context}: Repeated {len(concatenated_data)} measurements {repeats_needed}x to get {len(sample)} measurements")
                else:
                    print(f"    {context}: Skipped - no valid data")
    
    print(f"  Created {len(representative_samples)} representative samples")
    return representative_samples, context_labels

def create_fixed_windows_with_most_expensive_context(df: pd.DataFrame, error_cols: list, 
                                                   window_size: int = FIXED_WINDOW_SIZE,
                                                   context_col: str = 'context') -> Tuple[list, list]:
    """
    Creates fixed-size windows from reconstruction error signals and assigns the most expensive context label.
    This is used during evaluation to ensure billing-appropriate context assignment.
    
    Args:
        df: DataFrame with reconstruction error signals and context information
        error_cols: List of reconstruction error column names
        window_size: Size of fixed windows
        context_col: Name of context column
    
    Returns:
        Tuple of (windows_list, context_labels_list)
    """
    windows = []
    contexts = []
    
    if df.empty or context_col not in df.columns:
        return windows, contexts
    
    # Get total number of possible windows
    num_windows = len(df) // window_size
    print(f"  Creating {num_windows} fixed evaluation windows using 'most expensive' context strategy")
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        
        # Extract window data and contexts
        window_df = df.iloc[start_idx:end_idx]
        window_data = window_df[error_cols].values
        window_contexts = window_df[context_col].values
        
        # Assign most expensive context to window
        window_context = assign_context_to_fixed_window(window_contexts, 'expensive')
        
        windows.append(window_data)
        contexts.append(window_context)
    
    print(f"  Created {len(windows)} evaluation windows with context distribution: {dict(Counter(contexts))}")
    return windows, contexts

def create_fixed_windows_for_training(df: pd.DataFrame, error_cols: list,
                                      window_size: int = FIXED_WINDOW_SIZE,
                                      context_col: str = 'context') -> Tuple[List[np.ndarray], List[str]]:
    """
    Creates fixed-size windows from training data and assigns a context using the 'majority_expensive' strategy.
    
    Args:
        df: DataFrame with reconstruction error signals and context information.
        error_cols: List of reconstruction error column names.
        window_size: Size of fixed windows.
        context_col: Name of context column.
    
    Returns:
        Tuple of (windows_list, context_labels_list).
    """
    windows = []
    contexts = []
    
    if df.empty or context_col not in df.columns:
        return windows, contexts
    
    num_windows = len(df) // window_size
    print(f"  Extracting all {num_windows} possible windows from training data...")
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        
        window_df = df.iloc[start_idx:end_idx]
        window_data = window_df[error_cols].values
        window_contexts = window_df[context_col].values
        
        # Use majority_expensive strategy for a more robust label
        window_context = assign_context_to_fixed_window(window_contexts, 'majority_expensive')
        
        windows.append(window_data)
        contexts.append(window_context)
        
    print(f"  Extracted {len(windows)} windows with context distribution: {dict(Counter(contexts))}")
    return windows, contexts

def get_generic_context(specific_context: str) -> str:
    """Strips unique IDs from context strings (e.g., ramp_..._1 -> ramp_...)."""
    if not isinstance(specific_context, str): return "unknown"
    # Find the last underscore followed by a number
    match = re.search(r'_\d+$', specific_context)
    if match:
        return specific_context[:match.start()]
    return specific_context

def calculate_baseline_prices(fixed_windows: np.ndarray) -> np.ndarray:
    """
    Baseline pricing method using raw reconstruction error magnitudes.
    
    This method doesn't use context information - just raw deviations from normality.
    Uses RMS across signals and time as a measure of abnormality.
    
    Args:
        fixed_windows: Windows of reconstruction error signals (n_windows, window_size, n_signals)
    
    Returns:
        Array of prices based on error magnitudes (n_windows,)
    """
    if fixed_windows.shape[0] == 0:
        return np.array([])
    
    # Calculate RMS (Root Mean Square) across time and signals for each window
    # This gives us a measure of overall deviation magnitude
    rms_per_window = np.sqrt(np.mean(fixed_windows**2, axis=(1, 2)))
    
    # Handle NaN values (set to minimum)
    rms_per_window = np.nan_to_num(rms_per_window, nan=np.nanmin(rms_per_window[~np.isnan(rms_per_window)]))
    
    # Map RMS values to price range [1.0, 50.0]
    # Higher RMS = higher price (more abnormal = more expensive)
    if len(rms_per_window) > 0 and np.max(rms_per_window) > np.min(rms_per_window):
        # Normalize to [0, 1] then scale to [1, 50]
        normalized = (rms_per_window - np.min(rms_per_window)) / (np.max(rms_per_window) - np.min(rms_per_window))
        baseline_prices = 1.0 + normalized * 49.0  # Maps to [1, 50]
    else:
        # All values are the same - assign middle price
        baseline_prices = np.full(len(rms_per_window), 25.0)
    
    return baseline_prices

# --- Steps 3 & 4: Billing and Ground Truth Calculation ---

def calculate_prices(
    fixed_windows: np.ndarray,
    centroid_prices: np.ndarray,
    final_kmeans_centroids: np.ndarray,
    scaler_kmeans: StandardScaler,
    pricing_strategy_config: dict,
    hybrid_config: dict,
    post_process_config: dict
) -> dict:
    """Calculates prices for fixed-size windows using all requested methods."""
    if fixed_windows.shape[0] == 0:
        return {}

    results: Dict[str, np.ndarray] = {}
    kmeans_prices, baseline_prices = None, None

    # --- Feature Extraction (common for kmeans and hybrid) ---
    if any(m in METHODS_TO_RUN for m in ['kmeans', 'hybrid']):
        fixed_windows_features = extract_enhanced_features_from_reconstruction_errors(fixed_windows)
        assert isinstance(fixed_windows_features, np.ndarray)
        # Features for kmeans are now the scaled, high-dimensional features
        features_for_kmeans = scaler_kmeans.transform(fixed_windows_features)
        assert isinstance(features_for_kmeans, np.ndarray)

    # --- K-Means prices (needed for 'kmeans' and 'hybrid') ---
    if 'kmeans' in METHODS_TO_RUN or 'hybrid' in METHODS_TO_RUN:
        # Calculate distances in the high-dimensional space
        kmeans_distances = np.linalg.norm(features_for_kmeans[:, np.newaxis, :] - final_kmeans_centroids, axis=2)
        
        kmeans_prices = calculate_prices_with_strategy(
            kmeans_distances, centroid_prices,
            strategy=pricing_strategy_config['method'],
            threshold_distance=pricing_strategy_config['threshold_distance'],
            exponential_power=pricing_strategy_config['exponential_power']
        )
        if 'kmeans' in METHODS_TO_RUN:
            results["kmeans"] = kmeans_prices

    # --- Baseline prices (needed for 'baseline' and 'hybrid') ---
    if 'baseline' in METHODS_TO_RUN or 'hybrid' in METHODS_TO_RUN:
        baseline_prices = calculate_baseline_prices(fixed_windows)
        if 'baseline' in METHODS_TO_RUN:
            results["baseline"] = baseline_prices

    # --- Hybrid method (combines the two) ---
    if 'hybrid' in METHODS_TO_RUN:
        if kmeans_prices is not None and baseline_prices is not None:
            # Use POST-PROCESSED kmeans and ORIGINAL baseline for hybrid calculation.
            # The result is the final "original" cost for the hybrid method.
            kmeans_prices_processed = smart_post_process_prices(
                kmeans_prices,
                floor_method=post_process_config['floor_method'],
                gamma=post_process_config['gamma'],
                dead_zone_k=post_process_config['dead_zone_k']
            )
            
            alpha = hybrid_config.get('alpha', 0.3)
            median_baseline = np.median(baseline_prices)
            modulation = alpha * (baseline_prices - median_baseline)
            
            hybrid_prices = kmeans_prices_processed + modulation
            results['hybrid'] = np.clip(hybrid_prices, 1.0, 50.0)

    return results

def calculate_ground_truth_price(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """Calculates the ground-truth price for each fixed-size window."""
    num_windows = len(df) // window_size
    if num_windows == 0:
        return np.array([])
        
    context_col = 'CONTEXT' if 'CONTEXT' in df.columns else 'context'
    context_prices = df[context_col].apply(get_price_for_context).to_numpy()
    
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
    sim_id: int,
    post_process_config: dict
) -> dict:
    """Calculates metrics and generates plots for a single run - focused on original and stretched analysis."""
    eval_results = {}
    for method, calculated in results.items():
        if len(calculated) != len(ground_truth) or len(calculated) == 0:
            print(f"  Skipping evaluation for {method} due to length mismatch or empty data.")
            continue

        # Original metrics
        corr, _ = pearsonr(calculated, ground_truth)
        mae = mean_absolute_error(ground_truth, calculated)
        rmse = np.sqrt(mean_squared_error(ground_truth, calculated))
        
        # --- Smart post-processing (robust) ----------------------------------
        # --- Parameters ------------------------------------------------------
        floor_method = post_process_config['floor_method']
        gamma_val = post_process_config['gamma']
        n_bins_dbg = 40
        dead_zone_k_dbg = post_process_config['dead_zone_k']

        # --- Debug: compute baseline, epsilon, and intermediate arrays ------
        hist_dbg, bin_edges_dbg = np.histogram(calculated, bins=n_bins_dbg)
        mode_idx_dbg = int(np.argmax(hist_dbg))
        baseline_dbg = (bin_edges_dbg[mode_idx_dbg] + bin_edges_dbg[mode_idx_dbg + 1]) / 2.0

        median_dbg = np.median(calculated)

        deviations_dbg = np.abs(calculated - baseline_dbg)
        mad_dbg = np.median(np.abs(deviations_dbg - np.median(deviations_dbg)))
        sigma_hat_dbg = 1.4826 * mad_dbg
        epsilon_dbg = dead_zone_k_dbg * sigma_hat_dbg if sigma_hat_dbg > 0 else 0.0

        mirrored_dbg = np.where(calculated < baseline_dbg, 2.0 * baseline_dbg - calculated, calculated)
        mirrored_clipped_dbg = np.where(np.abs(mirrored_dbg - baseline_dbg) < epsilon_dbg, baseline_dbg, mirrored_dbg)

        # Final scaled using same path as helper (duplicated to keep arrays) --
        if np.isclose(np.max(mirrored_clipped_dbg), baseline_dbg):
            final_scaled_dbg = np.full_like(calculated, 25.0)
        else:
            normalised_dbg = (mirrored_clipped_dbg - baseline_dbg) / (np.max(mirrored_clipped_dbg) - baseline_dbg)
            final_scaled_dbg = 1.0 + 49.0 * (normalised_dbg ** gamma_val)

        smart_stretched_calculated = final_scaled_dbg  # keep naming consistent

        # --- Select cost for evaluation based on method ---
        if method == 'kmeans':
            eval_cost = smart_stretched_calculated
            print(f"  Using POST-PROCESSED costs for '{method}' evaluation.")
        else: # baseline, hybrid
            eval_cost = calculated
            print(f"  Using ORIGINAL costs for '{method}' evaluation.")

        # --- Final Metrics Calculation ---
        # Handle cases where all values are constant
        if np.all(eval_cost == eval_cost[0]) or np.all(ground_truth == ground_truth[0]):
            final_corr = 0.0 # Or some other indicator for constant data
        else:
            final_corr, _ = pearsonr(eval_cost, ground_truth)
        
        final_mae = mean_absolute_error(ground_truth, eval_cost)
        final_rmse = np.sqrt(mean_squared_error(ground_truth, eval_cost))
        
        eval_results[method] = {
            "pearson_corr": final_corr, "mae": final_mae, "rmse": final_rmse,
        }
        
        # Plotting directory
        plot_dir = os.path.join(RESULTS_DIR, group_name, method)
        os.makedirs(plot_dir, exist_ok=True)

        # ----- Tracking visualisation (more compact) -----------------------
        fig, axes = plt.subplots(3, 1, figsize=(10, 24))

        # --- Calculate metrics for each subplot ---
        # Original
        corr_orig, _ = pearsonr(calculated, ground_truth) if not (np.all(calculated == calculated[0]) or np.all(ground_truth == ground_truth[0])) else (0.0, 0.0)
        mae_orig = mean_absolute_error(ground_truth, calculated)
        # Mirrored
        corr_mid, _ = pearsonr(mirrored_clipped_dbg, ground_truth) if not (np.all(mirrored_clipped_dbg == mirrored_clipped_dbg[0]) or np.all(ground_truth == ground_truth[0])) else (0.0, 0.0)
        mae_mid = mean_absolute_error(ground_truth, mirrored_clipped_dbg)
        # Final
        corr_final_plot, _ = pearsonr(smart_stretched_calculated, ground_truth) if not (np.all(smart_stretched_calculated == smart_stretched_calculated[0]) or np.all(ground_truth == ground_truth[0])) else (0.0, 0.0)
        mae_final_plot = mean_absolute_error(ground_truth, smart_stretched_calculated)

        # (1) Original prices with diagnostics
        ax_orig = axes[0]
        ax_orig.plot(ground_truth, label='Ground-Truth', color='black', lw=2, alpha=0.5)
        ax_orig.plot(calculated, label='Raw Calc', color='blue', alpha=0.8)
        ax_orig.axhline(baseline_dbg, color='orange', ls='--', label=f'Baseline (mode={baseline_dbg:.2f})')
        ax_orig.axhline(median_dbg, color='green', ls=':', label=f'Median={median_dbg:.2f}')
        if epsilon_dbg > 0:
            ax_orig.axhline(baseline_dbg + epsilon_dbg, color='red', ls='--', alpha=0.5, label=f'+ε ({epsilon_dbg:.2f})')
            ax_orig.axhline(max(0, baseline_dbg - epsilon_dbg), color='red', ls='--', alpha=0.5)
        ax_orig.set_title(f'Original Prices\nCorr={corr_orig:.3f}, MAE={mae_orig:.3f}', fontsize=18)
        ax_orig.set_xlabel('Window', fontsize=18)
        ax_orig.set_ylabel('Price', fontsize=18)
        ax_orig.legend(fontsize=18)
        ax_orig.grid(True, ls='--', alpha=0.5)

        # (2) Mirrored + dead-zone prices
        ax_mid = axes[1]
        ax_mid.plot(ground_truth, label='Ground-Truth', color='black', lw=2, alpha=0.5)
        ax_mid.plot(mirrored_clipped_dbg, label='Mirrored+DeadZone', color='darkorange', alpha=0.8)
        ax_mid.axhline(baseline_dbg, color='orange', ls='--', alpha=0.7)
        if epsilon_dbg > 0:
            ax_mid.axhline(baseline_dbg + epsilon_dbg, color='red', ls='--', alpha=0.4)
        ax_mid.set_title(f'After Reflection & Dead-Zone\nCorr={corr_mid:.3f}, MAE={mae_mid:.3f}', fontsize=18)
        ax_mid.set_xlabel('Window', fontsize=18)
        ax_mid.set_ylabel('Price', fontsize=18)
        ax_mid.grid(True, ls='--', alpha=0.5)
        ax_mid.legend(fontsize=18)

        # (3) Final scaled prices
        ax_final = axes[2]
        ax_final.plot(ground_truth, label='Ground-Truth', color='black', lw=2, alpha=0.5)
        
        # Color based on method for final plot
        plot_color = 'purple'
        if method == 'baseline':
            plot_color = 'red'
        elif method == 'hybrid':
            plot_color = 'orange'
        
        ax_final.plot(smart_stretched_calculated, label='Final Scaled', color=plot_color, alpha=0.8)
        ax_final.set_title(f'Post-Processed [1-50]\nCorr={corr_final_plot:.3f}, MAE={mae_final_plot:.3f}', fontsize=18)
        ax_final.set_xlabel('Window', fontsize=18)
        ax_final.set_ylabel('Price', fontsize=18)
        ax_final.grid(True, ls='--', alpha=0.5)
        ax_final.legend(fontsize=18)

        fig.suptitle(f'{method.upper()} – Simulation {sim_id}', fontsize=18)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(os.path.join(plot_dir, f"sim_{sim_id}_tracking_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    return eval_results

def plot_pca_cluster_scatterplot(
    scaled_features: np.ndarray,
    cluster_labels: Union[np.ndarray, List[int]],
    contexts: List[str],
    group_name: str,
    output_dir: str
):
    """
    Generates a PCA scatter plot of clustered data, colored by context.

    Args:
        scaled_features: The scaled feature matrix (n_samples, n_features).
        cluster_labels: The cluster label for each sample.
        contexts: The ground-truth context for each sample.
        group_name: Name of the physics group for titling.
        output_dir: Directory to save the plot.
    """
    print("  Generating PCA cluster scatter plot for visualization...")

    # Always use PCA with 2 components for visualization
    try:
        pca_viz = PCA(n_components=2, random_state=42)
        principal_components = pca_viz.fit_transform(scaled_features)
    except Exception as e:
        print(f"    Could not perform PCA for visualization: {e}")
        return

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'PC1': principal_components[:, 0],
        'PC2': principal_components[:, 1],
        'context': [get_generic_context(c) for c in contexts],
        'cluster': cluster_labels
    })

    # Calculate centroids in the PCA space
    pca_centroids = df_plot.groupby('cluster')[['PC1', 'PC2']].mean()

    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Use a color palette
    unique_contexts = sorted(df_plot['context'].unique())
    palette = sns.color_palette("husl", n_colors=len(unique_contexts))
    context_colors = dict(zip(unique_contexts, palette))

    ax = sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='context',
        palette=context_colors,
        style='cluster',
        alpha=0.8,
        s=80
    )

    # Plot centroids
    ax.scatter(
        pca_centroids['PC1'],
        pca_centroids['PC2'],
        marker='X',
        s=200,
        c='black',
        edgecolors='white',
        linewidth=1,
        label='Cluster Centroids'
    )

    # Annotate centroids
    for i, centroid in pca_centroids.iterrows():
        ax.text(centroid['PC1'] + 0.05, centroid['PC2'] + 0.05, f'C{i}', fontsize=12, weight='bold', color='black')

    ax.set_title(f'PCA Scatter Plot of Clusters for {group_name}', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(title='Context', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{group_name}_pca_cluster_plot.png")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  PCA scatter plot saved to {plot_path}")

def plot_group_summary_tracking(group_name: str, all_run_results: list):
    """
    Plots the concatenated ground truth vs. calculated prices for an entire physics group,
    ordered by ground-truth price. Uses stretched version for kmeans.
    """
    if not all_run_results:
        return

    for method in METHODS_TO_RUN:
        # Gather parts for this method
        gt_parts = [r['ground_truth'] for r in all_run_results if r]
        calc_parts = [r['results'][method] for r in all_run_results if r and method in r['results']]

        if not gt_parts or not calc_parts:
            print(f"  Skipping group summary plot for method '{method}' - no data found.")
            continue

        # Concatenate results from all runs in the group
        ground_truth_all = np.concatenate(gt_parts)
        calculated_all = np.concatenate(calc_parts)
        
        if len(ground_truth_all) == 0 or len(calculated_all) == 0:
            continue

        # Use post-processed costs for specified methods for this plot
        if method == 'kmeans':
            calculated_all = smart_post_process_prices(calculated_all)
            display_method = f"{method.upper()} (Post-Processed)"
            plot_color = 'purple'
        else: # baseline, hybrid
            display_method = f"{method.upper()} (Original)"
            if method == 'baseline':
                plot_color = 'red'
            elif method == 'hybrid':
                plot_color = 'orange'
            else: # xgboost
                plot_color = 'green'

        # Sort data by ground-truth price
        sort_indices = np.argsort(ground_truth_all)
        ground_truth_sorted = ground_truth_all[sort_indices]
        calculated_sorted = calculated_all[sort_indices]

        plt.figure(figsize=(12, 12))
        plt.plot(ground_truth_sorted, label='Ground-Truth Price', color='black', lw=2, alpha=0.7)
        plt.plot(calculated_sorted, label=f'Calculated Price ({display_method})', color=plot_color, alpha=0.7, linestyle='--')
        
        # Add a diagonal reference line for perfect prediction
        plt.plot([ground_truth_sorted.min(), ground_truth_sorted.max()], 
                [ground_truth_sorted.min(), ground_truth_sorted.max()], 
                'g--', alpha=0.5, label='Perfect Prediction')

        plt.title(f'Group Summary Price Tracking: {group_name} - {display_method}', fontsize=18)
        plt.xlabel("Data Points (Sorted by Ground-Truth Price)", fontsize=18)
        plt.ylabel("Price", fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = os.path.join(RESULTS_DIR, group_name, f"summary_tracking_{method}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  Group summary plot saved to: {output_path}") 

# --- Utility: Robust Smart Post-Processing -----------------------------------

def smart_post_process_prices(
    prices: np.ndarray,
    *,
    floor_method: str = "mode",   # 'mode', 'median', 'percentile'
    gamma: float = 0.6,            # <1 expands mid-range, compresses extremes
    n_bins: int = 40,
    percentile_value: float = 40.0,
    dead_zone_k: float = 3.0       # k * σ̂ (σ̂≈1.4826·MAD) within which bumps are flattened
) -> np.ndarray:
    """Transform *calculated* prices into the final [1, 50] billing range.

    Steps
    -----
    1. Determine a *baseline* (the "floor") using *floor_method*:
       • 'mode' – bin the data and take the bin centre with the most samples.
       • 'median' – classic median.
       • 'percentile' – lower percentile (e.g. 40th).
    2. Compute MAD-based ε (dead-zone):
       • Median Absolute Deviation (MAD)
       • Robust σ estimate under normal noise (σ̂≈1.4826·MAD)
       • ε = k * σ̂ (k=3.0 for 99.7% coverage)
    3. Mirror values below the baseline so negative peaks become positive ones.
    4. Apply a power-law (gamma) to compress very large peaks (gamma<1).
    5. Stretch linearly to the [1, 50] billing range.

    The function gracefully handles corner-cases (empty / flat arrays).
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        return prices

    # --- Step 1: baseline ----------------------------------------------------
    if floor_method == "mode":
        hist, bin_edges = np.histogram(prices, bins=n_bins)
        mode_idx = int(np.argmax(hist))
        baseline = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
    elif floor_method == "median":
        baseline = float(np.median(prices))
    elif floor_method == "percentile":
        baseline = float(np.percentile(prices, percentile_value))
    else:
        raise ValueError(f"Unknown floor_method '{floor_method}'.")

    # --- Step 2: compute MAD-based ε (dead-zone) ----------------------------
    deviations = np.abs(prices - baseline)
    # Median Absolute Deviation
    mad = np.median(np.abs(deviations - np.median(deviations)))
    sigma_hat = 1.4826 * mad  # robust σ estimate under normal noise
    epsilon = dead_zone_k * sigma_hat if sigma_hat > 0 else 0.0

    # --- Step 3: mirror values below baseline ------------------------------
    mirrored = np.where(prices < baseline, 2.0 * baseline - prices, prices)

    # Apply dead-zone: bumps with deviation < ε are treated as baseline
    if epsilon > 0:
        mirrored = np.where(np.abs(mirrored - baseline) < epsilon, baseline, mirrored)

    max_val = float(np.max(mirrored))
    if np.isclose(max_val, baseline):
        # Flat profile – map to mid-price 25.0
        return np.full_like(prices, 25.0)

    # --- Step 4 & 5: normalise + power-law scaling --------------------------
    normalised = (mirrored - baseline) / (max_val - baseline)
    gamma = max(1e-3, float(gamma))
    transformed = normalised ** gamma

    # --- Step 6: map to [1, 50] --------------------------------------------
    scaled = 1.0 + 49.0 * transformed
    return scaled

# --- Main Execution ---

def main():
    """Main function to run the enhanced experiment using reconstruction error signals."""
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Run the enhanced billing validation experiment using reconstruction error signals.")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Run the experiment for a single specified physics group (e.g., 'group_1')."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_experiment2_old",
        help="Directory to save all results, plots, and summaries."
    )
    args = parser.parse_args()

    # Update RESULTS_DIR from CLI and ensure it exists
    RESULTS_DIR = args.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    groups_to_process = PHYSICS_GROUPS
    if args.group:
        if args.group in PHYSICS_GROUPS:
            groups_to_process = {args.group: PHYSICS_GROUPS[args.group]}
            print(f"--- Running enhanced experiment for single specified group: {args.group} ---")
        else:
            print(f"Error: Group '{args.group}' not found in PHYSICS_GROUPS. Available groups are: {list(PHYSICS_GROUPS.keys())}")
            return
    elif args.group is None:
        print("--- Running enhanced experiment for ALL physics groups ---")
    
    # Create config dict for post-processing to pass to functions
    post_process_config = {
        'floor_method': 'mode',
        'gamma': 0.6,
        'dead_zone_k': 3.0,
    }

    print("🚗 EXPERIMENT 2: Enhanced Billing Validation using Reconstruction Error Signals")
    print("=" * 80)
    print(f"📊 Features: Time-domain + Enhanced FFT + Morlet Wavelet")
    print(f"🔧 Input: {len(RECONSTRUCTION_ERROR_COLUMNS)} reconstruction error signals")
    print(f"📈 Methods: {METHODS_TO_RUN}")
    print(f"🎯 Dimensionality Reduction: {DIMENSIONALITY_REDUCTION['method']}")
    print(f"💰 Pricing Strategy: {PRICING_STRATEGY['method']}")
    print(f"⚙️ Post-Processing: gamma={post_process_config['gamma']}, dead_zone_k={post_process_config['dead_zone_k']}")
    print("=" * 80)

    all_results_list = []

    for group_name, params in groups_to_process.items():
        print(f"\n--- Processing {group_name} (Mass: {params['mass']}, Friction: {params['friction']}) ---")
        
        # === Load Trained Autoencoder ===
        try:
            print(f"Loading trained autoencoder for {group_name}...")
            autoencoder_model, autoencoder_scaler = load_trained_autoencoder(group_name)
            print("✅ Autoencoder model loaded successfully!")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print(f"Please train the autoencoder for {group_name} first using:")
            print(f"python recurrent_autoencoder_anomaly_detection.py --group {group_name}")
            continue
        except Exception as e:
            print(f"❌ Error loading autoencoder for {group_name}: {e}")
            continue
        
        # === Step 2: Centroid Identification ===
        
        # 1. Load Group Data
        group_csv_files = []
        for sim_num in params['seq_range']:
            # Correct glob pattern for filenames like 'simulation_1_...'
            pattern = os.path.join(PROCESSED_DATA_DIR, f"simulation_{sim_num}_*.csv")
            found_files = glob.glob(pattern)
            if found_files:
                group_csv_files.extend(found_files)
                print(f"  Found {len(found_files)} files for sim_num {sim_num} with pattern: {pattern}")
            else:
                print(f"  No files found for sim_num {sim_num} with pattern: {pattern}")

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
        
        # 2. Generate Reconstruction Error Signals for Training Data
        print(f"  Generating reconstruction error signals for training data...")
        train_dfs_with_errors = []
        
        for train_file in train_files:
            try:
                df = pd.read_csv(train_file)
                df = process_data_for_autoencoder(df)
                df_with_errors = generate_reconstruction_error_signals(df, autoencoder_model, autoencoder_scaler)
                train_dfs_with_errors.append(df_with_errors)
            except Exception as e:
                print(f"    Error processing {train_file}: {e}")
                continue
        
        if not train_dfs_with_errors:
            print(f"  Could not generate reconstruction error signals for any training files. Skipping group.")
            continue
        
        # Combine all training data
        train_df = pd.concat(train_dfs_with_errors, ignore_index=True)
        
        # Check if we have the required reconstruction error columns
        available_error_columns = [col for col in RECONSTRUCTION_ERROR_COLUMNS if col in train_df.columns]
        if not available_error_columns:
            print(f"  No reconstruction error columns found in training data. Skipping group.")
            continue
        
        print(f"  Using {len(available_error_columns)} reconstruction error signals as features")
        
        # 3. Create initial centroids from representative "archetype" windows
        context_col = 'CONTEXT' if 'CONTEXT' in train_df.columns else 'context'
        
        valid_rows = ~train_df[available_error_columns].isnull().any(axis=1)
        train_df_clean = train_df[valid_rows].copy()
        
        if train_df_clean.empty:
            print(f"  No valid reconstruction error data found for {group_name}. Skipping.")
            continue
        
        context_blocks = extract_contiguous_context_blocks(train_df_clean, available_error_columns, context_col)
        archetype_windows, archetype_contexts = create_context_representative_samples(context_blocks)

        if not archetype_windows:
            print(f"  Could not extract any archetype windows for centroid initialization. Skipping.")
            continue
            
        grouped_archetypes = defaultdict(list)
        for window, context in zip(archetype_windows, archetype_contexts):
            grouped_archetypes[get_generic_context(context)].append(window)

        initial_centroids = []
        centroid_contexts = []
        for ctx, w_list in grouped_archetypes.items():
            valid_windows = [w for w in w_list if w.shape[0] > 1]
            if not valid_windows: continue
            initial_centroids.append(random.choice(valid_windows))
            centroid_contexts.append(ctx)

        if not initial_centroids:
            print(f"  No valid initial centroids could be selected for {group_name}. Skipping group.")
            continue
            
        print(f"  Selected {len(initial_centroids)} initial centroids from archetype windows.")

        # 4. Create a balanced training set for KMeans fitting
        all_training_windows, all_training_contexts = create_fixed_windows_for_training(
            train_df_clean, available_error_columns, context_col=context_col
        )

        if not all_training_windows:
            print("  No training windows could be generated. Skipping group.")
            continue
            
        all_windows_grouped_by_context = defaultdict(list)
        for window, context in zip(all_training_windows, all_training_contexts):
            all_windows_grouped_by_context[get_generic_context(context)].append(window)
        
        # Remove contexts that didn't yield any windows
        all_windows_grouped_by_context = {k: v for k, v in all_windows_grouped_by_context.items() if v}

        if not all_windows_grouped_by_context:
            print("  Could not group any training windows by context. Skipping group.")
            continue

        min_windows_per_context = min(len(w) for w in all_windows_grouped_by_context.values())
        print(f"  Balancing dataset: Found {min_windows_per_context} windows for the least frequent context.")
        print(f"  Selecting {min_windows_per_context} windows from each context for training.")

        balanced_windows = []
        balanced_contexts = []
        for ctx, w_list in all_windows_grouped_by_context.items():
            sampled_windows = random.sample(w_list, min_windows_per_context)
            balanced_windows.extend(sampled_windows)
            balanced_contexts.extend([ctx] * min_windows_per_context)
            
        print(f"  Created a balanced training set with {len(balanced_windows)} total windows.")

        # 5. Set Up Pricing
        centroid_prices = np.array([get_price_for_context(c) for c in centroid_contexts])
        print(f"  Centroid contexts: {centroid_contexts}")
        print(f"  Centroid prices: {centroid_prices}")
        print(f"  Price range: {np.min(centroid_prices):.1f} to {np.max(centroid_prices):.1f}")
        print(f"  Using pricing strategy: {PRICING_STRATEGY['method']}")

        # === Enhanced Model Fitting ===
        
        n_clusters = len(initial_centroids)
        
        # === KMeans Model Fitting with Intelligent Dimensionality Reduction ===
        print("  Fitting KMeans with intelligent dimensionality reduction on the balanced dataset...")
        
        # A. Extract enhanced features from all balanced training windows
        all_windows_features_list = []
        for window in balanced_windows:
            window_array = window[np.newaxis, :, :] if window.ndim == 2 else window
            features = extract_enhanced_features_from_reconstruction_errors(window_array)
            all_windows_features_list.append(features[0])
        
        all_windows_features = np.vstack(all_windows_features_list)
        print(f"  Extracted features shape from balanced set: {all_windows_features.shape}")
        
        # B. Apply standardization
        scaler_kmeans = StandardScaler().fit(all_windows_features)
        scaled_features = scaler_kmeans.transform(all_windows_features)
        assert isinstance(scaled_features, np.ndarray)
        
        # C. Dimensionality reduction is no longer applied to the training pipeline.
        #    KMeans will be fitted on the scaled, high-dimensional features.
        
        # D. Transform initial centroids to same feature space
        initial_centroids_features_list = []
        for centroid in initial_centroids:
            centroid_array = centroid[np.newaxis, :, :] if centroid.ndim == 2 else centroid
            features = extract_enhanced_features_from_reconstruction_errors(centroid_array)
            initial_centroids_features_list.append(features[0])
        
        initial_centroids_features = np.vstack(initial_centroids_features_list)
        scaled_centroids = scaler_kmeans.transform(initial_centroids_features)
        # No further transformation needed for initial centroids

        # E. Fit KMeans model on scaled, high-dimensional features
        kmeans_model = KMeans(n_clusters=n_clusters, init=scaled_centroids, n_init=1, random_state=42).fit(scaled_features)  # type: ignore
        final_kmeans_centroids = kmeans_model.cluster_centers_
        
        print(f"  KMeans trained with {n_clusters} clusters in {scaled_features.shape[1]}D space")

        # Generate and save the PCA scatter plot for cluster visualization
        if getattr(kmeans_model, 'labels_', None) is not None:
            # The labels_ attribute exists and is not None
            labels_arr = np.asarray(kmeans_model.labels_)
            cluster_labels_safe = labels_arr.tolist()
        else:
            # Fallback if labels_ does not exist or is None
            cluster_labels_safe = [0] * len(scaled_features)

        plot_pca_cluster_scatterplot(
            scaled_features=scaled_features,
            cluster_labels=cluster_labels_safe,
            contexts=balanced_contexts,
            group_name=group_name,
            output_dir=os.path.join(RESULTS_DIR, group_name)
        )

        # === Steps 3, 4, 5: Runtime Simulation, Billing, and Evaluation ===
        
                        # TRAINING EVALUATION: Check performance on training data
        print("  Evaluating performance on TRAINING data...")
        train_run_results_for_plotting = []
        for csv_file in train_files:
            sim_id_match = re.search(r'simulation_(\d+)_', os.path.basename(csv_file))
            if not sim_id_match: continue
            sim_id = int(sim_id_match.group(1))
            
            print(f"    Evaluating training simulation {sim_id}...")
            
            try:
                run_df = pd.read_csv(csv_file)
                run_df = process_data_for_autoencoder(run_df)
                run_df_with_errors = generate_reconstruction_error_signals(run_df, autoencoder_model, autoencoder_scaler)
                
                if run_df_with_errors.empty: continue

                # Create fixed-size windows from reconstruction error signals
                error_data = run_df_with_errors[available_error_columns].values
                
                # Filter out NaN rows
                valid_mask = ~np.isnan(error_data).any(axis=1)
                error_data_clean = error_data[valid_mask]
                
                if len(error_data_clean) < FIXED_WINDOW_SIZE: continue
                
                fixed_windows_errors = create_windows(error_data_clean, FIXED_WINDOW_SIZE)
                if fixed_windows_errors.shape[0] == 0: continue

                # Calculate Price using trained models
                calculated_prices = calculate_prices(
                    fixed_windows_errors,
                    centroid_prices,
                    final_kmeans_centroids,
                    scaler_kmeans,
                    PRICING_STRATEGY,
                    HYBRID_PARAMS,
                    post_process_config
                )
                
                # Ground-Truth Price Calculation (using valid rows only)
                context_col = 'CONTEXT' if 'CONTEXT' in run_df_with_errors.columns else 'context'
                run_df_valid = run_df_with_errors[valid_mask].copy()
                assert isinstance(run_df_valid, pd.DataFrame)  # Type assertion for linter
                ground_truth_prices = calculate_ground_truth_price(run_df_valid, FIXED_WINDOW_SIZE)

                # Evaluation and Analysis (Training)
                run_evals = perform_evaluation_and_plot(
                    calculated_prices, 
                    ground_truth_prices, 
                    f"{group_name}_TRAIN", 
                    sim_id,
                    post_process_config
                )
                
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
                        "data_type": "TRAINING",
                        **metrics
                    }
                    all_results_list.append(result_row)
                    
            except Exception as e:
                print(f"    Error evaluating simulation {sim_id}: {e}")
                continue
        
                        # VALIDATION EVALUATION: Test on validation files (never seen during training)
        print("  Evaluating performance on VALIDATION data...")
        validation_run_results_for_plotting = []
        for csv_file in validation_files:
            sim_id_match = re.search(r'simulation_(\d+)_', os.path.basename(csv_file))
            if not sim_id_match: continue
            sim_id = int(sim_id_match.group(1))
            
            print(f"    Evaluating validation simulation {sim_id}...")
            
            try:
                run_df = pd.read_csv(csv_file)
                run_df = process_data_for_autoencoder(run_df)
                run_df_with_errors = generate_reconstruction_error_signals(run_df, autoencoder_model, autoencoder_scaler)
                
                if run_df_with_errors.empty: continue

                # Create fixed-size windows from reconstruction error signals
                error_data = run_df_with_errors[available_error_columns].values
                
                # Filter out NaN rows
                valid_mask = ~np.isnan(error_data).any(axis=1)
                error_data_clean = error_data[valid_mask]
                
                if len(error_data_clean) < FIXED_WINDOW_SIZE: continue
                
                fixed_windows_errors = create_windows(error_data_clean, FIXED_WINDOW_SIZE)
                if fixed_windows_errors.shape[0] == 0: continue

                # Calculate Price using trained models
                calculated_prices = calculate_prices(
                    fixed_windows_errors,
                    centroid_prices,
                    final_kmeans_centroids,
                    scaler_kmeans,
                    PRICING_STRATEGY,
                    HYBRID_PARAMS,
                    post_process_config
                )
                
                # Ground-Truth Price Calculation (using valid rows only)
                context_col = 'CONTEXT' if 'CONTEXT' in run_df_with_errors.columns else 'context'
                run_df_valid = run_df_with_errors[valid_mask].copy()
                assert isinstance(run_df_valid, pd.DataFrame)  # Type assertion for linter
                ground_truth_prices = calculate_ground_truth_price(run_df_valid, FIXED_WINDOW_SIZE)

                # Evaluation and Analysis (Validation)
                run_evals = perform_evaluation_and_plot(
                    calculated_prices, 
                    ground_truth_prices, 
                    f"{group_name}_VAL", 
                    sim_id,
                    post_process_config
                )
                
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
                        "data_type": "VALIDATION",
                        **metrics
                    }
                    all_results_list.append(result_row)
                    
            except Exception as e:
                print(f"    Error evaluating simulation {sim_id}: {e}")
                continue
        
        # After processing all runs in a group, create the summary plots
        plot_group_summary_tracking(f"{group_name}_TRAIN", train_run_results_for_plotting)
        plot_group_summary_tracking(f"{group_name}_VAL", validation_run_results_for_plotting)
    
    # === Step 6: Final Comparison Across Physics Groups ===
    if not all_results_list:
        print("\nNo results were generated. Exiting.")
        return
        
    summary_df = pd.DataFrame(all_results_list)
    
    # Save the full results table
    summary_path = os.path.join(RESULTS_DIR, "full_evaluation_results_experiment2.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nFull evaluation results saved to {summary_path}")

    # Create and print the summary table with mean and standard deviation for focused metrics
    agg_metrics = summary_df.groupby(['mass', 'friction', 'method', 'data_type']).agg(
        corr_mean=('pearson_corr', 'mean'),
        corr_std=('pearson_corr', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std')
    ).reset_index()
    
    agg_path = os.path.join(RESULTS_DIR, "summary_evaluation_by_group_experiment2.csv")
    agg_metrics.to_csv(agg_path, index=False)
    print("\n" + "="*100)
    print("🎯 EXPERIMENT 2 RESULTS: Enhanced Features + Reconstruction Error Signals + Baseline")
    print("="*100)
    print("📊 Methods: KMeans (context-aware) + Baseline (raw error magnitudes) + Hybrid")
    print("✍️ Analysis: Original cost for Baseline & Hybrid. Post-processed for KMeans.")
    print(f"🎯 KMeans Pricing Strategy: {PRICING_STRATEGY['method']} ({PRICING_STRATEGY['threshold_distance']*100}% closer)")
    print("="*100)
    print(agg_metrics.to_string())
    print(f"\nDetailed results saved to {summary_path}")
    print(f"Summary table saved to {agg_path}")
    print("\n💡 Key Metrics Explanation:")
    print("  - corr_mean: Mean pearson correlation")
    print("  - mae_mean: Mean absolute error")
    print("  - rmse_mean: Mean root mean squared error")
    print("  - Note: Costs used for evaluation are:")
    print("    - Baseline/Hybrid: Original calculated costs")
    print("    - KMeans: Post-processed costs")
    print("="*100)

if __name__ == "__main__":
    main() 