import numpy as np
import torch
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset, to_time_series
from tslearn.metrics import dtw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, sosfilt
from load_data import load_data
from loader_pastuch import load_data_pastuch
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

USE_PCA_PREPROCESSING = True
USE_PCA_AS_DEGRADATION = False
USE_SUM_CENTROID_FEATURES = True
USE_AVERAGE_CENTROIDS = False
PREPROCESSING_PCA_COMPONENTS = 1
FEATURE_EXTRACTION_DOMAIN = 'time'
# Step 1: Data Generation
def generate_simulated_data(num_samples, num_timestamps, num_sensors):
    """Generates simulated time series data for testing."""
    state_size = num_samples // 3
    data = np.empty((num_samples, num_timestamps, num_sensors))

    data[:state_size] = np.random.uniform(1, 2, (state_size, num_timestamps, num_sensors))
    data[state_size:2*state_size] = np.random.uniform(2, 3, (state_size, num_timestamps, num_sensors))
    data[2*state_size:] = np.random.uniform(3, 4, (state_size, num_timestamps, num_sensors))

    return data

# Step 2: Window Creation
def create_windows(data, id_data, target_data, window_size):
    """Creates non-overlapping windows from time series data."""
    print("len(data): ", len(data))
    print("len(id_data): ", len(id_data))
    print("len(target_data): ", len(target_data))

    windows, identifiers, targets = [], [], []
    for i, array in enumerate(data):
        num_timestamps = array.shape[0]
        num_windows = num_timestamps // window_size
        for j in range(num_windows):
            start, end = j * window_size, (j + 1) * window_size
            windows.append(array[start:end])
            identifiers.append(id_data[i])
            targets.append(target_data[i])

    return np.array(windows), np.array(identifiers), np.array(targets)

# Step 3: Preprocessing Functions
def dc_component_removal(signal):
    return signal - torch.mean(signal, dim=-1, keepdim=True)

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    signal_np = signal.cpu().numpy()
    filtered_signal_np = sosfilt(sos, signal_np, axis=1)
    return torch.from_numpy(filtered_signal_np).to(signal.device)

def extract_features(signal, domain='frequency'):
    """Extracts features from signals."""
    if domain == 'frequency':
        fft_signal = torch.fft.rfft(signal, dim=1).abs()
    else:
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
    features[:, 7, :] = torch.sqrt(torch.sum(fft_signal, dim=1)) / torch.mean(torch.abs(fft_signal), dim=1)
    features[:, 8, :] = torch.max(fft_signal, dim=1)[0] / torch.sqrt(torch.mean(fft_signal**2, dim=1))
    features[:, 9, :] = torch.sum(fft_signal**2, dim=1)

    psd = fft_signal**2
    psd_normalized = psd / torch.sum(psd, dim=1, keepdim=True)
    features[:, 10, :] = -torch.sum(psd_normalized * torch.log2(psd_normalized + 1e-8), dim=1)

    features[torch.isnan(features)] = 0
    features[torch.isinf(features)] = 0
    return features

def preprocess_and_extract(signal, lowcut, highcut, fs, domain='time'):

    if domain == 'frequency':
        signal = dc_component_removal(signal)
        signal = bandpass_filter(signal, lowcut, highcut, fs)
    
    features = extract_features(signal, domain)

    if USE_PCA_PREPROCESSING:
        pca = PCA(n_components=PREPROCESSING_PCA_COMPONENTS)
        features_reshaped = features.view(features.size(0), -1).cpu().numpy()
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_reshaped)
        features_pca = pca.fit_transform(features_normalized)
        features_tensor = torch.tensor(features_pca, dtype=torch.float32).to(signal.device)
    else:
        features_tensor = features

    return features_tensor

def preprocess_data(windowed_data, domain='time'):
    """Applies preprocessing and feature extraction."""
    preprocessed_list = []
    fs, lowcut, highcut = 10000, 50, 450 # TODO: PARAMETERS THAT NEED TO BE ADJUSTED

    if not USE_PCA_PREPROCESSING:

        for window in windowed_data:
            window_3d = window[np.newaxis, :, :]  # Convert 2D array to 3D array
            windowed_tensor = torch.tensor(window_3d, dtype=torch.float32)
            features_tensor = preprocess_and_extract(windowed_tensor, lowcut, highcut, fs, domain)
            preprocessed = features_tensor.reshape(features_tensor.shape[0], -1).cpu().numpy()
            preprocessed_list.append(preprocessed)

        preprocessed_data = np.vstack(preprocessed_list)
    else:
        preprocessed_data = preprocess_and_extract(torch.stack([torch.tensor(window, dtype=torch.float32) for window in windowed_data], dim=0), lowcut, highcut, fs, domain).cpu().numpy()

    if np.isnan(preprocessed_data).sum() > 0:
        print("Warning: NaN values detected in preprocessed data.")
    return preprocessed_data

# Step 4: Clustering
def dtw_clustering(windowed_data, n_clusters, initial_centroids=None):

    if initial_centroids is not None:
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", init=initial_centroids, n_jobs=-1) # TODO: test with n_init=1 later
    else:
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_jobs=-1)
    
    model.fit(windowed_data)
    distances = compute_dtw_distances(windowed_data, model.cluster_centers_)
    return model, distances

def compute_dtw_distances(data, centroids):
    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for i, sample in enumerate(data):
        for j, centroid in enumerate(centroids):
            distances[i, j] = dtw(sample, centroid)
    return distances

def kmeans_clustering(windowed_data, n_clusters, domain='time', initial_centroids=None):
    preprocessed_data = preprocess_data(windowed_data, domain)

    # Get the features at positions 0, 16, 43, 46, 59, 62, 63, 64, 75, 78, 86, 91, 103, 107, 123, 128, 139, 143, 156
    # preprocessed_data = preprocessed_data[:, [0, 16, 43, 46, 59, 62, 63, 64, 75, 78, 86, 91, 103, 107, 123, 128, 139, 143, 156]] # TODO: Testing features with higher correlation to mass.
    if initial_centroids is not None:
        
        # Get the third and fifth samples for the initial centroids
        init_centroids = preprocessed_data[[3, 49, 5]] # TODO: get fitted pca to apply to raw initial centroids instead of selecting from preprocessing data.

        model = KMeans(n_clusters=n_clusters, init=init_centroids).fit(preprocessed_data)
    else:
        model = KMeans(n_clusters=n_clusters).fit(preprocessed_data)

    distances = np.linalg.norm(preprocessed_data[:, None] - model.cluster_centers_, axis=2)
    return model, distances

def kmeans_with_dtw_clustering(windowed_data, n_clusters, domain='time', initial_centroids=None):

    # Get the DTW distance between each sample and the initial centroids
    dtw_distances = compute_dtw_distances(windowed_data, initial_centroids)
    
    preprocessed_data = preprocess_data(windowed_data, domain)

    # Concatenate the DTW distances to the preprocessed data
    preprocessed_data = np.concatenate((preprocessed_data, dtw_distances), axis=1)
    #preprocessed_data = dtw_distances
    if initial_centroids is not None:

        # Get the third and fifth samples for the initial centroids
        init_centroids = preprocessed_data[[3, 49, 5]] # TODO: get fitted pca to apply to raw initial centroids instead of selecting from preprocessing data.

        model = KMeans(n_clusters=n_clusters, init=init_centroids).fit(preprocessed_data)
    else:
        model = KMeans(n_clusters=n_clusters).fit(preprocessed_data)

    distances = np.linalg.norm(preprocessed_data[:, None] - model.cluster_centers_, axis=2)
    return model, distances

def sort_centroids_by_degradation(centroids):

    # Get feature values for each centroid
    features_tensor = extract_features(torch.tensor(centroids, dtype=torch.float32), domain=FEATURE_EXTRACTION_DOMAIN)
    features_reshaped = features_tensor.view(features_tensor.size(0), -1).cpu().numpy()
    
    if USE_PCA_AS_DEGRADATION:
        pca = PCA(n_components=1)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_reshaped)
        features_pca = pca.fit_transform(features_normalized)
        degradation_levels = features_pca.ravel()
    elif USE_SUM_CENTROID_FEATURES:

        # The degradation level is the sum of the features
        degradation_levels = np.sum(features_reshaped, axis=1)
    
    elif USE_AVERAGE_CENTROIDS:

        # The degradation level is the average of the averages of the centroids
        degradation_levels = np.mean(centroids, axis=(1, 2))

    result = np.argsort(degradation_levels)
    print(f"DEGRADATION LEVELS: {result}")
    return result

def normalize_and_compute_weighted_average(distances, order, prices):
    normalized_distances = distances / np.sum(distances, axis=1, keepdims=True)
    normalized_distances = 1.0 / normalized_distances
    normalized_distances /= np.sum(normalized_distances, axis=1, keepdims=True)
    normalized_distances = normalized_distances[:, order]
    return np.sum(normalized_distances * prices, axis=1)

# Step 5: Data Normalization
def normalize_data(data):
    min_values = np.min(data, axis=(0, 1), keepdims=True)
    max_values = np.max(data, axis=(0, 1), keepdims=True)
    return (data - min_values) / (max_values - min_values), min_values, max_values

def denormalize_data(data, min_values, max_values):
    return data * (max_values - min_values) + min_values

# Define plotting functions
def plot_cluster_features(sensor_names, features_per_sensor, denormalized_windowed_data, cluster_labels, filename):
    global USE_PCA_PREPROCESSING 
    prev_use_pca = USE_PCA_PREPROCESSING
    USE_PCA_PREPROCESSING = False
    min_cluster = [sample for sample, label in zip(denormalized_windowed_data, cluster_labels) if label == 0]
    mean_cluster = [sample for sample, label in zip(denormalized_windowed_data, cluster_labels) if label == 1]
    max_cluster = [sample for sample, label in zip(denormalized_windowed_data, cluster_labels) if label == 2]
    #max_cluster = [sample for sample, label in zip(denormalized_windowed_data, cluster_labels) if label == 1]
    print(len(min_cluster))
    print(len(mean_cluster))
    print(len(max_cluster))

    # Calculate the features for each cluster
    min_features = preprocess_data(min_cluster, domain=FEATURE_EXTRACTION_DOMAIN)
    mean_features = preprocess_data(mean_cluster, domain=FEATURE_EXTRACTION_DOMAIN)
    max_features = preprocess_data(max_cluster, domain=FEATURE_EXTRACTION_DOMAIN)

    # Reshape the features so they are 3D arrays of shape (n_samples, n_features=11, n_sensors)
    min_features = min_features.reshape(min_features.shape[0], 11, min_cluster[0].shape[1])
    mean_features = mean_features.reshape(mean_features.shape[0], 11, mean_cluster[0].shape[1])
    max_features = max_features.reshape(max_features.shape[0], 11, max_cluster[0].shape[1])

    print(min_features.shape)
    print(mean_features.shape)
    print(max_features.shape)

    # Example colors for clusters
    cluster_colors = ['green', 'blue', 'red']

    # Unified figure
    fig, axs = plt.subplots(4, 4, figsize=(25, 25))
    axs = axs.flatten()

    # For each sensor, create a boxplot
    for i, sensor in enumerate(sensor_names):
        ax = axs[i]

        # Prepare data for the current sensor
        all_data = []
        all_labels = []
        all_colors = []

        for c, features in enumerate([min_features, mean_features, max_features]):
            cluster_color = cluster_colors[c]
            for j, feature in enumerate(features_per_sensor):
                data = features[:, j, i]
                all_data.extend(data)
                all_labels.extend([f"{feature}"] * len(data))
                all_colors.extend([cluster_color] * len(data))

        # Convert to a format usable by seaborn boxplot
        sns.boxplot(x=all_labels, y=all_data, palette=cluster_colors, hue=all_colors, ax=ax, log_scale=True)

        # Set subplot title and adjust labels
        ax.set_title(f"{sensor}")
        ax.tick_params(axis='x', rotation=45)

    # Remove unused subplots if there are any
    for j in range(len(sensor_names), len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()

    # Save and show the unified figure
    plt.savefig(f"{filename}.png")
    #plt.show()


    # Plot another figure where each subplot has 3 histograms of the values of the sensor, one for each cluster
    fig, axs = plt.subplots(4, 4, figsize=(25, 25))
    axs = axs.flatten()

    for i, sensor in enumerate(sensor_names):
        ax = axs[i]

        for c, features in enumerate([min_cluster, mean_cluster, max_cluster]):
            cluster_color = cluster_colors[c]
            
            tmp = []
            for sample in features:
                tmp.extend(sample[:, i])

            sns.histplot(tmp, color=cluster_color, ax=ax, log_scale=True)

        ax.set_title(f"{sensor}")
        ax.legend()

    for j in range(len(sensor_names), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(f"{filename}_histograms.png")
    #plt.show()
    plt.close()

    USE_PCA_PREPROCESSING = prev_use_pca

def plot_acceleration_z_and_averages(acceleration_z, dtw_averages, kmeans_averages, kmeans_dtw_averages, window_size):
    """
    Plot the acceleration_z time series and compare weighted averages using DTW and KMeans.
    
    Args:
    - acceleration_z: np.array of shape (n_windows, window_size) containing the time series data.
    - dtw_averages: np.array of shape (n_windows,) containing DTW weighted averages for each window.
    - kmeans_averages: np.array of shape (n_windows,) containing KMeans weighted averages for each window.
    - window_size: int, the size of each window.
    """
    n_windows = acceleration_z.shape[0]
    total_length = n_windows * window_size
    time = np.arange(total_length)  # Full time axis
    
    # Create concatenated acceleration_z
    concatenated_z = np.concatenate(acceleration_z, axis=0)
    
    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot the time series
    axs[0].plot(time, concatenated_z, label="Acceleration Z", color="blue")
    for i in range(1, n_windows):  # Vertical lines to separate windows
        axs[0].axvline(i * window_size, color="black", linestyle="--", alpha=0.7)
    axs[0].set_title("Acceleration Z Time Series with Windows")
    axs[0].set_ylabel("Acceleration Z")
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot the weighted averages (DTW and KMeans)
    window_centers = np.arange(window_size // 2, total_length, window_size)
    axs[1].plot(window_centers, dtw_averages, label="DTW Weighted Average", marker="o", color="green")
    axs[1].plot(window_centers, kmeans_averages, label="KMeans Weighted Average", marker="s", color="orange")
    axs[1].plot(window_centers, kmeans_dtw_averages, label="KMeans with DTW Weighted Average", marker="x", color="red")
    for i in range(1, n_windows):  # Vertical lines to separate windows
        axs[1].axvline(i * window_size, color="black", linestyle="--", alpha=0.7)
    axs[1].set_title("Weighted Averages (DTW vs. KMeans vs. KMeans with DTW)")
    axs[1].set_ylabel("Weighted Average")
    axs[1].set_xlabel("Time")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("acceleration_z_and_averages.png")
    plt.show()


# Step 6: Main Execution
if __name__ == "__main__":
    # Real Data Loading
    path = os.path.join('data.csv.gz')
    rawest_data, target_data, id_arrays, _ = load_data(path)
    rawest_data, bad, medium, good = load_data_pastuch("ego_telemetry_log.csv")
    target_data = target_data[:8]
    id_arrays = id_arrays[:8]

    # Sensor names
    """ sensor_names = ['IMU_longitudinal', 'IMU_yaw_rate', 'IMU_lateral', 'drag',
                    'lateral_speed', 'IMU_vertical', 'gear', 'vertical_speed', 'IMU_yaw',
                    'engine_rpm', 'longitudinal_speed', 'throttle', 'steering',
                    'IMU_roll_rate', 'IMU_pitch_rate', 'IMU_pitch'] """
    
    # Sensor names for Pastuch data
    sensor_names = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x',
                    'gyro_y', 'gyro_z', 'heading', 'velocity_x', 'velocity_y', 'velocity_z',
                    'drag', 'gear', 'engine_rpm', 'throttle', 'steer', 'brake']

    # Features names
    features_per_sensor = ['min', 'max', 'mean', 'rms', 'var', 'skew', 'kurt', 'shape', 'crest', 'energy', 'entropy']

    print(target_data)

    # Identify the min, max, and mean values of the target data
    #minimum, maximum, mean = 9500, 13900, 11700
    
    # Get the id of the first sample with target value corresponding to the min, max, and mean values
    #id_min_sample = id_arrays[np.argmin(np.abs(target_data - minimum))]
    #id_max_sample = id_arrays[np.argmin(np.abs(target_data - maximum))]
    #id_mean_sample = id_arrays[np.argmin(np.abs(target_data - mean))]


    # Parameters
    window_size, n_clusters = 30, 3
    prices = np.array([2, 4, 6])
    #prices = np.array([1, 10])
    # Create windows
    windowed_data, _, _ = create_windows(rawest_data, id_arrays, target_data, window_size)
    windowed_data, min_values, max_values = normalize_data(windowed_data)

    print(windowed_data.shape)
    print(target_data.shape)

    """ raw_data = rawest_data.copy()
    # Preprocess the raw data
    for i, data in enumerate(rawest_data):
        # Normalize the data using the min and max values
        data = (data - min_values) / (max_values - min_values)
        # Drop the first dimension of the data
        data = data[0]
        raw_data[i] = data.tolist()
    raw_data = to_time_series_dataset(raw_data) """

    

    #print(f"RAW DATA: {raw_data.shape}")




     # Get the centroids for the min, max, and mean samples
    #min_centroid = rawest_data[np.where(id_arrays == id_min_sample)[0][0]]
    #mean_centroid = rawest_data[np.where(id_arrays == id_mean_sample)[0][0]]
    #max_centroid = rawest_data[np.where(id_arrays == id_max_sample)[0][0]]
    min_centroid = good
    mean_centroid = medium
    max_centroid = bad

    # Normalize the centroids
    min_centroid = (min_centroid - min_values) / (max_values - min_values)
    min_centroid = min_centroid[0]
    mean_centroid = (mean_centroid - min_values) / (max_values - min_values)
    mean_centroid = mean_centroid[0]
    max_centroid = (max_centroid - min_values) / (max_values - min_values)
    max_centroid = max_centroid[0]

    # Make them have the same shape by padding np.nan values at the end
    """ max_len = raw_data.shape[1]
    min_centroid = np.pad(min_centroid, ((0, max_len - min_centroid.shape[0]), (0, 0)))
    mean_centroid = np.pad(mean_centroid, ((0, max_len - mean_centroid.shape[0]), (0, 0)))
    max_centroid = np.pad(max_centroid, ((0, max_len - max_centroid.shape[0]), (0, 0))) """

    centroids = np.array([min_centroid, mean_centroid, max_centroid])
    #centroids = np.array([min_centroid, max_centroid])
    
    #centroids = centroids[:, :100, :]
    print(f"CENTROIDS: {centroids.shape}")

    #raw_data = raw_data[:, :100, :]

    #print(id_arrays)

    # DTW Clustering
    dtw_model, dtw_distances = dtw_clustering(windowed_data, n_clusters, initial_centroids=centroids)
    #dtw_sorted_indices = sort_centroids_by_degradation(dtw_model.cluster_centers_)
    dtw_sorted_indices = np.array([0, 1, 2])
    #dtw_sorted_indices = np.array([0, 1])
    dtw_weighted_averages = normalize_and_compute_weighted_average(dtw_distances, dtw_sorted_indices, prices)

    
    # Separate the samples from each cluster by using the labels
    cluster_labels = dtw_model.labels_
    denormalized_windowed_data = denormalize_data(windowed_data, min_values, max_values)
    #denormalized_windowed_data = rawest_data
    
    plot_cluster_features(sensor_names, features_per_sensor, denormalized_windowed_data, cluster_labels, "unified_cluster_features_dtw")

    # KMeans Clustering
    kmeans_model, kmeans_distances = kmeans_clustering(windowed_data, n_clusters, domain=FEATURE_EXTRACTION_DOMAIN, initial_centroids=centroids)
    #kmeans_sorted_indices = sort_centroids_by_degradation(kmeans_model.cluster_centers_.reshape(n_clusters, -1, PREPROCESSING_PCA_COMPONENTS))
    kmeans_sorted_indices = np.array([2, 1, 0])
    #kmeans_sorted_indices = np.array([0, 1])
    kmeans_weighted_averages = normalize_and_compute_weighted_average(kmeans_distances, kmeans_sorted_indices, prices)

    # Separate the samples from each cluster by using the labels
    cluster_labels = kmeans_model.labels_
    denormalized_windowed_data = denormalize_data(windowed_data, min_values, max_values)
    #denormalized_windowed_data = rawest_data
    
    plot_cluster_features(sensor_names, features_per_sensor, denormalized_windowed_data, cluster_labels, "unified_cluster_features_kmeans")


    # KMeans with DTW Clustering
    kmeans_dtw_model, kmeans_dtw_distances = kmeans_with_dtw_clustering(windowed_data, n_clusters, domain=FEATURE_EXTRACTION_DOMAIN, initial_centroids=centroids)
    #kmeans_dtw_sorted_indices = sort_centroids_by_degradation(kmeans_dtw_model.cluster_centers_.reshape(n_clusters, -1, PREPROCESSING_PCA_COMPONENTS))
    kmeans_dtw_sorted_indices = np.array([2, 1, 0])
    #kmeans_dtw_sorted_indices = np.array([0, 1])
    kmeans_dtw_weighted_averages = normalize_and_compute_weighted_average(kmeans_dtw_distances, kmeans_dtw_sorted_indices, prices)

    # Separate the samples from each cluster by using the labels
    cluster_labels = kmeans_dtw_model.labels_
    denormalized_windowed_data = denormalize_data(windowed_data, min_values, max_values)
    #denormalized_windowed_data = rawest_data

    plot_cluster_features(sensor_names, features_per_sensor, denormalized_windowed_data, cluster_labels, "unified_cluster_features_kmeans_dtw")

    # Output results
    with open("output_results.txt", "w") as f:
        f.write(f"Results for {len(id_arrays)} windows of size {window_size}:\n")
        f.write(f"Pricing: {prices}\n")
        f.write(f"Parameters: \nUse PCA Preprocessing: {USE_PCA_PREPROCESSING}\nUse PCA as Degradation: {USE_PCA_AS_DEGRADATION}\nUse Sum Centroid Features: {USE_SUM_CENTROID_FEATURES}\nUse Average Centroids: {USE_AVERAGE_CENTROIDS}\nPreprocessing PCA Components: {PREPROCESSING_PCA_COMPONENTS}\nFeature Extraction Domain: {FEATURE_EXTRACTION_DOMAIN}\n")
        f.write(f"DTW Weighted Averages with {dtw_model.inertia_} model inertia:\n")
        f.write(f"{dtw_weighted_averages}\n")
        f.write(f"KMeans Weighted Averages with {kmeans_model.inertia_} model inertia:\n")
        f.write(f"{kmeans_weighted_averages}\n")
        f.write(f"KMeans with DTW Weighted Averages with {kmeans_dtw_model.inertia_} model inertia:\n")
        f.write(f"{kmeans_dtw_weighted_averages}\n")


    # Plot the acceleration_z feature for each window, concatenated side by side, and below the dtw and kmeans weighted averages for the windows
    
    plot_acceleration_z_and_averages(windowed_data[:, :, 2], dtw_weighted_averages, kmeans_weighted_averages, kmeans_dtw_weighted_averages, window_size)