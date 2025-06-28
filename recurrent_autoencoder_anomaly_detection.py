import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_data"
MODELS_DIR = "autoencoder_models"
RESULTS_DIR = "autoencoder_results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
WINDOW_SIZE = 30  # 3 seconds at 10Hz
SENSORS_FOR_AUTOENCODER = [
    'IMU_ACC_Z_DYNAMIC', 'IMU_ACC_X', 'IMU_ACC_Y', 
    'ENGINE_RPM', 'SPEED', 'THROTTLE'
]

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Vehicle groups from controle.txt
VEHICLE_GROUPS = {
    'group_1': {'seq_range': range(1, 21), 'mass': 8300, 'friction': 1.0},
    'group_2': {'seq_range': range(21, 41), 'mass': 10900, 'friction': 1.0},
    'group_3': {'seq_range': range(41, 61), 'mass': 13500, 'friction': 1.0},
    'group_4': {'seq_range': range(61, 81), 'mass': 13500, 'friction': 1.0},
    'group_5': {'seq_range': range(81, 101), 'mass': 10900, 'friction': 1.0},
    'group_6': {'seq_range': range(101, 121), 'mass': 8300, 'friction': 0.75},
    'group_7': {'seq_range': range(121, 141), 'mass': 10900, 'friction': 0.75},
    'group_8': {'seq_range': range(141, 161), 'mass': 13500, 'friction': 0.75},
    'group_9': {'seq_range': range(161, 181), 'mass': 13500, 'friction': 0.5},
    'group_10': {'seq_range': range(181, 201), 'mass': 10900, 'friction': 0.5},
    'group_11': {'seq_range': range(201, 221), 'mass': 8300, 'friction': 0.5},
}

def find_sim_file(sim_id: int) -> str:
    """Finds the CSV file for a given simulation ID."""
    pattern = os.path.join(PROCESSED_DATA_DIR, f"simulation_{sim_id}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No simulation file found for ID {sim_id}")
    return files[0]

def process_data_for_autoencoder(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as diagnostic_plot.py"""
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

def find_continuous_road_sequences(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find continuous sequences of 'road' context."""
    road_mask = (df['CONTEXT'] == 'road').values
    sequences = []
    
    if len(road_mask) == 0:
        return sequences
    
    # Find transitions
    diff = np.diff(np.concatenate([[False], road_mask, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if end - start >= WINDOW_SIZE:  # Only keep sequences >= 30 timesteps
            sequences.append((start, end))
    
    return sequences

def create_sliding_windows(sequence_data: np.ndarray, window_size: int) -> np.ndarray:
    """Create sliding windows with step=1 from a sequence."""
    if len(sequence_data) < window_size:
        return np.array([])
    
    windows = []
    for i in range(len(sequence_data) - window_size + 1):
        windows.append(sequence_data[i:i + window_size])
    
    return np.array(windows)

def load_group_data(group_name: str, group_params: Dict) -> Tuple[np.ndarray, MinMaxScaler | None]:
    """Load and preprocess data for a vehicle group."""
    print(f"\n--- Loading data for {group_name} ---")
    print(f"Mass: {group_params['mass']}, Friction: {group_params['friction']}")
    print(f"Simulation range: {list(group_params['seq_range'])}")
    
    all_windows = []
    files_processed = 0
    
    for sim_id in group_params['seq_range']:
        try:
            file_path = find_sim_file(sim_id)
            df = pd.read_csv(file_path)
            
            # Apply preprocessing (same as diagnostic_plot.py)
            df = process_data_for_autoencoder(df)
            
            # Check if required sensors are available
            available_sensors = [sensor for sensor in SENSORS_FOR_AUTOENCODER if sensor in df.columns]
            missing_sensors = [sensor for sensor in SENSORS_FOR_AUTOENCODER if sensor not in df.columns]
            
            if missing_sensors:
                print(f"  Sim {sim_id}: Missing sensors {missing_sensors}, skipping...")
                continue
            
            if len(available_sensors) < len(SENSORS_FOR_AUTOENCODER):
                print(f"  Sim {sim_id}: Only {len(available_sensors)}/{len(SENSORS_FOR_AUTOENCODER)} sensors available, skipping...")
                continue
            
            # Find continuous road sequences
            road_sequences = find_continuous_road_sequences(df)
            
            if not road_sequences:
                print(f"  Sim {sim_id}: No road sequences >= {WINDOW_SIZE} timesteps found")
                continue
            
            print(f"  Sim {sim_id}: Found {len(road_sequences)} road sequences")
            
            # Extract windows from each road sequence
            sim_windows = []
            for start_idx, end_idx in road_sequences:
                sequence_data = df.iloc[start_idx:end_idx][SENSORS_FOR_AUTOENCODER].values
                windows = create_sliding_windows(sequence_data, WINDOW_SIZE)
                if len(windows) > 0:
                    sim_windows.append(windows)
            
            if sim_windows:
                sim_windows = np.vstack(sim_windows)
                all_windows.append(sim_windows)
                print(f"  Sim {sim_id}: Generated {len(sim_windows)} windows")
                files_processed += 1
            
        except FileNotFoundError:
            print(f"  Sim {sim_id}: File not found, skipping...")
            continue
        except Exception as e:
            print(f"  Sim {sim_id}: Error processing file: {e}")
            continue
    
    if not all_windows:
        print(f"No valid data found for {group_name}")
        return np.array([]), None
    
    # Combine all windows
    all_windows = np.vstack(all_windows)
    print(f"Total windows for {group_name}: {len(all_windows)}")
    print(f"Files processed: {files_processed}")
    
    # Normalize data (fit scaler on all windows for this group)
    print("Fitting MinMax scaler on all windows...")
    original_shape = all_windows.shape
    windows_reshaped = all_windows.reshape(-1, all_windows.shape[-1])
    
    scaler = MinMaxScaler()
    windows_normalized = scaler.fit_transform(windows_reshaped)
    all_windows_normalized = windows_normalized.reshape(original_shape)
    
    print(f"Data shape: {all_windows_normalized.shape}")
    print(f"Data range after normalization: [{all_windows_normalized.min():.3f}, {all_windows_normalized.max():.3f}]")
    
    return all_windows_normalized, scaler

class VehicleDataset(Dataset):
    """Dataset for vehicle sensor data windows."""
    
    def __init__(self, windows: np.ndarray):
        self.windows = torch.FloatTensor(windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # For autoencoder, input and target are the same
        return self.windows[idx], self.windows[idx]

class VehicleAutoencoder(nn.Module):
    """LSTM-based Autoencoder for vehicle sensor data."""
    
    def __init__(self, input_size=6, hidden_sizes=[128, 64, 32], dropout=0.2):
        super(VehicleAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = hidden_sizes[-1]
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, dropout=dropout)
        self.encoder_lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True, dropout=dropout)
        self.encoder_lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        # Decoder
        self.decoder_lstm1 = nn.LSTM(hidden_sizes[2], hidden_sizes[1], batch_first=True, dropout=dropout)
        self.decoder_lstm2 = nn.LSTM(hidden_sizes[1], hidden_sizes[0], batch_first=True, dropout=dropout)
        self.decoder_lstm3 = nn.LSTM(hidden_sizes[0], input_size, batch_first=True)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Encoder
        encoded, _ = self.encoder_lstm1(x)
        encoded, _ = self.encoder_lstm2(encoded)
        encoded, _ = self.encoder_lstm3(encoded)
        
        # Decoder
        decoded, _ = self.decoder_lstm1(encoded)
        decoded, _ = self.decoder_lstm2(decoded)
        decoded, _ = self.decoder_lstm3(decoded)
        
        return decoded

def train_autoencoder(train_loader: DataLoader, val_loader: DataLoader, 
                     input_size: int, group_name: str) -> VehicleAutoencoder:
    """Train the autoencoder model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = VehicleAutoencoder(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining autoencoder for {group_name}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{group_name}_best_model.pth"))
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{group_name}_best_model.pth")))
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Curves - {group_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{group_name}_training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Best validation loss: {best_val_loss:.6f}")
    return model

def calculate_reconstruction_errors(model: VehicleAutoencoder, data_loader: DataLoader, 
                                  sensor_names: List[str]) -> Dict[str, np.ndarray]:
    """Calculate reconstruction errors for each sensor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_errors = {f"{sensor}_reconstruction_error": [] for sensor in sensor_names}
    
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            # Calculate MAE for each sensor across the sequence
            mae_per_sensor = torch.mean(torch.abs(outputs - batch_x), dim=1)  # [batch, sensors]
            
            for i, sensor in enumerate(sensor_names):
                all_errors[f"{sensor}_reconstruction_error"].extend(mae_per_sensor[:, i].cpu().numpy())
    
    # Convert lists to numpy arrays
    for key in all_errors:
        all_errors[key] = np.array(all_errors[key])
    
    return all_errors

def main():
    parser = argparse.ArgumentParser(description="Train recurrent autoencoders for vehicle anomaly detection")
    parser.add_argument("--group", type=str, help="Specific group to train (e.g., 'group_1')")
    parser.add_argument("--all", action="store_true", help="Train models for all groups")
    args = parser.parse_args()
    
    if not args.group and not args.all:
        print("Please specify either --group GROUP_NAME or --all")
        return
    
    groups_to_process = {}
    if args.all:
        groups_to_process = VEHICLE_GROUPS
    else:
        if args.group in VEHICLE_GROUPS:
            groups_to_process[args.group] = VEHICLE_GROUPS[args.group]
        else:
            print(f"Unknown group: {args.group}")
            print(f"Available groups: {list(VEHICLE_GROUPS.keys())}")
            return
    
    print(f"Training autoencoders for {len(groups_to_process)} group(s)")
    print(f"Window size: {WINDOW_SIZE} timesteps")
    print(f"Target sensors: {SENSORS_FOR_AUTOENCODER}")
    
    for group_name, group_params in groups_to_process.items():
        try:
            print(f"\n{'='*60}")
            print(f"Processing {group_name}")
            print(f"{'='*60}")
            
            # Load and preprocess data
            windows, scaler = load_group_data(group_name, group_params)
            
            if len(windows) == 0:
                print(f"No data available for {group_name}, skipping...")
                continue
            
            if len(windows) < 100:
                print(f"Insufficient data for {group_name} ({len(windows)} windows), skipping...")
                continue
            
            # Split data
            print(f"\nSplitting data: train={TRAIN_SPLIT}, val={VAL_SPLIT}, test={TEST_SPLIT}")
            train_windows, temp_windows = train_test_split(windows, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42)
            val_windows, test_windows = train_test_split(temp_windows, test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT), random_state=42)
            
            print(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
            
            # Create datasets and dataloaders
            train_dataset = VehicleDataset(np.array(train_windows))
            val_dataset = VehicleDataset(np.array(val_windows))
            test_dataset = VehicleDataset(np.array(test_windows))
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # Train model
            model = train_autoencoder(train_loader, val_loader, len(SENSORS_FOR_AUTOENCODER), group_name)
            
            # Calculate reconstruction errors on test set
            print("\nCalculating reconstruction errors on test set...")
            test_errors = calculate_reconstruction_errors(model, test_loader, SENSORS_FOR_AUTOENCODER)
            
            # Save results
            results = {
                'group_name': group_name,
                'group_params': group_params,
                'num_windows': len(windows),
                'train_size': len(train_windows),
                'val_size': len(val_windows),
                'test_size': len(test_windows),
                'scaler': scaler,
                'test_errors': test_errors,
                'sensor_names': SENSORS_FOR_AUTOENCODER
            }
            
            with open(os.path.join(RESULTS_DIR, f"{group_name}_results.pkl"), 'wb') as f:
                pickle.dump(results, f)
            
            # Print summary statistics
            print(f"\nReconstruction Error Summary for {group_name}:")
            for sensor in SENSORS_FOR_AUTOENCODER:
                error_key = f"{sensor}_reconstruction_error"
                errors = test_errors[error_key]
                print(f"  {sensor}:")
                print(f"    Mean: {np.mean(errors):.6f}")
                print(f"    Std:  {np.std(errors):.6f}")
                print(f"    Min:  {np.min(errors):.6f}")
                print(f"    Max:  {np.max(errors):.6f}")
            
            print(f"\nModel and results saved for {group_name}")
            
        except Exception as e:
            print(f"Error processing {group_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main() 