#!/usr/bin/env python3
"""
Demonstration script for the Recurrent Autoencoder Anomaly Detection System

This script shows how to:
1. Train autoencoders for vehicle groups
2. Generate enhanced diagnostic plots with reconstruction errors
3. Analyze anomaly detection results

Usage Examples:
    # Train autoencoder for a specific group
    python run_anomaly_detection_demo.py --train --group group_1
    
    # Train autoencoders for all groups  
    python run_anomaly_detection_demo.py --train --all
    
    # Generate enhanced diagnostic plot for a simulation
    python run_anomaly_detection_demo.py --analyze --sim_id 9
    
    # Do both training and analysis
    python run_anomaly_detection_demo.py --train --group group_1 --analyze --sim_id 9
"""

import subprocess
import argparse
import os
import sys

def run_command(command: list, description: str):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        print("Make sure you're in the correct directory and the script exists.")
        return False

def train_autoencoder(group_name: str | None = None, train_all: bool = False):
    """Train autoencoder models."""
    if train_all:
        command = ["python", "recurrent_autoencoder_anomaly_detection.py", "--all"]
        description = "Training autoencoders for all vehicle groups"
    elif group_name is not None:
        command = ["python", "recurrent_autoencoder_anomaly_detection.py", "--group", group_name]
        description = f"Training autoencoder for {group_name}"
    else:
        print("Error: Must specify either --group GROUP_NAME or --all for training")
        return False
    
    return run_command(command, description)

def analyze_simulation(sim_id: int):
    """Generate enhanced diagnostic plots for a simulation."""
    command = ["python", "enhanced_diagnostic_plot.py", "--sim_id", str(sim_id)]
    description = f"Generating enhanced diagnostic plots for simulation {sim_id}"
    
    return run_command(command, description)

def check_requirements():
    """Check if required files exist."""
    required_files = [
        "recurrent_autoencoder_anomaly_detection.py",
        "enhanced_diagnostic_plot.py",
        "processed_data",
        "controle.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files/directories:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure you have:")
        print("1. The autoencoder training script")
        print("2. The enhanced diagnostic plot script") 
        print("3. Processed CSV data in 'processed_data/' directory")
        print("4. The controle.txt file with vehicle group definitions")
        return False
    
    print("✅ All required files found!")
    return True

def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Train autoencoder for group_1", 
         "python run_anomaly_detection_demo.py --train --group group_1"),
        
        ("Train autoencoders for all groups", 
         "python run_anomaly_detection_demo.py --train --all"),
        
        ("Analyze simulation 9 (requires trained model for its group)", 
         "python run_anomaly_detection_demo.py --analyze --sim_id 9"),
        
        ("Train group_1 and then analyze simulation 9", 
         "python run_anomaly_detection_demo.py --train --group group_1 --analyze --sim_id 9"),
        
        ("Check which group a simulation belongs to",
         "# Simulation ranges from controle.txt:\n"
         "# group_1: 1-20, group_2: 21-40, group_3: 41-60, etc."),
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        print(f"\n{i}. {description}:")
        print(f"   {command}")
    
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description="Demonstration script for Recurrent Autoencoder Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Training options
    training_group = parser.add_mutually_exclusive_group()
    training_group.add_argument("--group", type=str, 
                               help="Train autoencoder for specific group (e.g., 'group_1')")
    training_group.add_argument("--all", action="store_true", 
                               help="Train autoencoders for all groups")
    
    # Analysis options
    parser.add_argument("--sim_id", type=int, 
                       help="Simulation ID to analyze (requires trained model for its group)")
    
    # Action flags
    parser.add_argument("--train", action="store_true", 
                       help="Train autoencoder models")
    parser.add_argument("--analyze", action="store_true", 
                       help="Generate enhanced diagnostic plots")
    
    # Utility options
    parser.add_argument("--check", action="store_true", 
                       help="Check if required files exist")
    parser.add_argument("--examples", action="store_true", 
                       help="Show usage examples")
    
    args = parser.parse_args()
    
    # Handle utility options
    if args.examples:
        print_usage_examples()
        return
    
    if args.check:
        check_requirements()
        return
    
    # Check if no action specified
    if not args.train and not args.analyze:
        print("No action specified. Use --train and/or --analyze, or --examples for help.")
        parser.print_help()
        return
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot proceed due to missing requirements.")
        return
    
    success = True
    
    # Training phase
    if args.train:
        if not args.group and not args.all:
            print("Error: --train requires either --group GROUP_NAME or --all")
            return
        
        success = train_autoencoder(args.group, args.all)
        
        if not success:
            print("\n❌ Training failed. Cannot proceed to analysis.")
            return
    
    # Analysis phase
    if args.analyze:
        if not args.sim_id:
            print("Error: --analyze requires --sim_id SIM_ID")
            return
        
        success = analyze_simulation(args.sim_id)
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All operations completed successfully!")
        
        if args.analyze:
            print(f"\nResults saved in:")
            print(f"  - Models: autoencoder_models/")
            print(f"  - Training results: autoencoder_results/") 
            print(f"  - Enhanced plots: enhanced_diagnostics/")
    else:
        print("❌ Some operations failed. Check the output above for details.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 