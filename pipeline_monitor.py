#!/usr/bin/env python3
"""
Comprehensive Pipeline Monitor for Vehicle Anomaly Detection Training
=====================================================================

This script monitors the autoencoder training process (PID 27556) and then
executes the complete analysis pipeline including enhanced visualizations
for all vehicle groups and experiment2.py.

Author: AI Assistant
Date: 2024
"""

import psutil
import subprocess
import time
import sys
import os
from datetime import datetime
from typing import List, Tuple, Dict
import logging

# === Configuration ===
TRAINING_PID = 27556
LOG_FILE = f"pipeline_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Vehicle groups and representative simulations for visualization
VEHICLE_GROUPS = {
    'group_1': {'sim_range': range(1, 21), 'representative_sims': [1, 5, 10, 15, 20]},
    'group_2': {'sim_range': range(21, 41), 'representative_sims': [21, 25, 30, 35, 40]},
    'group_3': {'sim_range': range(41, 61), 'representative_sims': [41, 45, 50, 55, 60]},
    'group_4': {'sim_range': range(61, 81), 'representative_sims': [61, 65, 70, 75, 80]},
    'group_5': {'sim_range': range(81, 101), 'representative_sims': [81, 85, 90, 95, 100]},
    'group_6': {'sim_range': range(101, 121), 'representative_sims': [101, 105, 110, 115, 120]},
    'group_7': {'sim_range': range(121, 141), 'representative_sims': [121, 125, 130, 135, 140]},
    'group_8': {'sim_range': range(141, 161), 'representative_sims': [141, 145, 150, 155, 160]},
    'group_9': {'sim_range': range(161, 181), 'representative_sims': [161, 165, 170, 175, 180]},
    'group_10': {'sim_range': range(181, 201), 'representative_sims': [181, 185, 190, 195, 200]},
    'group_11': {'sim_range': range(201, 221), 'representative_sims': [201, 205, 210, 215, 220]},
}

# Additional interesting simulations for comprehensive analysis
SPECIAL_ANALYSIS_SIMS = [9, 50, 100, 150, 200]  # Cross-group interesting cases

def setup_logging():
    """Set up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def log_system_info(logger):
    """Log system information for debugging."""
    logger.info("=" * 80)
    logger.info("🚀 VEHICLE ANOMALY DETECTION PIPELINE MONITOR")
    logger.info("=" * 80)
    logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"💻 Python version: {sys.version}")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    logger.info(f"🔍 Monitoring PID: {TRAINING_PID}")
    logger.info(f"📄 Log file: {LOG_FILE}")
    logger.info("=" * 80)

def monitor_training_process(pid: int, logger) -> bool:
    """
    Monitor the training process until completion.
    
    Args:
        pid: Process ID to monitor
        logger: Logger instance
    
    Returns:
        True if process completed normally, False if process not found
    """
    try:
        process = psutil.Process(pid)
        logger.info(f"🔍 Found training process: {process.name()}")
        logger.info(f"📋 Command: {' '.join(process.cmdline())}")
        logger.info(f"💾 Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        start_time = time.time()
        check_count = 0
        
        while process.is_running():
            check_count += 1
            elapsed_time = time.time() - start_time
            
            # Get current memory and CPU usage
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                logger.info(f"⏱️  Training running... ({elapsed_time/60:.1f} min elapsed)")
                logger.info(f"   💾 Memory: {memory_mb:.1f} MB | 🖥️  CPU: {cpu_percent:.1f}%")
                
                # Check every 5 minutes
                time.sleep(300)
                
                # Log progress every hour
                if check_count % 12 == 0:
                    logger.info(f"🕐 Hourly update: Training has been running for {elapsed_time/3600:.1f} hours")
                    
            except psutil.NoSuchProcess:
                break
                
        total_time = time.time() - start_time
        logger.info(f"✅ Training process completed! Total time: {total_time/3600:.1f} hours")
        return True
        
    except psutil.NoSuchProcess:
        logger.warning(f"❌ Process {pid} not found! It may have already completed.")
        return False
    except Exception as e:
        logger.error(f"❌ Error monitoring process: {e}")
        return False

def run_command(command: str, description: str, logger, timeout: int = 3600) -> Tuple[bool, str]:
    """
    Run a command with comprehensive logging and error handling.
    
    Args:
        command: Command to execute
        description: Human-readable description
        logger: Logger instance
        timeout: Timeout in seconds (default 1 hour)
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🔧 Starting: {description}")
    logger.info(f"💻 Command: {command}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        logger.info(f"✅ {description} completed successfully in {execution_time:.1f}s")
        
        # Log last 1000 characters of output for debugging
        if result.stdout:
            output_preview = result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout
            logger.info(f"📤 Output preview:\n{output_preview}")
        
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} timed out after {timeout}s")
        return False, f"Timeout after {timeout}s"
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ {description} failed after {execution_time:.1f}s")
        logger.error(f"🔴 Return code: {e.returncode}")
        if e.stderr:
            logger.error(f"🔴 Error output:\n{e.stderr}")
        return False, e.stderr or f"Command failed with return code {e.returncode}"
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Unexpected error in {description} after {execution_time:.1f}s: {e}")
        return False, str(e)

def generate_enhanced_visualizations(logger) -> bool:
    """
    Generate enhanced visualizations for representative simulations from all groups.
    
    Returns:
        True if all visualizations generated successfully
    """
    logger.info(f"\n{'🎨'*20}")
    logger.info("🎨 STARTING ENHANCED VISUALIZATION GENERATION")
    logger.info(f"{'🎨'*20}")
    
    # Collect all simulation IDs to process
    all_sims_to_process = []
    
    # Add representative simulations from each group
    for group_name, group_info in VEHICLE_GROUPS.items():
        all_sims_to_process.extend(group_info['representative_sims'])
        logger.info(f"📊 {group_name}: Will visualize simulations {group_info['representative_sims']}")
    
    # Add special analysis simulations
    all_sims_to_process.extend(SPECIAL_ANALYSIS_SIMS)
    
    # Remove duplicates and sort
    all_sims_to_process = sorted(list(set(all_sims_to_process)))
    
    logger.info(f"📈 Total simulations to visualize: {len(all_sims_to_process)}")
    logger.info(f"🎯 Simulation IDs: {all_sims_to_process}")
    
    successful_visualizations = 0
    failed_visualizations = []
    
    for sim_id in all_sims_to_process:
        command = f"python enhanced_diagnostic_plot.py --sim_id {sim_id}"
        description = f"Enhanced visualization for simulation {sim_id}"
        
        success, output = run_command(command, description, logger, timeout=1800)  # 30 min timeout
        
        if success:
            successful_visualizations += 1
            logger.info(f"✅ Visualization {sim_id} completed ({successful_visualizations}/{len(all_sims_to_process)})")
        else:
            failed_visualizations.append(sim_id)
            logger.error(f"❌ Visualization {sim_id} failed")
    
    # Summary
    logger.info(f"\n{'📊'*20}")
    logger.info("📊 VISUALIZATION GENERATION SUMMARY")
    logger.info(f"{'📊'*20}")
    logger.info(f"✅ Successful: {successful_visualizations}/{len(all_sims_to_process)}")
    logger.info(f"❌ Failed: {len(failed_visualizations)}")
    
    if failed_visualizations:
        logger.warning(f"⚠️  Failed simulation IDs: {failed_visualizations}")
    
    # Return True if at least 80% succeeded
    success_rate = successful_visualizations / len(all_sims_to_process)
    if success_rate >= 0.8:
        logger.info(f"🎉 Visualization phase completed successfully! Success rate: {success_rate:.1%}")
        return True
    else:
        logger.error(f"💥 Visualization phase had too many failures. Success rate: {success_rate:.1%}")
        return False

def run_experiment2(logger) -> bool:
    """
    Run experiment2.py for all groups.
    
    Returns:
        True if experiment completed successfully
    """
    logger.info(f"\n{'🧪'*20}")
    logger.info("🧪 STARTING EXPERIMENT 2: Enhanced Billing Validation")
    logger.info(f"{'🧪'*20}")
    
    command = "python experiment2.py"
    description = "Experiment 2 - Enhanced Billing Validation using Reconstruction Error Signals"
    
    # Experiment2 might take a long time, so set a 4-hour timeout
    success, output = run_command(command, description, logger, timeout=14400)
    
    if success:
        logger.info("🎉 Experiment 2 completed successfully!")
        return True
    else:
        logger.error("💥 Experiment 2 failed!")
        return False

def generate_final_summary(logger, start_time: float, results: Dict[str, bool]):
    """Generate a comprehensive final summary."""
    total_time = time.time() - start_time
    
    logger.info(f"\n{'🏁'*30}")
    logger.info("🏁 PIPELINE EXECUTION SUMMARY")
    logger.info(f"{'🏁'*30}")
    logger.info(f"📅 Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏱️  Total execution time: {total_time/3600:.1f} hours")
    
    logger.info(f"\n📋 RESULTS:")
    for task, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"   {task}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        logger.info(f"\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        logger.info(f"🎯 You can now analyze the results in:")
        logger.info(f"   📊 enhanced_diagnostics/ - Enhanced visualizations")
        logger.info(f"   📈 results_experiment2/ - Experiment 2 results")
    else:
        failed_tasks = [task for task, success in results.items() if not success]
        logger.warning(f"\n⚠️  SOME TASKS FAILED: {failed_tasks}")
    
    logger.info(f"\n📄 Complete log saved to: {LOG_FILE}")
    logger.info(f"{'🏁'*30}")

def main():
    """Main pipeline execution function."""
    # Setup
    logger = setup_logging()
    start_time = time.time()
    log_system_info(logger)
    
    # Track results
    results = {}
    
    try:
        # Phase 1: Monitor training process
        logger.info("🔍 Phase 1: Monitoring autoencoder training...")
        training_success = monitor_training_process(TRAINING_PID, logger)
        results["Training Monitoring"] = training_success
        
        if not training_success:
            logger.warning("⚠️  Training monitoring failed, but continuing with pipeline...")
        
        # Phase 2: Generate enhanced visualizations
        logger.info("🎨 Phase 2: Generating enhanced visualizations...")
        viz_success = generate_enhanced_visualizations(logger)
        results["Enhanced Visualizations"] = viz_success
        
        if not viz_success:
            logger.error("💥 Visualization phase failed. Stopping pipeline.")
            return
        
        # Phase 3: Run experiment2
        logger.info("🧪 Phase 3: Running experiment 2...")
        exp2_success = run_experiment2(logger)
        results["Experiment 2"] = exp2_success
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        results["Pipeline Status"] = False
    except Exception as e:
        logger.error(f"\n💥 Unexpected error in pipeline: {e}")
        results["Pipeline Status"] = False
    finally:
        # Generate final summary
        generate_final_summary(logger, start_time, results)

if __name__ == "__main__":
    main() 