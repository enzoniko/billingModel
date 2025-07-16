import os
import subprocess
import json
from sklearn.model_selection import ParameterGrid
import hashlib
import argparse

def generate_commands():
    """
    Generates a list of commands for a grid search, but does not execute them.
    """
    parser = argparse.ArgumentParser(description="Generate commands for a hyperparameter grid search for experiment5.py.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate commands for a fast test mode."
    )
    args = parser.parse_args()

    # --- Define the Hyperparameter Search Space ---
    if args.test:
        print("--- Generating commands for TEST mode ---")
        param_grid = {
            'post_process_gamma': [0.5, 0.7],
            'post_process_dead_zone_k': [3.0],
            'dim_redux_target_variance': [0.95],
        }
    else:
        print("--- Generating commands for FULL grid search ---")
        param_grid = {
            'post_process_gamma': [0.5, 0.6, 0.7],
            'post_process_dead_zone_k': [2.5, 3.0, 3.5],
            'pricing_strategy_threshold_distance': [0.7, 0.8, 0.9],
            'dim_redux_target_variance': [0.90, 0.95],
            'hybrid_alpha': [0.2, 0.3, 0.4]
        }

    grid = ParameterGrid(param_grid)
    commands = []
    
    base_dir_name = "grid_search_results_test" if args.test else "grid_search_results"
    os.makedirs(base_dir_name, exist_ok=True)

    for i, params in enumerate(grid):
        params_str = json.dumps(params, sort_keys=True)
        run_hash = hashlib.md5(params_str.encode()).hexdigest()[:10]
        run_name = f"run_{i:03d}_{run_hash}"
        run_results_dir = os.path.join(base_dir_name, run_name)

        # Save parameters file ahead of time
        os.makedirs(run_results_dir, exist_ok=True)
        with open(os.path.join(run_results_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        command_parts = ['python', 'experiment5.py', '--results-dir', run_results_dir]
        for key, value in params.items():
            command_parts.append(f"--{key.replace('_', '-')}")
            command_parts.append(str(value))

        # Force CPU for all parallel runs to avoid CUDA memory issues
        command_parts.extend(['--device', 'cpu'])

        if args.test:
            command_parts.extend(['--group', 'group_1', '--limit-files', '1'])
        
        commands.append(' '.join(command_parts))

    # --- Write commands to a batch file for execution ---
    num_workers = 20 # Define how many to run in parallel at a time
    batch_file_name = "run_grid_search.bat"
    
    with open(batch_file_name, 'w') as f:
        f.write("@echo off\n")
        f.write("SETLOCAL EnableDelayedExpansion\n")
        f.write("echo Starting parallel grid search...\n")
        
        for i in range(0, len(commands), num_workers):
            batch_num = i // num_workers + 1
            f.write(f"echo --- Running batch {batch_num} --- \n")
            
            procs_info = []
            for j in range(num_workers):
                if i + j < len(commands):
                    cmd = commands[i+j]
                    
                    # Extract run_results_dir from the command string
                    run_results_dir = cmd.split('--results-dir')[1].strip().split(' ')[0]
                    
                    proc_title = f"RUN_{i+j}"
                    f.write(f'START "{proc_title}" {cmd}\n')
                    procs_info.append({'title': proc_title, 'dir': run_results_dir})
            
            # Wait for this batch of processes to finish by checking for the output file
            if procs_info:
                f.write("echo Waiting for batch to complete...\n")
                wait_label = f"wait_loop_batch_{batch_num}"
                f.write(f":{wait_label}\n")
                f.write('set "completed_procs=0"\n')
                
                for info in procs_info:
                    summary_file_path = os.path.join(info['dir'], 'summary_evaluation_by_group_experiment5.csv')
                    f.write(f'IF EXIST "{summary_file_path}" (set /a completed_procs+=1)\n')

                f.write(f'if !completed_procs! lss {len(procs_info)} (\n')
                f.write(f'    echo !completed_procs!/{len(procs_info)} jobs in batch {batch_num} complete. Waiting...\n')
                f.write('    TIMEOUT /T 5 /NOBREAK > nul\n')
                f.write(f'    goto {wait_label}\n')
                f.write(')\n')
                f.write("echo Batch complete.\n\n")

        f.write("echo Grid search finished.\n")
        f.write("pause\n")

    print(f"\n--- Generation Complete ---")
    print(f"Generated {len(commands)} commands.")
    print(f"To run the grid search, execute the batch file from your command prompt:")
    print(f"  {batch_file_name}")


if __name__ == "__main__":
    generate_commands() 