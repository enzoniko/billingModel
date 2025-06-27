import pandas as pd
from ydata_profiling import ProfileReport
import os

def generate_profile_report(csv_path: str, output_dir: str = "."):
    """
    Loads data from a simulation CSV, prepares it for time-series analysis,
    and generates a ydata-profiling report.

    Args:
        csv_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the output HTML report.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        return

    # Extract a clean name for the report title and filename
    sim_name = os.path.basename(csv_path).replace(".csv", "")
    
    # Load the data
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # The first column is unnamed and seems to be a row index, we'll set it as the df index
    if 'Unnamed: 0' in df.columns:
        df = df.set_index('Unnamed: 0')
        df.index.name = 'time_index'

    # Ensure the index is a datetime object for time-series analysis
    df.index = pd.to_datetime(df.index)

    # Encode the categorical 'context' column as integers for better profiling
    df['context_encoded'] = pd.factorize(df['context'])[0]

    # Generate the profile report in time-series mode
    print("Generating data profile report (this may take a few moments)...")
    profile = ProfileReport(
        df,
        tsmode=True,
        title=f"Time-Series Analysis: {sim_name}"
    )

    # Save the report to an HTML file
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{sim_name}_report.html"
    output_path = os.path.join(output_dir, output_filename)
    profile.to_file(output_path)
    
    print(f"Successfully generated report: {output_path}")


if __name__ == "__main__":
    # Define the target CSV file and where to save the report
    CSV_FILE = "processed_data/simulation_100_mass_10900_friction_1.0.csv"
    REPORT_OUTPUT_DIR = "reports"
    
    # Run the report generation function
    generate_profile_report(CSV_FILE, REPORT_OUTPUT_DIR) 