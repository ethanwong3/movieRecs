import subprocess
import os

def run_script(script_name):
    """Run a script and handle errors."""
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python3", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        raise

def initialize_project():
    """Prepare the project end-to-end."""
    scripts = [
        "src/data_clean.py",
        "src/data_preprocessing.py",
        "src/data_explore.py"
    ]

    # Check if data exists
    data_files = ["data/cleaned_movies.csv", "data/processed_movies.csv"]
    if not all(os.path.exists(f) for f in data_files):
        print("Data files not found, preparing the dataset...")
        for script in scripts:
            if not os.path.exists(script):
                print(f"Error: {script} not found.")
                continue
            run_script(script)
    else:
        print("Data files already exist. Skipping preprocessing steps.")

if __name__ == "__main__":
    initialize_project()
    print("Project initialized. You can now run the app.")
