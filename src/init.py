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
        "src/data_clean.py",          # Step 1: Clean raw data
        "src/data_preprocessing.py", # Step 2: Preprocess cleaned data
        "src/data_explore.py"        # Optional: Explore data
    ]

    # Check if all cleaned and preprocessed files exist
    cleaned_files = [
        "data/cleaned_movies.csv",
        "data/cleaned_ratings.csv",
        "data/cleaned_tags.csv"
    ]
    preprocessed_files = [
        "data/genre_similarity_matrix.npy"
    ]

    if not all(os.path.exists(f) for f in cleaned_files + preprocessed_files):
        print("Some data files are missing, running the data pipeline...")
        for script in scripts:
            if not os.path.exists(script):
                print(f"Error: {script} not found.")
                continue
            run_script(script)
    else:
        print("All data files already exist. Skipping preprocessing steps.")

if __name__ == "__main__":
    initialize_project()
    print("Project initialized. You can now run the app.")
