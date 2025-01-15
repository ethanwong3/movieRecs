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

    # Check if all data files exist
    data_files = [
        "data/cleaned_movies.csv",
        "data/genre_similarity_matrix.npy",
        "data/tag_similarity_matrix.npy",
        "data/ratings_similarity_matrix.npy"
    ]
    if not all(os.path.exists(f) for f in data_files):
        print("Data files are missing, running the data pipeline...")
        for script in scripts:
            run_script(script)
    else:
        print("All data files exist. Skipping preprocessing steps.")

if __name__ == "__main__":
    initialize_project()
    print("Project initialized. You can now run the app.")
