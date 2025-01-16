import subprocess
import os

"""
Run a python script as a subprocess and handle any errors
Args: script_name is a string that is the relative path to the script to run
"""

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python3", script_name], check=True)
    # Re-raise the exception to halt execution in case of failure
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        raise

"""
Prepare the project by executing the required set-up scripts in correct order
data_clean: cleans and saves movies, ratings, and tags datasets
data_preprocessing: precomputes similarity matrices for genres, tags, ratings
"""

def initialize_project():
    scripts = [
        "src/data_clean.py",
        "src/data_preprocessing.py"
    ]

    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    # Start the initialization process
    initialize_project()
    print("Project initialized. You can now run the app.")
