import os
import subprocess

def run_script(script_path):
    try:
        print(f"Running {script_path}...")
        subprocess.run(['python', script_path], check=True)
        print(f"{script_path} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}: {e}\n")
        raise

def initialize_project():
    # Get the absolute path of the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # List of scripts to run, with absolute paths
    scripts = [
        os.path.join(base_dir, "data_load.py"),
        os.path.join(base_dir, "data_clean.py"),
        os.path.join(base_dir, "data_preprocessing.py"),
        os.path.join(base_dir, "data_explore.py"),
    ]

    # Check and execute scripts
    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: {script} not found.")
            return
        run_script(script)

if __name__ == "__main__":
    initialize_project()
