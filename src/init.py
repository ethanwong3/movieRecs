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
    """Initialize the project by running all scripts in order."""
    scripts = [
        "src/data_clean.py",
        "src/data_preprocessing.py"  # Include preprocessing script only as subprocess
    ]

    for script in scripts:
        run_script(script)

    print("All preprocessing steps completed successfully.")

if __name__ == "__main__":
    initialize_project()
    print("Project initialized. You can now run the app.")
