import subprocess
import os

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}\n")
        raise

def initialize_project():
    scripts = [
        "data_clean.py",
        "data_explore.py",
        "data_load.py",
        "data_preprocessing.py"
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: {script} not found.")
            return
        run_script(script)

if __name__ == "__main__":
    initialize_project()
