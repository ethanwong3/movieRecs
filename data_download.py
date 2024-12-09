import os
import requests

"""
https://drive.google.com/file/d/1vFl7Mz5nU3Nxz6WOn7rIzs_b56DCGAAa/view?usp=drive_link
https://drive.google.com/file/d/1PGxUrRzQU1wK2lIO0j3E2oAGSuzKDpjk/view?usp=drive_link
https://drive.google.com/file/d/1I2zr1GYjUk5PX6RcIqr5oBPb-wzoJUNW/view?usp=drive_link
https://drive.google.com/file/d/1EfEBIw3cr20gmTEEv4QGtSjDbDJxnXWc/view?usp=drive_link
https://drive.google.com/file/d/18wBVOE1F8e5TBkV3CltNB6GElEqDk82z/view?usp=drive_link
https://drive.google.com/file/d/1gFdO9k5IHwNc7_Dh5hqL-b_Dd4OxCzKQ/view?usp=drive_link
"""

FILES = {
    "links.csv": "1vFl7Mz5nU3Nxz6WOn7rIzs_b56DCGAAa",
    "movies.csv": "1PGxUrRzQU1wK2lIO0j3E2oAGSuzKDpjk",
    "genome-scores.csv": "1I2zr1GYjUk5PX6RcIqr5oBPb-wzoJUNW",
    "genome-tags.csv": "1EfEBIw3cr20gmTEEv4QGtSjDbDJxnXWc",
    "tags.csv": "18wBVOE1F8e5TBkV3CltNB6GElEqDk82z",
    "ratings.csv": "1gFdO9k5IHwNc7_Dh5hqL-b_Dd4OxCzKQ",
}

DATA_DIR = "data/ml-latest"
os.makedirs(DATA_DIR, exist_ok=True)

# downloads a file from google drive

def download_file_from_google_drive(file_id, dest_path):
    print(f"Downloading {dest_path}...")
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"Saved {dest_path}")
    else:
        print(f"Failed to download {dest_path} (status code: {response.status_code})")

# download each file

for filename, file_id in FILES.items():
    dest_path = os.path.join(DATA_DIR, filename)
    download_file_from_google_drive(file_id, dest_path)
