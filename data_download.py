import os
import requests

def download_file_from_google_drive(file_id, dest_path):
    """
    Downloads a file from Google Drive that requires a virus scan confirmation for large files.
    """
    print(f"Downloading {dest_path}...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Extract the confirmation token from the cookies (if present)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(url, stream=True)
            break
    
    # Save the file
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                f.write(chunk)
        print(f"Saved {dest_path}")
    else:
        print(f"Failed to download {dest_path}. HTTP Status Code: {response.status_code}")

# Update your files dictionary
FILES = {
    "movies.csv": "1PGxUrRzQU1wK2lIO0j3E2oAGSuzKDpjk",  # Replace with the file ID for movies.csv
    "ratings.csv": "1gFdO9k5IHwNc7_Dh5hqL-b_Dd4OxCzKQ",  # Replace with the file ID for ratings.csv
}


DATA_DIR = "data/ml-latest"
os.makedirs(DATA_DIR, exist_ok=True)

# Download files
for filename, file_id in FILES.items():
    dest_path = os.path.join(DATA_DIR, filename)
    download_file_from_google_drive(file_id, dest_path)
