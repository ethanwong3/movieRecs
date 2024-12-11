import os
import requests

# dict for files

FILES = {
    "movies.csv": "https://www.dropbox.com/scl/fi/2u52ipw8gickvdt5opyqc/movies.csv?rlkey=8dabh25vl33fye5rdb3yfnvy2&st=foe1d0oi&dl=1",
    "ratings.csv": "https://www.dropbox.com/scl/fi/zr0kn2vpw3ahh8k8a03s6/ratings.csv?rlkey=5kt4ntrav7nzfk0at580pwi2p&st=bhwuqkri&dl=1",
}

DATA_DIR = "data/ml-latest"
os.makedirs(DATA_DIR, exist_ok=True)

# downloads file from dropbox

def download_file_from_dropbox(url, dest_path):

    print(f"Downloading {dest_path} from Dropbox...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print(f"Saved {dest_path}")
        else:
            print(f"Failed to download {dest_path}. HTTP Status Code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while downloading {dest_path}: {e}")

# fun dropbox download script for each file in dict

for filename, url in FILES.items():
    dest_path = os.path.join(DATA_DIR, filename)
    download_file_from_dropbox(url, dest_path)
