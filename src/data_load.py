import pandas as pd
import os

# loads movie, rating, tags, genome-tags, and genome-scores data into pd memory

def load():

    files = {
        'ratings': 'data/ml-latest/ratings.csv',
        'movies': 'data/ml-latest/movies.csv',
        'tags': 'data/ml-latest/tags.csv',
        'genome_tags': 'data/ml-latest/genome-tags.csv',
        'genome_scores': 'data/ml-latest/genome-scores.csv',
    }

    datasets = {}

    # validate file existence and load data preview

    try:

        for key, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            datasets[key] = pd.read_csv(path)

            print(f"{key.capitalize()} Data Preview:")
            print(datasets[key].head())

        return datasets

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the files is empty or corrupted: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# main ########################################################################

if __name__ == "__main__":
    datasets = load()

    if datasets:
        print("Datasets loaded successfully.")
    else:
        print("Failed to load datasets. Please check the files and paths.")
