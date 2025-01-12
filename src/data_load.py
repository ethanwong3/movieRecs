import pandas as pd

def load_movies(file_path):
    """Load cleaned movies dataset."""
    return pd.read_csv(file_path)

def load_tags(file_path):
    """Load cleaned tags dataset."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    movies = load_movies('data/cleaned_movies.csv')
    tags = load_tags('data/cleaned_tags.csv')
    print(f"Loaded {len(movies)} movies and {len(tags)} tags.")
