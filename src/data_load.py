import pandas as pd

"""
Load the cleaned movies dataset from a CSV file
Args: file_path is a string specifying the path to the cleaned movies CSV file
Returns: a Pandas DataFrame containing the movies dataset
"""

def load_movies(file_path):
    return pd.read_csv(file_path)

"""
Load the cleaned tags dataset from a CSV file
Args: file_path is a string specifying the path to the cleaned tags CSV file
Returns: a Pandas DataFrame containing the tags dataset
"""

def load_tags(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Load the cleaned movies and tags datasets
    movies = load_movies('data/cleaned_movies.csv')
    tags = load_tags('data/cleaned_tags.csv')
    
    # Print the number of records in each dataset
    print(f"Loaded {len(movies)} movies and {len(tags)} tags.")
