import pandas as pd

# cleaning:
# - remove incomplete cols
# - formatting / normalising columns

def clean_movies(movies):
    required_columns = {'movieId', 'title', 'genres'}
    if not required_columns.issubset(movies.columns):
        raise ValueError(f"Missing required columns in movies dataset: {required_columns - set(movies.columns)}")

    movies = movies.dropna()
    movies['genres'] = movies['genres'].str.split('|')
    return movies


def clean_ratings(ratings):
    ratings = ratings.dropna()
    return ratings

def clean_tags(tags):
    tags = tags.dropna()
    tags['tag'] = tags['tag'].str.lower().str.strip()
    return tags

def clean_genome_tags(genome_tags):
    genome_tags = genome_tags.dropna()
    genome_tags['tag'] = genome_tags['tag'].str.lower().str.strip()
    return genome_tags

def clean_genome_scores(genome_scores):
    genome_scores = genome_scores.dropna()
    return genome_scores

# saving: filters then duplicates clean data into new CSV (without row indices)

def save_clean_data(cleaned_data):
    
    paths = {
        'movies': 'data/cleaned_movies.csv',
        'ratings': 'data/cleaned_ratings.csv',
        'tags': 'data/cleaned_tags.csv',
        'genome_tags': 'data/cleaned_genome_tags.csv',
        'genome_scores': 'data/cleaned_genome_scores.csv',
    }

    for key, df in cleaned_data.items():
        df.to_csv(paths[key], index=False) # exclude row index
        print(f"Cleaned {key} dataset saved to {paths[key]}.")

# main ########################################################################

if __name__ == "__main__":
    
    # load raw data
    from data_load import load
    datasets = load()

    # clean data
    if datasets:
        cleaned_data = {            
            'movies': clean_movies(datasets['movies']),
            'ratings': clean_ratings(datasets['ratings']),
            'tags': clean_tags(datasets['tags']),
            'genome_tags': clean_genome_tags(datasets['genome_tags']),
            'genome_scores': clean_genome_scores(datasets['genome_scores']),
        }

    # save cleaned data
    save_clean_data(cleaned_data)
