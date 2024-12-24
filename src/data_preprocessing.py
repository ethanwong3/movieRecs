import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load Datasets
movies_df = pd.read_csv('data/cleaned_movies.csv')
ratings_df = pd.read_csv('data/cleaned_ratings.csv')
tags_df = pd.read_csv('data/cleaned_tags.csv')
genome_tags_df = pd.read_csv('data/cleaned_genome_tags.csv')
genome_scores_df = pd.read_csv('data/cleaned_genome_scores.csv')

# tags.csv
def preprocess_tags(tags_df):
    # aggregate tags for each movie
    aggregated_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
    return aggregated_tags

# 2. genome-scores.csv
def preprocess_genome_data(genome_tags_df, genome_scores_df):
    # create a dictionary mapping tagId to tagName
    tag_dict = genome_tags_df.set_index('tagId')['tag'].to_dict()
    
    # add tag names to genome_scores
    genome_scores_df['tagName'] = genome_scores_df['tagId'].map(tag_dict)
    
    # create a movie-tag relevance matrix
    relevance_matrix = genome_scores_df.pivot(index='movieId', columns='tagName', values='relevance').fillna(0)
    return relevance_matrix

# preprocess data
aggregated_tags = preprocess_tags(tags_df)
relevance_matrix = preprocess_genome_data(genome_tags_df, genome_scores_df)

# merge tags into movies_df
movies_df = movies_df.merge(aggregated_tags, on='movieId', how='left')
movies_df['tags'] = movies_df['tags'].fillna('')  # Fill missing tags with empty string

# save preprocessed data for further use
movies_df.to_csv('data/processed_movies.csv', index=False)
relevance_matrix.to_csv('data/movie_tag_relevance.csv', index=True)

