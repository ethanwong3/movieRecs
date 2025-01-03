import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load Datasets
try:
    movies_df = pd.read_csv('data/cleaned_movies.csv')
    tags_df = pd.read_csv('data/cleaned_tags.csv')
    genome_tags_df = pd.read_csv('data/cleaned_genome_tags.csv')
    genome_scores_df = pd.read_csv('data/cleaned_genome_scores.csv')
    print("All datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# preprocess tags
def preprocess_tags(tags_df):
    
    # format data handling NaN and tag aggregation
    tags_df['tag'] = tags_df['tag'].fillna('').astype(str).str.lower().str.strip()
    aggregated_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
    aggregated_tags = aggregated_tags.reset_index().rename(columns={'tag': 'tags'})
    return aggregated_tags

# preprocess genome data
def preprocess_genome_data(genome_tags_df, genome_scores_df):
    
    # mapping tagId to tagName to attach to genome scores
    tag_dict = genome_tags_df.set_index('tagId')['tag'].to_dict()
    genome_scores_df['tagName'] = genome_scores_df['tagId'].map(tag_dict)
    
    # create a movie-tag relevance matrix
    relevance_matrix = genome_scores_df.pivot(index='movieId', columns='tagName', values='relevance').fillna(0)
    return relevance_matrix

# precompute genre similarity
def precompute_genre_similarity(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movies_df['genres'])
    genre_similarity = cosine_similarity(genre_matrix)
    return genre_similarity

# preprocess data
aggregated_tags = preprocess_tags(tags_df)
relevance_matrix = preprocess_genome_data(genome_tags_df, genome_scores_df)
genre_similarity = precompute_genre_similarity(movies_df)

# merge tags and save
movies_df = movies_df.merge(aggregated_tags, on='movieId', how='left').fillna('')
movies_df.to_csv('data/processed_movies.csv', index=False)
pd.DataFrame(genre_similarity).to_csv('data/genre_similarity_matrix.csv', index=False)
relevance_matrix.to_csv('data/movie_tag_relevance.csv', index=True)

print("Data preprocessing completed successfully.")
