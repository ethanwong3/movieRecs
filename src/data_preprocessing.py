import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load Datasets
try:
    movies_df = pd.read_csv('data/cleaned_movies.csv')
    tags_df = pd.read_csv('data/cleaned_tags.csv')
    genome_tags_df = pd.read_csv('data/cleaned_genome_tags.csv')
    genome_scores_df = pd.read_csv('data/cleaned_genome_scores.csv')
    print("All datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Preprocess tags
def preprocess_tags(tags_df):
    tags_df['tag'] = tags_df['tag'].fillna('').astype(str).str.lower().str.strip()
    aggregated_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
    aggregated_tags = aggregated_tags.reset_index().rename(columns={'tag': 'tags'})
    return aggregated_tags

# Preprocess genome data
def preprocess_genome_data(genome_tags_df, genome_scores_df):
    tag_dict = genome_tags_df.set_index('tagId')['tag'].to_dict()
    genome_scores_df['tagName'] = genome_scores_df['tagId'].map(tag_dict)
    relevance_matrix = genome_scores_df.pivot(index='movieId', columns='tagName', values='relevance').fillna(0)
    return relevance_matrix

# Precompute genre similarity
def precompute_genre_similarity(movies_df, chunk_size=1000, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movies_df['genres'])
    num_movies = genre_matrix.shape[0]
    
    # Sparse format: Only store top-N similar movies
    sparse_similarity = {}

    for start in range(0, num_movies, chunk_size):
        end = min(start + chunk_size, num_movies)
        chunk_similarity = cosine_similarity(genre_matrix[start:end], genre_matrix)

        # Only store top-N similar movies for each movie
        for i, row in enumerate(chunk_similarity):
            top_indices = row.argsort()[-top_n-1:-1][::-1]
            sparse_similarity[start + i] = {j: row[j] for j in top_indices}
        print(f"Computed similarity for movies {start} to {end}")
    
    return sparse_similarity

# Start timing
start_time = time.time()

# Preprocess data
aggregated_tags = preprocess_tags(tags_df)
relevance_matrix = preprocess_genome_data(genome_tags_df, genome_scores_df)
genre_similarity = precompute_genre_similarity(movies_df)

# Merge tags and save data
movies_df = movies_df.merge(aggregated_tags, on='movieId', how='left').fillna('')
movies_df.to_csv('data/processed_movies.csv', index=False)

# Save genre similarity and relevance matrix in binary format
np.save('data/genre_similarity_matrix.npy', genre_similarity)
relevance_matrix.to_csv('data/movie_tag_relevance.csv', index=True)

print(f"Data preprocessing completed successfully in {time.time() - start_time:.2f} seconds.")
