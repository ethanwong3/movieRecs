import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, csr_matrix, vstack
from joblib import Parallel, delayed
import time
import gc

# Function to preprocess tags
def preprocess_tags(tags_df):
    tags_df['tag'] = tags_df['tag'].fillna('').astype(str).str.lower().str.strip()
    aggregated_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
    aggregated_tags = aggregated_tags.reset_index().rename(columns={'tag': 'tags'})
    return aggregated_tags

# Function to preprocess genome data
def preprocess_genome_data(genome_tags_df, genome_scores_df):
    tag_dict = genome_tags_df.set_index('tagId')['tag'].to_dict()
    genome_scores_df['tagName'] = genome_scores_df['tagId'].map(tag_dict)
    relevance_matrix = genome_scores_df.pivot(index='movieId', columns='tagName', values='relevance').fillna(0)
    return csr_matrix(relevance_matrix)

# Function to compute genre similarity in chunks
def compute_genre_similarity_chunk(tfidf_matrix, start, end):
    chunk_sim = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
    return csr_matrix(chunk_sim)

def precompute_genre_similarity_sparse(movies_df, chunk_size=500, output_path="data/genre_similarity_matrix.npz"):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    num_movies = tfidf_matrix.shape[0]

    # Process in chunks
    print("Starting sparse computation of genre similarity matrix...")
    similarity_matrices = []
    for start in range(0, num_movies, chunk_size):
        end = min(start + chunk_size, num_movies)
        print(f"Computing similarity for movies {start} to {end}...")
        similarity_chunk = compute_genre_similarity_chunk(tfidf_matrix, start, end)
        similarity_matrices.append(similarity_chunk)
        gc.collect()  # Force garbage collection to free memory

    # Combine chunks and save
    genre_similarity = vstack(similarity_matrices)
    save_npz(output_path, genre_similarity)
    print(f"Saved genre similarity matrix to {output_path}")
    return genre_similarity

# Main preprocessing function
def preprocess_data():
    start_time = time.time()
    
    # Load datasets
    try:
        print("Loading datasets...")
        movies_df = pd.read_csv('data/cleaned_movies.csv')
        tags_df = pd.read_csv('data/cleaned_tags.csv')
        genome_tags_df = pd.read_csv('data/cleaned_genome_tags.csv')
        genome_scores_df = pd.read_csv('data/cleaned_genome_scores.csv')
        print("All datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Preprocess tags and genome data
    print("Processing tags and genome data...")
    aggregated_tags = preprocess_tags(tags_df)
    relevance_matrix = preprocess_genome_data(genome_tags_df, genome_scores_df)

    # Save relevance matrix (sparse)
    save_npz('data/tag_similarity_matrix.npz', relevance_matrix)
    print("Saved tag similarity matrix as sparse matrix.")

    # Precompute genre similarity matrix
    print("Computing genre similarity matrix...")
    genre_similarity = precompute_genre_similarity_sparse(movies_df)

    # Save processed data
    print("Saving processed data...")
    movies_df = movies_df.merge(aggregated_tags, on='movieId', how='left').fillna('')
    movies_df.to_csv('data/processed_movies.csv', index=False)

    print(f"Data preprocessing completed successfully in {time.time() - start_time:.2f} seconds.")

    # Explicit cleanup to release resources
    del movies_df, tags_df, genome_tags_df, genome_scores_df
    del genre_similarity, relevance_matrix, aggregated_tags
    gc.collect()

if __name__ == "__main__":
    preprocess_data()
