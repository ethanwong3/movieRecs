import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

def preprocess_tags(tags_df):
    """Aggregate tags into a single string per movie."""
    tags_df['tag'] = tags_df['tag'].fillna('').str.lower().str.strip()
    aggregated_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    aggregated_tags.rename(columns={'tag': 'tags'}, inplace=True)
    return aggregated_tags

def preprocess_genome_data(genome_tags_df, genome_scores_df):
    """Create a movie-tag relevance matrix."""
    tag_dict = genome_tags_df.set_index('tagId')['tag'].to_dict()
    genome_scores_df['tagName'] = genome_scores_df['tagId'].map(tag_dict)
    relevance_matrix = genome_scores_df.pivot(index='movieId', columns='tagName', values='relevance').fillna(0)
    return relevance_matrix

def precompute_genre_similarity(movies_df):
    """Compute genre similarity using TF-IDF."""
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movies_df['genres'].apply(lambda x: ' '.join(x)))
    genre_similarity = cosine_similarity(genre_matrix)
    return genre_similarity

def preprocess_data():
    """Main preprocessing pipeline."""
    start_time = time.time()
    movies_df = pd.read_csv('data/cleaned_movies.csv')
    tags_df = pd.read_csv('data/cleaned_tags.csv')

    # Process tags and genome data
    aggregated_tags = preprocess_tags(tags_df)
    movies_df = movies_df.merge(aggregated_tags, on='movieId', how='left').fillna('')

    # Compute genre similarity
    print("Computing genre similarity matrix...")
    genre_similarity = precompute_genre_similarity(movies_df)

    # Save results
    movies_df.to_csv('data/processed_movies.csv', index=False)
    np.save('data/genre_similarity_matrix.npy', genre_similarity)

    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    preprocess_data()
