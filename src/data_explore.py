import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def explore_rating_distribution(ratings):
    """
    Visualizes the distribution of movie ratings.
    """
    sns.histplot(ratings['rating'], bins=10, kde=False)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

def explore_movie_ratings(ratings, movies):
    """
    Finds the top-rated movies and their average ratings.
    """
    # Merge ratings and movies data
    merged_data = pd.merge(ratings, movies, on='movieId')
    avg_ratings = merged_data.groupby('title')['rating'].mean().sort_values(ascending=False)

    # Print the top 10 movies
    print("Top 10 movies:")
    print(avg_ratings.head(10))

def explore_genre_popularity(movies):
    """
    Explores the most common genres in the dataset.
    """
    # Explode the genres column to analyze individual genres
    genres = movies['genres'].explode()  # Correct column name is 'genres'
    count_genres = Counter(genres)

    # Print the most popular genres
    print("Most popular genres:")
    for genre, count in count_genres.most_common(10):
        print(f"{genre}: {count} movies")
    
    # Visualize the genre popularity
    genre_counts = pd.Series(dict(count_genres)).sort_values(ascending=False).head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('Most Popular Genres')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genres')
    plt.show()

if __name__ == "__main__":
    # Load data
    movies = pd.read_csv('data/cleaned_movies.csv')
    ratings = pd.read_csv('data/cleaned_ratings.csv')
    
    # Explore data
    explore_rating_distribution(ratings)
    explore_movie_ratings(ratings, movies)
    explore_genre_popularity(movies)
