import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data Exploration:
# - visualise distribution of movie ratings
# - find top-rated movie's avg ratings
# - find common genres

def explore_rating_distribution(ratings):
    
    # create histogram of rating column using 10 groups without smoothing
    sns.histplot(ratings['rating'], bins=10, kde=False)
    
    # label and show the plot
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

def explore_movie_ratings(ratings, movies):

    # join on movieId column, group by title, calculate mean, order descending
    merged_data = pd.merge(ratings, movies, on='movieId')
    avg_ratings = merged_data.groupby('title')['rating'].mean().sort_values(ascending=False)

    # print the top 10 movies
    print("Top 10 movies:")
    print(avg_ratings.head(10))

def explore_genre_popularity(movies):

    # count genre occurrences
    genres = movies['genres'].explode() # flatten genre lists into separate rows
    count_genres = Counter(genres)

    # print the most popular genres and their count
    print("Most popular genres:")
    for genre, count in count_genres.most_common(10):
        print(f"{genre}: {count} movies")
    
    # visualize the top 10 genre's popularity in a horizontal bar chart 
    genre_counts = pd.Series(dict(count_genres)).sort_values(ascending=False).head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('Most Popular Genres')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genres')
    plt.show()

# main ###############################################################################

if __name__ == "__main__":
    # Load data
    movies = pd.read_csv('data/cleaned_movies.csv')
    ratings = pd.read_csv('data/cleaned_ratings.csv')
    
    # Explore data
    explore_rating_distribution(ratings)
    explore_movie_ratings(ratings, movies)
    explore_genre_popularity(movies)
