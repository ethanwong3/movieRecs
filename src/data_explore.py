import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# data Exploration:
# - visualise distribution of movie ratings
# - find avg ratings of top-rated movies
# - find common genres
# - find common tags
# - find relevant tags

def explore_rating_distribution(ratings):
    
    # create histogram with 10 cols no smoothing, then label and show
    sns.histplot(ratings['rating'], bins=10, kde=False)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

def explore_movie_ratings(ratings, movies):
    
    # join on movieId column, group by title, calculate mean, order descending, then print top 10
    merged_data = pd.merge(ratings, movies, on='movieId')
    avg_ratings = merged_data.groupby('title')['rating'].mean().sort_values(ascending=False)
    print("Top 10 movies:")
    print(avg_ratings.head(10))

def explore_genre_popularity(movies):
    
    if movies.empty:
        print("No data available to explore")
        return
    
    # count occurrences once lists flattened
    genres = movies['genres'].explode()
    count_genres = Counter(genres)
    if not count_genres:
        print("No genres available in the dataset")
        return
    
    # demo
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

def explore_tag_usage(tags):
    
    tag_counts = tags['tag'].value_counts().head(10)
    print("Most used tags:")
    print(tag_counts)
    sns.barplot(x=tag_counts.values, y=tag_counts.index)
    plt.title('Most Common Tags')
    plt.xlabel('Frequency')
    plt.ylabel('Tags')
    plt.show()

def explore_genome_relevance(genome_scores, genome_tags):
    
    # join genome-scores with genome-tags to map tagId to tag
    merged_tags = pd.merge(genome_scores, genome_tags, on='tagId')
    
    # sort tags by avg relevance and demo in bar graph
    avg_relevance = merged_tags.groupby('tag')['relevance'].mean().sort_values(ascending=False).head(10)
    print("Top 10 relevant tags:")
    print(avg_relevance)
    avg_relevance.plot(kind='barh', color='skyblue')
    plt.title('Top Relevant Tags by Average Relevance')
    plt.xlabel('Relevance')
    plt.ylabel('Tags')
    plt.show()

# main ########################################################################

if __name__ == "__main__":
    
    # retrieve data
    movies = pd.read_csv('data/cleaned_movies.csv')
    ratings = pd.read_csv('data/cleaned_ratings.csv')
    tags = pd.read_csv('data/cleaned_tags.csv')
    genome_tags = pd.read_csv('data/cleaned_genome_tags.csv')
    genome_scores = pd.read_csv('data/cleaned_genome_scores.csv')

    # explore data
    explore_rating_distribution(ratings)
    explore_movie_ratings(ratings, movies)
    explore_genre_popularity(movies)
    explore_tag_usage(tags)
    explore_genome_relevance(genome_scores, genome_tags)
