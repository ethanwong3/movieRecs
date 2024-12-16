import pandas as pd

movies_df = pd.read_csv('data/cleaned_movies.csv')
print(movies_df.head())        # Check the first rows
print(movies_df['genres'].isnull().sum())  # Count missing values in genres
print(movies_df['genres'].unique())  # See unique values in genres
