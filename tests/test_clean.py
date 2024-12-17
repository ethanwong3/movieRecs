import pandas as pd

# Load the cleaned movies dataset
movies_df = pd.read_csv('data/cleaned_movies.csv')

# Verify the genres column
print("Checking the genres column after cleaning...\n")
print(movies_df['genres'].head(10))
print("\nUnique genres format:")
print(movies_df['genres'].unique())
