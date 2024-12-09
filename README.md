# Movie Recommender System

A movie recommendation system built using **Python** and **machine learning techniques**. This project demonstrates collaborative filtering, content-based filtering, and hybrid recommendation models. The data is sourced from **MovieLens**.

---

## Features

- Collaborative Filtering (User-to-User and Item-to-Item)
- Content-Based Filtering (Based on movie metadata)
- Hybrid Models (Combines collaborative and content-based approaches)
- Data download script to fetch datasets from Google Drive
- Interactive Web App using Streamlit or Flask

---

## Project Structure

movieRecs/
├── app/ # Web app files
├── data/ # Placeholder for downloaded datasets
│ └── README.md # Placeholder to explain the data folder
├── data_sample/ # Sample data including small, non-sensitive data for testing
├── notebooks/ # Jupyter notebooks for experimentation
├── src/ # Python scripts for the project
│ ├── data_clean.py # Data cleaning script
│ ├── data_explore.py # Data exploration script
│ └── data_load.py # Data loading script
├── tests/ # Test cases
├── download_data.py # Script to download datasets
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## Dependencies

_streamlit_
create and deploy public interactive apps

_pandas_
data manipulation and loading data sets

_matplotlib.pyplot_
data visualisation

_seaborn_
higher data visualisation

_cosine similarity_
measures similarity between two vectors, here it calculates similarity of movies based on genres

_TfidVectorizer_
converts text or genres into numerical vectors to identify importance of each genre to its movie

## Progress

Data Handling

- download_data.py: automates process of fetching data from Google Drive to ensure efficient and lightweight repo management.

- data_clean.py: processes raw data by removing missing values and formatting data.

EDA (Exploratory Data Analysis)

- data_explore.py: provides tools for analysing and visualising data. Currently there are functions for:
  - rating distribution histograms
  - top-rated movies
  - genre popularity

Streamlit Web App

- main.py: a lightweight streamlit app serves as the interface for future recommendations

Testing

...

## How the Recommender Works

Input:

User provides the title of a movie (e.g., "Toy Story (1995)").
The function retrieves the genres of all movies from the dataset.

TF-IDF Vectorization:

Converts genres into numerical vectors, assigning weights to terms based on their importance (frequent genres like "Comedy" are weighted lower than unique genres like "Sci-Fi").

Cosine Similarity:

Compares the genre vector of the input movie with all other movies to compute similarity scores.

Output:

A list of the most similar movies based on genres.
