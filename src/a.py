import numpy as np

genre_similarity = np.load("data/genre_similarity_matrix.npy")
tag_similarity = np.load("data/tag_similarity_matrix.npy")
ratings_similarity = np.load("data/ratings_similarity_matrix.npy")

print("Genre Similarity Matrix Sample:", genre_similarity[0][:10])
print("Tag Similarity Matrix Sample:", tag_similarity[0][:10])
print("Ratings Similarity Matrix Sample:", ratings_similarity[0][:10])
