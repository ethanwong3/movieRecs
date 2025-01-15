import numpy as np

genre_similarity = np.load("data/genre_similarity_matrix.npy")

for i in range(5):
    print(f"Row {i}:", genre_similarity[i][:10])
