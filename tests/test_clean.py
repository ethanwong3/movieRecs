import unittest
import pandas as pd
from src.data_clean import clean_movies

class TestDataClean(unittest.TestCase):
    def test_clean_movies(self):
        data = {
            'movieId': [1, 2],
            'title': ['Toy Story', 'Jumanji'],
            'genres': ['Adventure|Comedy', None],
        }
        df = pd.DataFrame(data)
        cleaned_df = clean_movies(df)
        self.assertEqual(len(cleaned_df), 1)  # 1 row remains after dropping NaN
        self.assertEqual(cleaned_df.iloc[0]['title'], 'Toy Story')

if __name__ == '__main__':
    unittest.main()
