import unittest

import pandas as pd

from src.scripts.process_data import read_file
from src.scripts.process_data import transform_data


class TestDataProcessing(unittest.TestCase):
    def test_transform_data(self):
        """Test transformation of data based on skewness."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 12, 10, 15],
            'C': [1, 2, 1, 2],
            'Class': [0, 1, 0, 1]
        })
        transformed_df = transform_data(df)
        # Check that the shape remains the same
        self.assertEqual(transformed_df.shape, df.shape)  # shape should be the same
        # Check that the 'Class' column remains unchanged
        self.assertTrue(all(transformed_df['Class'] == df['Class']))  # Class column should remain unchanged

    def test_transform_data_no_skew(self):
        """Test that data remains unchanged when no skewness is present."""
        df = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [1, 1, 1, 1],
            'C': [0, 0, 0, 0],
            'Class': [0, 1, 0, 1]
        })
        transformed_df = transform_data(df)
        # Ensure no transformation was applied
        pd.testing.assert_frame_equal(transformed_df, df, "Data should remain unchanged when no skewness")


if __name__ == '__main__':
    unittest.main()
