# Dependencies
import numpy as np
import pandas as pd


class Dataset:

    # Attributes
    columns = None  # DataFrame columns type
    df = None  # DataFrame object containing data

    # Construtcor
    def __init__(self, df=None, columns={}):
        # Define dataset columns
        self.columns = columns
        # Instantiate new data container
        self.df = pd.DataFrame(data=df, columns=self.columns)

    # Load inner dataset from disk (.json file)
    def from_json(self, in_path, date_columns=[]):
        # Load entries into inner DataFrame
        self.df = pd.read_json(
            in_path,
            orient='records',
            convert_dates=date_columns,
            dtype=self.columns
        )

    # Save inner dataset to disk (.json file)
    def to_json(self, out_path):
        # Store pandas Dataframe as json object
        self.df.to_json(out_path, orient='records')


# Test
if __name__ == '__main__':
    # Instantiate new empty dataset
    ds = Dataset(columns={
            'test_id': np.unicode_,
            'test_text': np.unicode_
    })
    # Show dataset
    print(ds.df.head())
