import pandas as pd

def read_file(filepath):
    return pd.read_csv(filepath, sep=",")
