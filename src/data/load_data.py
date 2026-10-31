import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from src.config import RAW_PATH

def load_raw_data() -> pd.DataFrame:
    """ Load the diabetes dataset and return it as a DataFrame. """
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame.copy()
    return df

def save_dataset(df: pd.DataFrame, path: str) -> None:
    """ Save a DataFrame to a CSV file, creating directories as needed.

    Args: df: DataFrame to save
    path: Destination file path
     """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)

def main() -> None:
    """ Run the data ingestion pipeline. """
    
    print("Loading dataset...")
    df = load_raw_data()
    print("Saving dataset...")
    save_dataset(df, RAW_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Output path: {RAW_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()