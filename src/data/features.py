import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_PATH, FEATURED_PATH

def load_data(path : Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering steps and return the enriched DataFrame.
    
    New features created:
    1. bmi_age_interaction: Interaction between BMI and Age [bmi * age]
    2. bp_bmi_interaction: Interaction between Blood Pressure and BMI [bp * bmi]
    3. s1_s2_ratio: Ratio of serum measurements (checking for division by zero) [s1/(s2 + 1e-9)]    
    """
    df = df.copy()
    # Multiply body mass index by the age
    df["bmi_age_interaction"] = df["bmi"] * df["age"]
    # Multiply the average blood pressure by the body mass index
    df["bp_bmi_interaction"] = df["bp"] * df["bmi"]
    # Ratio between the first and second measurement    
    df["s1_s2_ratio"] = df["s1"] / (df["s2"] + 1e-9)

    return df

def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index.
    
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full feature engineering pipeline."""
    print("Loading processed data...")
    df = load_data(PROCESSED_PATH)
    print(f"Rows before feature engineering: {len(df)}")

    print("\nApplying feature engineering...")
    df = feature_engineering(df)

    print("\nSaving dataset...")
    save_data(df, FEATURED_PATH)

    print(f"Final shape: {df.shape}")
    print(f"\nOutput path: {FEATURED_PATH}")

if __name__ == "__main__":
    main()