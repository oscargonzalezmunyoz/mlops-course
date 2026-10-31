from typing import Optional
import sys
import pandas as pd
from pydantic import BaseModel, ValidationError

class DiabetesData(BaseModel):
    age: float # Age of the patient
    sex: float # Gender of the patient
    bmi: float # Body mass index
    bp:  float # Average blood pressure
    s1:  float # Total serum cholesterol (tc)
    s2:  float # Low-density lipoproteins (ldl)
    s3:  float # High-density lipoproteins (hdl)
    s4:  float # Total cholesterol / HDL (tch)
    s5:  float # Possibly log of serum triglycerides level (ltg)
    s6:  float # Blood sugar level (glu)
    target: float # targeat column

def validate_row(row: dict, index: int) -> bool:
    """ Try to validate a single row against the Diabetes model.
    Args:
        row: A dictionary representing one row of the dataset
        index: The row index, used for error messages
    Returns:
        True if the row is valid, False otherwise
    """
    try:
        DiabetesData(**row) # '**' unpacks the dictionary in named arguments
        return True
    except ValidationError as e:
        print(f"Row {index} failed validation: {e}")
        return False

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate all rows in the dataset and print a summary
    
    Args:
        df: The pandas DataFrame to validate.
    """

    total = len(df)
    valid = 0
    invalid = 0

    for index, row in enumerate(df.to_dict("records")):
        if validate_row(row, index):
            valid += 1
        else:
            invalid +=1
    
    print("\n--- Validation Summary ---")
    print(f"Total rows: {total}")
    print(f"Valid rows: {valid}")
    print(f"Invalid rows: {invalid}")
    if invalid > 0:
        print("Validation failed.")
        # Early exit when the validation fails
        sys.exit(1)

def main():
    """Load the raw dataset and run validation."""
    path = "data/raw/diabetes.csv"
    print(f"Load dataset from: {path}")
    df = pd.read_csv(path)
    validate_dataset(df)    

if __name__ == "__main__":
    main()