"""
Pydantic schemas for the diabetes prediction API

Defines the input model (DiabetesFeatures) and the output model (PredictionResponse)
used by FastAPI to validate requests and serialise responses automatically.
"""
from pydantic import BaseModel

# --------------------------------------------------------------------
# Feature columns — ordered list that matches train_v1.csv (no target)
# --------------------------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]

# ---------------------------------------------------------
# Schemas - request body (input) and response body (output)
# ---------------------------------------------------------

class DiabetesFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp:  float
    s1:  float
    s2:  float
    s3:  float
    s4:  float
    s5:  float
    s6:  float

class PredictionResponse(BaseModel):
    """ Output: predicted quantitative measure of disease progression and the model version that answered."""
    prediction: float
    model_version: str
