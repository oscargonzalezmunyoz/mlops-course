"""
Pydantic schemas for the diabetes validation and prediction API

Defines the input model (DiabetesFeatures) and the output model (PredictionResponse)
used by FastAPI to validate requests and serialise responses automatically.
"""
from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from .config import FEATURE_MIN, FEATURE_MAX

# All input columns must be scaled. We reject unscaled values in columns that 
# were suposed to be scaled, limiting the range from -1 to +1
ScaledFeature = Annotated[float, Field(ge=FEATURE_MIN, le=FEATURE_MAX)]
class DiabetesFeatures(BaseModel):
    """The 10 scaled input features (API request body)."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "age": 0.038, "sex": 0.05, "bmi": 0.0617, "bp": 0.0219,
                    "s1": -0.0442, "s2": -0.0348, "s3": -0.0434,
                    "s4": -0.0026, "s5": 0.0199, "s6": -0.0176,
                }
            ]
        }
    )
    age: ScaledFeature # Age of the patient
    sex: ScaledFeature # Gender of the patient
    bmi: ScaledFeature # Body mass index
    bp:  ScaledFeature # Average blood pressure
    s1:  ScaledFeature # Total serum cholesterol (tc)
    s2:  ScaledFeature # Low-density lipoproteins (ldl)
    s3:  ScaledFeature # High-density lipoproteins (hdl)
    s4:  ScaledFeature # Total cholesterol / HDL (tch)
    s5:  ScaledFeature # Possibly log of serum triglycerides level (ltg)
    s6:  ScaledFeature # Blood sugar level (glu)

class DiabetesRecord(DiabetesFeatures):
    """A full dataset row including the target, for pipeline validation"""
    target: float # target column, unscaled

class PredictionResponse(BaseModel):
    """ Output: predicted quantitative measure of disease progression and the model version that answered."""
    prediction: float
    model_version: str