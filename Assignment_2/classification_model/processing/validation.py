# Changes in validate.py. This should be for titanic dataset.

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from classification_model.processing.data_manager import get_dataframe_ready
from classification_model.config.core import config


# def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
#     """Check model inputs for na values and filter."""
#     validated_data = input_data.copy()
#     new_vars_with_na = [
#         var
#         for var in config.model_config.features
#         if var
#         not in config.model_config.categorical_vars_with_na_frequent
#         + config.model_config.categorical_vars_with_na_missing
#         + config.model_config.numerical_vars_with_na
#         and validated_data[var].isnull().sum() > 0
#     ]
#     validated_data.dropna(subset=new_vars_with_na, inplace=True)

#     return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    
    ready_data = get_dataframe_ready(dataframe= input_data)
    validated_data = ready_data[config.model.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleHouseDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    fare: Optional[float]
    age: Optional[int]
    sibsp: Optional[int]
    cabin: Optional[str]
    embarked: Optional[str]
    body: Optional[int]
    sex: Optional[str]
    parch: Optional[int]
    ticket: Optional[int]
    boat: Optional[Union[str, int]]


class MultipleHouseDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
