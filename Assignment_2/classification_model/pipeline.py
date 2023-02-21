import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# ========== NEW IMPORTS ========
# Respect to notebook 02-Predicting-Survival-Titanic-Solution

# pipeline
from sklearn.pipeline import Pipeline

# for the preprocessors
from sklearn.base import BaseEstimator, TransformerMixin

# for imputation
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

# for encoding categorical variables
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)
from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer, get_title



# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', variables=config.model_config.categorical_vars)),

    # add missing indicator to numerical variables
    ('missing_indicator',AddMissingIndicator(variables=config.model_config.numerical_vars) ),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=config.model_config.numerical_vars)),


    # Extract first letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=config.model_config.cabin_vars) ),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder',RareLabelEncoder(
        tol=0.05, n_categories=1, variables=config.model_config.categorical_vars) ),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder',OneHotEncoder(
        drop_last=True, variables=config.model_config.categorical_vars) ),

    # scale using standardization
    ('scaler', StandardScaler() ),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit',LogisticRegression(C=0.0005, random_state=0) ),
])