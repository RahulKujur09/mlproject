import os
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models: dict):
    try:
        report = []

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report.append(
                {
                    "Model": model_name,
                    "R2 Score": r2,
                    "Mean Absolute Error": mae,
                    "Root Mean Square Error": rmse,
                }
            )

        return pd.DataFrame(report)

    except Exception as e:
        raise CustomException(e, sys)


