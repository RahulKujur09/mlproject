import os
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    model_report = []

    param_grids = {
        "random_forest_regressor": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "decision_tree_regressor": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "gradient_boosting_regressoe": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "ada_bost_regressor": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1, 1.0]
        },
        "linear_regression": {},
        "lasso": {
            "alpha": [0.01, 0.1, 1.0, 10]
        },
        "ridge": {
            "alpha": [0.01, 0.1, 1.0, 10]
        },
        "xg_boost_regressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "cat_boost_regressor": {
            "iterations": [100, 200],
            "learning_rate": [0.05, 0.1],
            "depth": [4, 6]
        }
    }

    for model_name, model in models.items():
        
        # 🚨 Special handling for CatBoost
        if model_name == "cat_boost_regressor":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            model_report.append({
                "Model": model_name,
                "R2 Score": score,
                "Best Params": "Default"
            })

            models[model_name] = model
            continue

        params = param_grids.get(model_name, {})

        if params:
            gs = GridSearchCV(
                model,
                param_grid=params,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        y_pred = best_model.predict(X_test)
        score = r2_score(y_test, y_pred)

        model_report.append({
            "Model": model_name,
            "R2 Score": score,
            "Best Params": params if params else "Default"
        })

        # update original model with best model
        models[model_name] = best_model

    return pd.DataFrame(model_report)



