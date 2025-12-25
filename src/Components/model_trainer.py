import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path : str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "cat_boost_regressor": CatBoostRegressor(verbose=False),
                "random_forest_regressor": RandomForestRegressor(),
                "ada_bost_regressor": AdaBoostRegressor(),
                "gradient_boosting_regressoe": GradientBoostingRegressor(),
                "linear_regression": LinearRegression(),
                "lasso": Lasso(),
                "ridge": Ridge(),
                "xg_boost_regressor": XGBRegressor(objective="reg:squarederror"),
                "decision_tree_regressor": DecisionTreeRegressor()
            }

            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_index = model_report["R2 Score"].idxmax()
            best_model_score = float(model_report.loc[best_model_index, "R2 Score"])
            best_model_name = str(model_report.at[best_model_index, "Model"])
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model found!", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
