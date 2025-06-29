import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test data")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose = 0, allow_writing_files = False),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test,models)

            best_model_score = 0
            best_model_name = ""

            for model_name, model_score in model_report.items():
                if(model_score > best_model_score):
                    best_model_name = model_name
                    best_model_score = model_score

            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
                logging.info("No best model found")
            else:
                logging.info("Best Model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_test_predicted = best_model.predict(X_test)
            print(best_model_name)

            return r2_score(y_test,y_test_predicted)
        
        except Exception as e:
            raise CustomException(e,sys)
        