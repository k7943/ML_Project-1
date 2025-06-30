import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb",) as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_trained_models = {}
        # model_list = []
        # r2_list = []
        # mae_list = []
        # rmse_list = []


        for model_name, model in models.items():
            para = params[model_name]

            gs = GridSearchCV(model,para, cv=5, )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            best_trained_models[model_name] = model
            # model_list.append(model_name)
            # r2_list.append(test_model_score)
            # mae_list.append(mean_absolute_error(y_test, y_test_pred))
            # rmse_list.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

        # results_df = pd.DataFrame({
        #     'Model': model_list,
        #     'R2_Score': r2_list,
        #     'MAE': mae_list,
        #     'RMSE': rmse_list
        # })

        # print("Model Performance After Hyperparameter Tuning:")
        # print(results_df.sort_values(by='R2_Score', ascending=False))

        return report, best_trained_models


    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
