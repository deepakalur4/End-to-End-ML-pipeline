from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
import sys


def save_object(file_path,obj):
    try:
        with open(file_path,"wb") as file_obj:
            return pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

models={"Linear_regression":LinearRegression(),
        "Lasso":Lasso(),
                    "Ridge":Ridge(),
           "Elasticnet":ElasticNet(),
        "Random_forest":RandomForestRegressor(),
        "Decision_tree":DecisionTreeRegressor(),
            "Adaboost":AdaBoostRegressor(),
            "Naive_bayes":GaussianNB(),
                    "K_neibours":KNeighborsRegressor() 
                    }

def evaluate_model(x_train,x_test,y_train,y_test,models):
    try:
        score=dict()
        for k in models.items():
            model=k[1]
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            rscore=r2_score(y_test,y_pred)
            score[k[0]]=rscore
        return score



    except Exception as e:
        raise CustomException(e,sys)
            
