import os,sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from src.utils import evaluate_model
from sklearn.metrics import r2_score

@dataclass
class model_trainer_config:
    model_file_path=os.path.join("artifacts","model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer=model_trainer_config()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model_training_started")
            X_train,X_test,Y_train,Y_test=(train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1])

            models={"Linear_regression":LinearRegression(),
                    "Lasso":Lasso(),
                    "Ridge":Ridge(),
                    "Elasticnet":ElasticNet(),
                    "Random_forest":RandomForestRegressor(),
                    "Decision_tree":DecisionTreeRegressor(),
                    "SVM_algo":SVR(),
                    "Adaboost":AdaBoostRegressor(),
                    "Gradient_boosting":GradientBoostingRegressor(),
                    "Naive_bayes":GaussianNB(),
                    "K_neibours":KNeighborsRegressor() 
                    }
            
            model_score=evaluate_model(x_train=X_train,x_test=X_test,y_train=Y_train,y_test=Y_test,models=models)

            max_score=(max(model_score.values()))

            best_model_name=([i[0] for i in model_score.items() if i[1]==max_score])

            best_model=models[best_model_name[0]]

            y_pred=best_model.predict(X_test)
            
            return r2_score(Y_test,y_pred)



        except Exception as e:
            raise CustomException(e,sys)
        