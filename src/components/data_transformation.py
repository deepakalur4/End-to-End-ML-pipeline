import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sys
import pandas as pd
import numpy as np
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_pickle_file_path=os.path.join("artifacts","preprocessor.pkl")

class data_transformation:
    def __init__(self):
        self.data_transformation=data_transformation_config()
    
    def get_preprocessor_object(self):
        try:
            logging.info("Entering the preprocessor object path")
            num_columns=["reading_score","writing_score"]
            cat_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline([
                ("Imputer",SimpleImputer(strategy="median")),
                ("Scaling",StandardScaler())                
                ])            
            

            cat_pipeline=Pipeline([
                ("Imputer",SimpleImputer(strategy="most_frequent")),
                ("Encoder",OneHotEncoder()),
                ("Scaling",StandardScaler(with_mean=False))          
                ])
            
            preprocessor=ColumnTransformer([
                ("Num_pipeline",num_pipeline,num_columns),
                ("Cat_pipeline",cat_pipeline,cat_columns)               
                                ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transoformation(self,train_path,test_path):
        try:
            logging.info("Started data transformation")
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            preprocessor_object=self.get_preprocessor_object()

            output_feature="math_score"

            input_feature_train_data=train_data.drop(columns=[output_feature],axis=1)
            output_feature_train_data=train_data[output_feature]


            input_feature_test_data=test_data.drop(columns=[output_feature],axis=1)
            output_feature_test_data=test_data[output_feature]

            train_data_scaled=preprocessor_object.fit_transform(input_feature_train_data)
            test_data_scaled=preprocessor_object.fit_transform(input_feature_test_data)

            train_data_arr=np.c_[train_data_scaled,np.array(output_feature_train_data)]
            test_data_arr=np.c_[test_data_scaled,np.array(output_feature_test_data)]

            save_object(file_path=self.data_transformation.preprocessor_pickle_file_path,obj=preprocessor_object)


            return (train_data_arr,test_data_arr,self.data_transformation.preprocessor_pickle_file_path)

        except Exception as e:
            raise CustomException(e,sys)