from src.exception import CustomException
from src.logger import logging
import os
from src.utils import load_object
import sys
import pandas as pd

class pipeline:
    try:
        def __init__(self) -> None:
            pass

        def predict_pipeline(self,features):
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            scaled_data=preprocessor.fit_transform(features)
            preds=model.predict(scaled_data)
            return preds
        
    except Exception as e:
         raise CustomException(e,sys)

class custom_data:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_1= {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_1)
        
        except Exception as e:
            raise CustomException(e,sys) 