from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from data_transformation import * 

@dataclass
class data_ingestion_config:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

class data_ingestion:
    def __init__(self):
        self.data_ingestion=data_ingestion_config()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started and entered the block of data ingestion")
            df=pd.read_csv(r"src\notebooks\data\stud.csv")
            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False,header=True)

            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.data_ingestion.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion.test_data_path,index=False,header=True)

            return(
                self.data_ingestion.train_data_path,self.data_ingestion.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    data_ingestion_obj=data_ingestion()
    train_data_path,test_data_path=data_ingestion_obj.initiate_data_ingestion()
    data_transformation_obj=data_transformation()
    train_ar,test_arr,_=data_transformation_obj.initiate_data_transoformation(train_data_path,test_data_path)








