from src.exception import CustomException
from src.logger import logging
import pickle


def save_object(file_path,obj):
    with open(file_path,"wb") as file_obj:
        return pickle.dump(obj,file_obj)