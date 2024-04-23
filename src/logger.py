import logging
from datetime  import datetime
import os
from src.exception import CustomException
import sys

log_file_name=f"{datetime.now().strftime("%m_%d_%y_%H_%M_%S")}.log"
log_path=os.path.join(os.getcwd(),"Logs",log_file_name)
os.makedirs(log_path,exist_ok=True)

log_file_path_name=os.path.join(log_path,log_file_name)


logging.basicConfig(
    filename=log_file_path_name,level=logging.INFO
)

