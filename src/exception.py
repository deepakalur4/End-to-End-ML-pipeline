import os
import sys

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message=f"There is a error in the file name {filename} and the line number {exc_tb.tb_lineno} and the error is {error}"
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error=error_message,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
    
