from setuptools import find_packages,setup
from typing import List

def pack(file_path:str)->list:
    '''
    This Function returns the list of packages within the requirements file
    '''
    with open(file_path,"r") as file_obj:
        package=file_obj.readlines()
        return [i.replace("\n","") for i in package if i not in "-e ."]



setup(
    name="End to End ML Pipeline",
    version="0.0.0.0",
    author="Deepak S Alur",
    author_email="deepakalur4@gmail.com",
    packages=find_packages(),
    install_requires=pack("requirements.txt")
)
