from setuptools import setup,find_packages
from typing import List

REQUIREMENT_FILE_NAME='requirements.txt'
HYPEN_E_DOT='-e .'

def get_requirements(file_name:str)->List[str]:
    requirements=[]
    with open(file_name)as file_obj:
        requirements=file_obj.readlines()
    requirements=[require.replace('\n',' ') for require in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='Sensor',
    version='0.0.1',
    author='Sumanth',
    author_email='kmsumanth08@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENT_FILE_NAME)
)