from typing import List
from setuptools import find_packages, setup

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_ptr:
        requirements=file_ptr.readlines()
        requirements=[req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Ankit Rajput',
    author_email='rajputankit72106@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages() 
)