from setuptools import find_packages, setup

from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """_summary_
    This function will return the requirement in list

    Args:
        file_path (str): default : none

    Returns:
        List[str]: default : none
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [obj.replace("\n", "") for obj in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="ML_Project",
    version="0.0.1",
    author="Rahul",
    author_email="rahulkjr9435@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)