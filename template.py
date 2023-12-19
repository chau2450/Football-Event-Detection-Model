import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# CHANGE BASED ON YOUR PROJECT NAME
project_name = "Football_Event_Detector"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/conponents/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "environment.yaml",
    "setup.py",
    "research/eda.ipynb",
    "research/pipeline_test.ipynb",
    "research/model_test.ipynb",
    "data/raw_data/about_data.ipynb",
    "data/staging_data/about_data.ipynb",
    "data/consumption_data/about_data.ipynb",
    "kaggle/main_report.ipynb",
    "kaggle/main_report.ipynb",
    "interactive_dashboards/test.ipynb",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")
