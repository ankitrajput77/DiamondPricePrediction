import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':

    # Ingestion
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    # Transformation
    data_transformation=DataTransformation()
    train,test=data_transformation.initiate_data_transformation(train_path,test_path)

    # Training
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train,test)
