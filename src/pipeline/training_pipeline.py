from src.components.data_ingestion import DataIngestion

from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_testing import ModelTesting

# import os
import sys

# from src.logger import loging
from src.exception.exception import customexception
from src.logger.loging import logging

# import pandas as pd


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            (imagetrain, imagetest, labeltrain, labeltest, imagesize) = (
                data_ingestion.initiate_data_ingestion()
            )
            logging.info("Completed all processing. Proceeding to push to xcom")
            return imagetrain, imagetest, labeltrain, labeltest, imagesize
        except Exception as e:
            raise customexception(e, sys)

    def start_model_training(
        self, imagetrain, imagetest, labeltrain, labeltest, imagesize
    ):
        try:
            model_trainer = ModelTrainer()
            model_trainer.initate_model_training(
                imagetrain, imagetest, labeltrain, labeltest, imagesize
            )
        except Exception as e:
            raise customexception(e, sys)

    def start_model_testing(
        self,
    ):
        try:
            model_tester = ModelTesting()
            model_tester.initiate_model_testing()
        except Exception as e:
            raise customexception(e, sys)
