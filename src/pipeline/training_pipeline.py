from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

# from src.components.model_evaluation import ModelEvaluation


# import os
import sys

# from src.logger import loging
from src.exception.exception import customexception

# import pandas as pd


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            (imagetrain, imagetest, labeltrain, labeltest, imagesize) = (
                data_ingestion.initiate_data_ingestion()
            )
            return imagetrain, imagetest, labeltrain, labeltest, imagesize
        except Exception as e:
            raise customexception(e, sys)

    # def start_data_transformation(
    #     self, imagetrain, imagetest, labeltrain, labeltest, imagesize
    # ):

    #     try:
    #         data_transformation = DataTransformation()
    #         train_arr, test_arr = data_transformation.initialize_data_transformation(
    #             imagetrain, imagetest, labeltrain, labeltest, imagesize
    #         )
    #         return train_arr, test_arr
    #     except Exception as e:
    #         raise customexception(e, sys)

    def start_model_training(imagetrain, imagetest, labeltrain, labeltest, imagesize):
        try:
            model_trainer = ModelTrainer()
            model_trainer.initate_model_training(
                imagetrain, imagetest, labeltrain, labeltest, imagesize
            )
        except Exception as e:
            raise customexception(e, sys)

    def start_trainig(self):
        try:
            imagetrain, imagetest, labeltrain, labeltest, imagesize = (
                self.start_data_ingestion()
            )
            # train_arr, test_arr = self.start_data_transformation(
            #     imagetrain, imagetest, labeltrain, labeltest, imagesize
            # )
            self.start_model_training(
                imagetrain, imagetest, labeltrain, labeltest, imagesize
            )
        except Exception as e:
            raise customexception(e, sys)
