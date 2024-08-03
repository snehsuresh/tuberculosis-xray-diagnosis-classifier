from src.components.data_ingestion import DataIngestion

from src.components.model_trainer import ModelTrainer
from src.components.model_tester import ModelTesting
import pickle
import os
import sys
import numpy as np

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

    def start_model_training(self, imagetrain, labeltrain, imagesize):
        try:
            model_trainer = ModelTrainer()
            model_trainer.initate_model_training(imagetrain, labeltrain, imagesize)
        except Exception as e:
            raise customexception(e, sys)

    def start_model_testing(self, test_images, test_labels):
        try:
            model_tester = ModelTesting()
            report = model_tester.initiate_model_testing(test_images, test_labels)
            return report
        except Exception as e:
            raise customexception(e, sys)

    # def start_model_evaluation():
    #     try:
    #         model_evaluator = ModelEvaluation()
    #         model_evaluator.initiate_model_testing(test_images, test_labels, image_size)
    #         return predicted_labels, predictions
    #     except Exception as e:
    #         raise customexception(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()

    # Ingesting
    imagetrain, imagetest, labeltrain, labeltest, imagesize = (
        pipeline.start_data_ingestion()
    )
    pipeline.start_model_training(imagetrain, labeltrain, imagesize)

    base_path = "data/ingested_data"
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, "train_images.pkl"), "wb") as f:
        pickle.dump(imagetrain, f)
    with open(os.path.join(base_path, "test_images.pkl"), "wb") as f:
        pickle.dump(imagetest, f)
    with open(os.path.join(base_path, "train_labels.pkl"), "wb") as f:
        pickle.dump(labeltrain, f)
    with open(os.path.join(base_path, "test_labels.pkl"), "wb") as f:
        pickle.dump(labeltest, f)

    # Training
    with open(os.path.join(base_path, "train_images.pkl"), "rb") as f:
        train_images = np.array(pickle.load(f))
    with open(os.path.join(base_path, "train_labels.pkl"), "rb") as f:
        train_labels = np.array(pickle.load(f))

    logging.info("Starting model training.")
    pipeline.start_model_training(train_images, train_labels, imagesize)
