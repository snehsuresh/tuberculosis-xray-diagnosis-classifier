import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger.loging import logging
from src.exception.exception import customexception
from src.utils.utils import load_model_from_file
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class ModelTesterConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTesting:
    def __init__(self):
        logging.info("Testing started")
        self.config = ModelTesterConfig()

    def initiate_model_testing(self, imagetest, labeltest):
        try:
            print("TESTING DATA:")
            cnn = load_model_from_file(self.config.trained_model_file_path)

            cnn.evaluate(imagetest, labeltest, batch_size=32, verbose=2)

            logging.info("ADVANCED TESTING METRICS:")

            predictions = cnn.predict(imagetest, batch_size=32)
            predicted_labels = (predictions > 0.5).astype("int32")

            report = classification_report(labeltest, predicted_labels)
            confusion = confusion_matrix(labeltest, predicted_labels)
            logging.info(report)
            logging.info(confusion)
            report_df = pd.DataFrame(report).transpose()
            return report_df

        except Exception as e:
            raise customexception(e, sys)
