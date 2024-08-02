import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger.loging import logging
from src.exception.exception import customexception


class ModelTesting:
    def __init__(self):
        logging.info("Testing started")

    def initiate_model_testing(self):
        try:
            print("TESTING DATA:")

            cnn.evaluate(imagetest, labeltest, batch_size=32, verbose=2)

            print("ADVANCED TESTING METRICS:")
            from sklearn.metrics import classification_report, confusion_matrix

            predictions = cnn.predict(imagetest, batch_size=32)
            predicted_labels = (predictions > 0.5).astype("int32")
            print(classification_report(labeltest, predicted_labels))
            print(confusion_matrix(labeltest, predicted_labels))

        except Exception as e:
            raise customexception(e, sys)
