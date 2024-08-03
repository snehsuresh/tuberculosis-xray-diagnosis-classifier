import os
import sys
import pickle
from src.logger.loging import logging
from src.exception.exception import customexception
import cv2 as cv
from sklearn.metrics import r2_score
import tensorflow as tf


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            # train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception occured during model training")
        raise customexception(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception Occured in load_object function utils")
        raise customexception(e, sys)


def read_and_resize_image(image_path, size):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    return cv.resize(image, (size, size))


def load_model_from_file(file_path):
    return tf.keras.models.load_model(file_path)
