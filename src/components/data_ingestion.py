import pandas as pd
import numpy as np
from src.logger.loging import logging
from src.exception.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import cv2 as cv
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    normal_data_path: str = os.path.join(
        "data/TB_Chest_Radiography_Database/",
        "Normal/",
    )
    tuberculosis_data_path: str = os.path.join(
        "data/TB_Chest_Radiography_Database/",
        "Tuberculosis/",
    )
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    images = []
    labels = []
    imagesize = 256


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            logging.info("Reading from Normal directory")
            for x in os.listdir(self.ingestion_config.normal_data_path):
                imagedir = os.path.join(self.ingestion_config.normal_data_path, x)
                image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
                image = cv.resize(
                    image,
                    (self.ingestion_config.imagesize, self.ingestion_config.imagesize),
                )
                self.ingestion_config.images.append(image)
                self.ingestion_config.labels.append(0)
            logging.info("Reading from TB directory")
            for y in os.listdir(self.ingestion_config.tuberculosis_data_path):
                imagedir = os.path.join(self.ingestion_config.tuberculosis_data_path, y)
                image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
                image = cv.resize(
                    image,
                    (self.ingestion_config.imagesize, self.ingestion_config.imagesize),
                )
                self.ingestion_config.images.append(image)
                self.ingestion_config.labels.append(1)

            images = np.array(self.ingestion_config.images)
            labels = np.array(self.ingestion_config.labels)

            # Splitting the images and labels into training and testing sets, then normalizing the values within them for computational efficiency (from 0-255 scale to 0-1 scale)
            imagetrain, imagetest, labeltrain, labeltest = train_test_split(
                images, labels, test_size=0.3, random_state=42
            )
            imagetrain = (imagetrain.astype("float32")) / 255
            imagetest = (imagetest.astype("float32")) / 255

            # Flattening the image array into 2D (making it [2940 images] x [all the pixels of the image in just one 1D array]) to be suitable for SMOTE oversampling
            imagetrain = imagetrain.reshape(
                2940,
                (
                    self.ingestion_config.imagesize.imagesize
                    * self.ingestion_config.imagesize.imagesize
                ),
            )

            # Performing oversampling
            smote = SMOTE(random_state=42)
            imagetrain, labeltrain = smote.fit_resample(imagetrain, labeltrain)

            # Unflattening the images now to use them for convolutional neural network (4914 images of 256x256 size, with 1 color channel (grayscale, as compared to RGB with 3 color channels))
            imagetrain = imagetrain.reshape(
                -1,
                self.ingestion_config.imagesize.imagesize,
                self.ingestion_config.imagesize.imagesize,
                1,
            )
            print(imagetrain.shape)
            return (
                imagetrain,
                imagetest,
                labeltrain,
                labeltest,
                self.ingestion_config.imagesize,
            )

        except Exception as e:
            logging.info()
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    obj.initiate_data_ingestion()
