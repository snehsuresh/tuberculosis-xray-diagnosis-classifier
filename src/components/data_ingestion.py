import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from memory_profiler import profile
from src.logger.loging import logging
from src.exception.exception import customexception
from multiprocessing import Pool
from functools import partial
from src.utils.utils import read_and_resize_image


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
        # self.helper = Helper()

    def process_images_in_batches(self, image_paths, batch_size):
        images = []
        labels = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            with Pool() as pool:
                # Use partial to pass additional arguments to the function
                results = pool.map(
                    partial(
                        read_and_resize_image,
                        size=self.ingestion_config.imagesize,
                    ),
                    batch_paths,
                )
            images.extend(results)
            labels.extend([0 if "Normal" in path else 1 for path in batch_paths])
        return np.array(images), np.array(labels)

    def prepare_data(self, images, labels):
        images = images.reshape(
            -1, self.ingestion_config.imagesize * self.ingestion_config.imagesize
        )
        smote = SMOTE(random_state=42)
        images, labels = smote.fit_resample(images, labels)
        return (
            images.reshape(
                -1, self.ingestion_config.imagesize, self.ingestion_config.imagesize, 1
            ),
            labels,
        )

    @profile
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            normal_image_paths = [
                os.path.join(self.ingestion_config.normal_data_path, x)
                for x in os.listdir(self.ingestion_config.normal_data_path)
            ]
            tb_image_paths = [
                os.path.join(self.ingestion_config.tuberculosis_data_path, y)
                for y in os.listdir(self.ingestion_config.tuberculosis_data_path)
            ]

            normal_images, normal_labels = self.process_images_in_batches(
                normal_image_paths, batch_size=50
            )
            tb_images, tb_labels = self.process_images_in_batches(
                tb_image_paths, batch_size=20
            )

            images = np.concatenate([normal_images, tb_images])
            labels = np.concatenate([normal_labels, tb_labels])

            imagetrain, imagetest, labeltrain, labeltest = train_test_split(
                images, labels, test_size=0.3, random_state=42
            )
            imagetrain = imagetrain.astype("float32") / 255
            imagetest = imagetest.astype("float32") / 255

            imagetrain, labeltrain = self.prepare_data(imagetrain, labeltrain)

            print(imagetrain.shape)
            return (
                imagetrain,
                imagetest,
                labeltrain,
                labeltest,
                self.ingestion_config.imagesize,
            )

        except Exception as e:
            logging.info("Exception occurred while ingesting data")
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
