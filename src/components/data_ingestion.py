import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import gc  # Import garbage collection module
import multiprocessing  # Import multiprocessing module

# from memory_profiler import profile
from src.logger.loging import logging
from src.exception.exception import customexception
from multiprocessing import Pool
from functools import partial
from src.utils.utils import read_and_resize_image

# docker system prune -f && docker compose build && docker compose up -d


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
    imagesize = 128


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # self.helper = Helper()

    def process_images_in_batches(self, image_paths, batch_size):
        all_images = []
        all_labels = []
        num_threads = multiprocessing.cpu_count()  # Get the number of CPU cores

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            logging.info(
                f"Starting processing batch {i // batch_size + 1} with {num_threads} threads"
            )
            with Pool() as pool:
                results = pool.map(
                    partial(
                        read_and_resize_image,
                        size=self.ingestion_config.imagesize,
                    ),
                    batch_paths,
                )
            batch_images = np.array(results, dtype=np.float32)
            batch_labels = np.array(
                [0 if "Normal" in path else 1 for path in batch_paths], dtype=np.uint8
            )

            all_images.extend(batch_images)
            all_labels.extend(batch_labels)

            # Clear batch-specific data to free memory
            del batch_images
            del batch_labels
            gc.collect()
            logging.info(f"Completed processing batch {i // batch_size + 1}")

        return np.array(all_images, dtype=np.float32), np.array(
            all_labels, dtype=np.uint8
        )

    def prepare_data(self, images, labels):
        logging.info("inside prepare_data")
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

    # @profile
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Gather paths to the images
            normal_image_paths = [
                os.path.join(self.ingestion_config.normal_data_path, x)
                for x in os.listdir(self.ingestion_config.normal_data_path)
            ]
            tb_image_paths = [
                os.path.join(self.ingestion_config.tuberculosis_data_path, y)
                for y in os.listdir(self.ingestion_config.tuberculosis_data_path)
            ]

            logging.info(f"Found {len(normal_image_paths)} normal images")
            logging.info(f"Found {len(tb_image_paths)} tuberculosis images")

            # Process normal images in batches
            logging.info("Processing normal images in batches")
            normal_images, normal_labels = self.process_images_in_batches(
                normal_image_paths, batch_size=200
            )
            logging.info(
                f"Processed {len(normal_images)} normal images with {len(normal_labels)} labels"
            )

            # Process tuberculosis images in batches
            logging.info("Processing tuberculosis images in batches")
            tb_images, tb_labels = self.process_images_in_batches(
                tb_image_paths, batch_size=100
            )
            logging.info(
                f"Processed {len(tb_images)} tuberculosis images with {len(tb_labels)} labels"
            )

            # Concatenate images and labels
            logging.info("Concatenating normal and tuberculosis images and labels")
            images = np.concatenate([normal_images, tb_images])
            labels = np.concatenate([normal_labels, tb_labels])
            logging.info(f"Concatenated images shape: {images.shape}")
            logging.info(f"Concatenated labels shape: {labels.shape}")

            # Clear temporary data
            del normal_images
            del normal_labels
            del tb_images
            del tb_labels
            gc.collect()

            # Split data into training and testing sets
            logging.info("Splitting data into training and testing sets")
            imagetrain, imagetest, labeltrain, labeltest = train_test_split(
                images, labels, test_size=0.3, random_state=42
            )
            logging.info(f"Training images shape: {imagetrain.shape}")
            logging.info(f"Testing images shape: {imagetest.shape}")

            # Normalize images
            logging.info("Normalizing images")
            imagetrain = imagetrain.astype("float32") / 255
            imagetest = imagetest.astype("float32") / 255

            # Prepare data
            logging.info("Preparing data")
            imagetrain, labeltrain = self.prepare_data(imagetrain, labeltrain)
            logging.info(f"Prepared training data shape: {imagetrain.shape}")

            return (
                imagetrain,
                imagetest,
                labeltrain,
                labeltest,
                self.ingestion_config.imagesize,
            )

        except Exception as e:
            logging.error("Exception occurred while ingesting data", exc_info=True)
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
