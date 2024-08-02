from src.logger.loging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass

# Importing the necessary libraries
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from src.utils.utils import save_object, evaluate_model
from keras.callbacks import ReduceLROnPlateau
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.logger.loging import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    imagesize = 128


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(
        self, imagetrain, imagetest, labeltrain, labeltest, imagesize
    ):
        try:
            logging.info("Training Initiated")
            # The CNN model has 3 convolutional layers, each followed by pooling to summarize the features found by the layer, starting with 16 and multiplying by 2 each time for computational efficiency, as bits are structured in powers of 2. 3x3 filters and ReLU activation used.
            cnn = keras.Sequential(
                [
                    # Input layer, same shape as all the images (256x256x1):
                    keras.Input(
                        shape=(
                            self.model_trainer_config.imagesize,
                            self.model_trainer_config.imagesize,
                            1,
                        )
                    ),
                    # 1st convolutional layer:
                    Conv2D(16, (3, 3), activation="relu"),
                    MaxPooling2D((2, 2)),
                    # 2nd convolutional layer:
                    Conv2D(32, (3, 3), activation="relu"),
                    MaxPooling2D((2, 2)),
                    # 3rd convolutional layer:
                    Conv2D(64, (3, 3), activation="relu"),
                    MaxPooling2D((2, 2)),
                    # Flattening layer for the dense layers:
                    Flatten(),
                    # 1st dense layer following the convolutional layers:
                    Dense(64, activation="relu"),
                    # Dropout layer with heavy dropout rate to avoid overfitting in the large-ish dataset
                    Dropout(0.5),
                    # Output layer that squeezes each image to either 0 or 1 with sigmoid activation
                    Dense(1, activation="sigmoid"),
                ]
            )
            # Compiling the model with parameters best suited for the task at hand:
            cnn.compile(
                loss="binary_crossentropy",  # Best for binary classification
                optimizer=keras.optimizers.Adam(
                    learning_rate=0.001
                ),  # Good starting LR for dataset of this size
                metrics=["accuracy"],  # Looking for accuracy
            )
            # Fitting the model, with the ReduceLROnPlateau callback added to it to reduce the learning rate to take smaller steps in increasing the accuracy whenever the learning rate plateaus (goes in the wrong direction)
            # Doing this with patience=1, meaning it will perform this if it even plateaus for one epoch, since only 10 epochs are used
            # factor=0.1 means that for every time the learning rate is reduced, it is reduced by a factor of 0.1 - it also won't go lower than 0.00001

            reduce_lr = ReduceLROnPlateau(
                monitor="accuracy", factor=0.1, patience=1, min_lr=0.00001, verbose=1
            )

            # Fitting the model w/ the callback. ON VS CODE, batch size of 16 makes each epoch take around a minute in this case w/ good accuracy, making the whole training process 10 min, but on Kaggle it should take longer due to less computational resources:
            cnn.fit(
                imagetrain,
                labeltrain,
                batch_size=16,
                epochs=10,
                verbose=2,
                callbacks=[reduce_lr],
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=cnn,
            )
            logging.info("Training Ended")
        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise customexception(e, sys)
