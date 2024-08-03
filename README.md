# Project Documentation

## Overview

This project involves building and deploying a Convolutional Neural Network (CNN) for image classification using Docker and Apache Airflow. The dataset used is a Kaggle dataset of chest X-ray images for tuberculosis detection.

## Dataset

- **Source**: [Tuberculosis Chest X-ray Dataset on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **Description**: The dataset includes chest X-ray images categorized into normal and tuberculosis classes.

## Convolutional Neural Network (CNN)

### Model Architecture

The CNN used for this project consists of:

- **Input Layer**: Accepts images of size 128x128 pixels with a single channel.
- **Convolutional Layers**:
  - 16 filters with a 3x3 kernel and ReLU activation.
  - 32 filters with a 3x3 kernel and ReLU activation.
  - 64 filters with a 3x3 kernel and ReLU activation.
- **Pooling Layers**: MaxPooling2D layers are applied to reduce the spatial dimensions.
- **Flatten Layer**: Flattens the output from the convolutional layers.
- **Dense Layers**:
  - A dense layer with 64 units and ReLU activation.
  - A dropout layer with a 50% dropout rate to prevent overfitting.
  - An output layer with a single unit and sigmoid activation for binary classification.

### Model Evaluation

The model was evaluated using a classification report with the following metrics:

- **Precision**: 
  - Normal: 0.984
  - Tuberculosis: 0.935
- **Recall**:
  - Normal: 0.987
  - Tuberculosis: 0.922
- **F1-Score**:
  - Normal: 0.985
  - Tuberculosis: 0.928
- **Accuracy**: 0.975
- **Macro Average**:
  - Precision: 0.959
  - Recall: 0.954
  - F1-Score: 0.957
- **Weighted Average**:
  - Precision: 0.975
  - Recall: 0.975
  - F1-Score: 0.975

## Docker

### Docker Setup

The project uses Docker for containerization. There are two key Docker images:

- **Base Image**: Provides the foundational setup for the environment.
- **Service Images**: Separate images for Airflow and Flask applications.

#### Building Docker Images

To build the Docker images:

1. Build the base image:
   ```bash
   docker build -f Dockerfile.base -t my-base-image:latest .
2. Build and run the service images using Docker Compose:
    ```bash
    docker compose build
    docker compose up -d
    ```

### Managing Docker Containers

Volumes: Ensure that volumes are correctly mounted to persist data and share it between containers.
Copying Files: Use docker cp to copy files between Docker containers and the host system.

## Airflow

The workflow DAG (Directed Acyclic Graph) for this project includes:

Data Ingestion Task: Responsible for ingesting and preparing data.
Model Trainer Task: Trains the CNN model.
Model Testing Task: Evaluates the trained model.
The tasks are connected in sequence:
    ```bash
    data_ingestion_task >> model_trainer_task >> model_testing_task
    ```

### XCom Serialization

Airflow uses XComs (Cross-Communication) to pass data between tasks. XCom serialization issues can arise when data exceeds the default size limits or when non-serializable objects are used.

## Memory Management

### Efficient Data Processing

The data ingestion process is optimized for memory efficiency:

Batch Processing: Images are processed in batches to manage memory usage effectively.
Multiprocessing: Utilizes multiple CPU cores to speed up image processing.
Garbage Collection: Clears memory by explicitly deleting unnecessary variables and invoking garbage collection.


## Data Version Control (DVC)

DVC is used to manage dataset versions and pipeline dependencies:

Adding Dataset: The Kaggle dataset was added to DVC for version control.
Pipeline Configuration: A YAML file defines the pipeline, including dependencies and outputs such as ingested data and the trained model.