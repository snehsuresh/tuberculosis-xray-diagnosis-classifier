from __future__ import annotations
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.training_pipeline import TrainingPipeline
import numpy as np
from src.logger.loging import logging
import pickle
import os

training_pipeline = TrainingPipeline()

with DAG(
    "xray_training_pipeline",
    default_args={"retries": 2},
    description="X-ray diagnosis training pipeline",
    # here you can test based on hour or mints but make sure here you container is up and running
    schedule="@weekly",
    start_date=pendulum.datetime(2024, 7, 26, tz="UTC"),
    catchup=False,
    tags=["deep_learning ", "classification", "tuberculosis"],
) as dag:

    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        imagetrain, imagetest, labeltrain, labeltest, imagesize = (
            training_pipeline.start_data_ingestion()
        )
        logging.info("Proceeding to save data to files")
        base_path = "data/ingested_data"
        with open(os.path.join(base_path, "train_images.pkl"), "wb") as f:
            pickle.dump(imagetrain, f)
        with open(os.path.join(base_path, "test_images.pkl"), "wb") as f:
            pickle.dump(imagetest, f)
        with open(os.path.join(base_path, "train_labels.pkl"), "wb") as f:
            pickle.dump(labeltrain, f)
        with open(os.path.join(base_path, "test_labels.pkl"), "wb") as f:
            pickle.dump(labeltest, f)
        # Push references to XCom
        logging.info("Proceeding to push XCom references")
        ti.xcom_push(
            "data_ingestion_artifact",
            {
                "train_images_path": os.path.join(base_path, "train_images.pkl"),
                "test_images_path": os.path.join(base_path, "test_images.pkl"),
                "train_labels_path": os.path.join(base_path, "train_labels.pkl"),
                "test_labels_path": os.path.join(base_path, "test_labels.pkl"),
                "image_size": imagesize,
            },
        )
        logging.info("Completed XCom push")

    def model_trainer(**kwargs):
        ti = kwargs["ti"]

        # Pull the XCom data
        data_ingestion_artifact = ti.xcom_pull(
            task_ids="data_ingestion", key="data_ingestion_artifact"
        )

        logging.info("Pulled XCom references successful!")
        # Extract file paths and other metadata
        train_images_path = data_ingestion_artifact["train_images_path"]
        test_images_path = data_ingestion_artifact["test_images_path"]
        train_labels_path = data_ingestion_artifact["train_labels_path"]
        test_labels_path = data_ingestion_artifact["test_labels_path"]
        image_size = data_ingestion_artifact["image_size"]

        # Load data from files
        with open(train_images_path, "rb") as f:
            train_images = np.array(pickle.load(f))

        with open(test_images_path, "rb") as f:
            test_images = np.array(pickle.load(f))

        with open(train_labels_path, "rb") as f:
            train_labels = np.array(pickle.load(f))

        with open(test_labels_path, "rb") as f:
            test_labels = np.array(pickle.load(f))

        logging.info("Starting model training.")
        training_pipeline.start_model_training(
            train_images, test_images, train_labels, test_labels, image_size
        )

        logging.info("Model training started.")

    # def push_data_to_s3(**kwargs):
    #     bucket_name = "reposatiory_name"
    #     artifact_folder = "/app/artifacts"
    #     # os.system(f"aws s3 sync {artifact_folder} s3:/{bucket_name}/artifact")

    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Ingestion task
    this task creates a train and test file.
    """
    )

    # data_transform_task = PythonOperator(
    #     task_id="data_transformation",
    #     python_callable=data_transformations,
    # )

    # data_transform_task.doc_md = dedent(
    #     """\
    # #### Transformation task
    # this task performs the transformation
    # """
    # )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### model trainer task
    this task perform training
    """
    )

    # push_data_to_s3_task = PythonOperator(
    #     task_id="push_data_to_s3", python_callable=push_data_to_s3
    # )


# data_ingestion_task  >> model_trainer_task >> push_data_to_s3_task
data_ingestion_task >> model_trainer_task
