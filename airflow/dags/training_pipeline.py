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

    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(
            task_ids="data_ingestion", key="data_ingestion_artifact"
        )
        train_arr, test_arr = training_pipeline.start_data_transformation(
            data_ingestion_artifact["train_data_path"],
            data_ingestion_artifact["test_data_path"],
        )
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        ti.xcom_push(
            "data_transformations_artifcat",
            {"train_arr": train_arr, "test_arr": test_arr},
        )

    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(
            task_ids="data_ingestion", key="data_ingestion_artifact"
        )
        train_images = np.array(data_ingestion_artifact["train_images"])
        test_images = np.array(data_ingestion_artifact["test_images"])
        train_labels = np.array(data_ingestion_artifact["train_labels"])
        test_labels = np.array(data_ingestion_artifact["test_labels"])
        image_size = data_ingestion_artifact["image_size"]
        training_pipeline.start_model_training(
            train_images, test_images, train_labels, test_labels, image_size
        )

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
