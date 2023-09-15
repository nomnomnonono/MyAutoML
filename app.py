import glob
import json
import os
import subprocess
from io import BytesIO
from subprocess import PIPE

import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage


def upload(bucket, local_folder_path):
    file_paths = []
    for dirpath, _, filenames in os.walk(local_folder_path):
        for filename in filenames:
            dirpath = "/".join(dirpath.split("/")[2:])
            file_paths.append(os.path.join(dirpath, filename))

    destination_folder_name = os.path.split(local_folder_path)[-1]
    for file_path in file_paths:
        destination_blob_name = os.path.join(destination_folder_name, file_path)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(os.path.join(local_folder_path, file_path))


def get_dataset_list(bucket):
    blobs = bucket.list_blobs()
    return list(set([blob.name.split("/")[0] for blob in blobs]))


def get_model_list(data_type, target_task):
    if data_type == "table":
        if target_task == "classification":
            return [
                "LogisticRegression",
                "RandomForestClassifier",
                "XGBoostClassifier",
                "LightGBMClassifier",
            ]
        elif target_task == "regression":
            return [
                "LinearRegression",
                "RandomForestRegressor",
                "XGBoostRegressor",
                "LGMBRegressor",
            ]
        else:
            return []
    elif data_type == "text":
        if target_task == "classification":
            return ["TF-IDF + LogisticRegression", "BERT", "RoBERTa", "DEBERTa"]
        elif target_task == "summarization":
            return ["T5", "BertSum"]
        else:
            return []
    elif data_type == "image":
        if target_task == "classification":
            return [
                "ResNet18",
                "ResNet34",
                "ResNet50",
                "EfficientNet",
                "EfficientNetV2",
                "MobileNetV2",
                "MobileNetV3",
            ]
        elif target_task == "detection":
            return ["Faster R-CNN", "SSD", "YOLOv5"]
        elif target_task == "semantic-segmentation":
            return ["U-Net", "DeepLabV3", "FCN", "LRASPP"]
        else:
            return []
    else:
        raise ValueError("data_type must be table, text or image")


def get_metric_list(data_type, target_task):
    if data_type == "table":
        if target_task == "classification":
            return ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        elif target_task == "regression":
            return ["MSE", "RMSE", "MAE", "R2"]
        else:
            return []
    elif data_type == "text":
        if target_task == "classification":
            return ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        elif target_task == "summarization":
            return ["BLEU", "ROUGE", "BERTScore", "METEOR"]
        else:
            return []
    elif data_type == "image":
        if target_task == "classification":
            return ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        elif target_task == "detection":
            return ["mAP", "IoU"]
        elif target_task == "semantic-segmentation":
            return ["mIoU", "IoU", "Dice", "Jaccard"]
        else:
            return []
    else:
        raise ValueError("data_type must be table, text or image")


def main():
    client = storage.Client()
    bucket = client.bucket(os.environ.get("DATA_BUCKET").lstrip("gs://"))

    st.title("MyAutoML")
    tab1, tab2, tab3 = st.tabs(["Upload Dataset", "Train/Eval Model", "Deploy Model"])

    with tab1:
        paths = glob.glob("data/*")
        local_folder_path = st.selectbox(
            "Select Dataset to Upload", [os.path.split(path)[-1] for path in paths]
        )
        if st.button(
            "Click on the button to upload",
            key="upload_button",
            help="このボタンをクリックしてアクションを実行します",
        ):
            try:
                upload(bucket, local_folder_path)
                st.text("Upload Success")
            except Exception as e:
                print(e)
                st.text("Upload Failed")

        st.markdown(
            """
        #### フォルダ構造
        ```bash
        data/
        ├── xxx
        │   ├── train/
        │   │   ├── xxx
        │   ├── valid/
        │   │   ├── xxx
        │   ├── test/
        │   │   ├── xxx
        │   └── config.json
        ├── yyy
        ...
        ```
        
        #### config.jsonの内容
        ```json
        data_type: text/image/table  # データ種類
        target_task: classification/regression/summarization/detection/semantic-segmentation  # タスク種類
        ```
                    
        #### タスクごとのフォルダ構造
        ##### table
        ```bash
        data/
        ├── xxx
        │   ├── train/
        │   │   └── *.csv
        │   ├── valid/
        │   │   └── *.csv
        │   ├── test/
        │   │   └── *.csv
        │   └── config.json
        ├── yyy
        ...
        ```
        ターゲットラベルのカラム名は`target`とする。

        ##### text
        ```bash
        data/
        ├── xxx
        │   ├── train/
        │   │   └── *.csv
        │   ├── valid/
        │   │   └── *.csv
        │   ├── test/
        │   │   └── *.csv
        │   └── config.json
        ├── yyy
        ...
        ```
        ターゲットラベルのカラム名は`target`とする。

        ##### image
        ```bash
        data/
        ├── xxx
        │   ├── train/
        │   │   ├── images/
        │   │   │   ├── *.png
        │   │   │   ...
        │   │   └── *.csv
        │   ├── valid/
        │   │   ├── images/
        │   │   │   ├── *.png
        │   │   │   ...
        │   │   └── *.csv
        │   ├── test/
        │   │   ├── images/
        │   │   │   ├── *.png
        │   │   │   ...
        │   │   └── *.csv
        │   └── config.json
        ├── yyy
        ...
        ```
        ターゲットラベルのカラム名は`target`とする。
        """
        )

    with tab2:
        dataset = st.selectbox("Select Dataset", get_dataset_list(bucket))

        blob = bucket.blob(os.path.join(dataset, "config.json"))
        content = blob.download_as_bytes()
        config = json.load(BytesIO(content))
        st.selectbox("Data Type", (config["data_type"],))
        st.selectbox("Target Task", (config["target_task"],))
        model = st.selectbox(
            "Model", get_model_list(config["data_type"], config["target_task"])
        )
        main_metric = st.selectbox(
            "Main Metric", get_metric_list(config["data_type"], config["target_task"])
        )
        sub_metric = st.multiselect(
            "Sub Metric", get_metric_list(config["data_type"], config["target_task"])
        )
        machine_type = st.selectbox("Machine Type", ("n1-standard-4",))
        if st.button("Submit", key="submit_button", help="このボタンをクリックしてアクションを実行します"):
            try:
                command = f"python pipeline.py --dataset {dataset} --data_type {config['data_type']} --target_task {config['target_task']} \
                    --model {model} --main_metric {main_metric} --sub_metric"
                for metric in sub_metric:
                    command += f" {metric}"
                command += f" --machine_type {machine_type} --is_train"
                proc = subprocess.run(command.split(" "), stdout=PIPE, stderr=PIPE)
                print(proc.stdout.decode("utf-8").split("\n"))
                st.text("Pipeline Execution Successed")
            except Exception as e:
                print(e)
                st.text("Train Failed")

    with tab3:
        st.selectbox("Select Model to Deploy", ("select1", "select2"))


if __name__ == "__main__":
    load_dotenv(".env")
    main()
