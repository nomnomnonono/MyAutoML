import os
import streamlit as st
import glob

from google.cloud import storage
from dotenv import load_dotenv
import json
from io import BytesIO


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
            return ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier", "LightGBMClassifier"]
        elif target_task == "regression":
            return ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor", "LGMBRegressor"]
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
            return ["ResNet18", "ResNet34", "ResNet50", "EfficientNet", "EfficientNetV2", "MobileNetV2", "MobileNetV3"]
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
        local_folder_path = st.selectbox("Select Dataset to Upload", paths)
        if st.button("Click on the button to upload", key="upload_button", help='このボタンをクリックしてアクションを実行します'):
            try:
                upload(bucket, local_folder_path)
                st.text("Upload Success")
            except Exception as e:
                print(e)
                st.text("Upload Failed")

        st.markdown("""
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
        ##### text
        ##### image
        """)

    with tab2:
        dataset = st.selectbox("Select Dataset", get_dataset_list(bucket))

        blob = bucket.blob(os.path.join(dataset, "config.json"))
        content = blob.download_as_bytes()
        config = json.load(BytesIO(content))
        st.selectbox("Data Type", (config["data_type"],))
        st.selectbox("Target Task", (config["target_task"],))
        model = st.selectbox("Model", get_model_list(config["data_type"], config["target_task"]))
        main_metric = st.selectbox("Main Metric", get_metric_list(config["data_type"], config["target_task"]))
        sub_metric = st.multiselect("Sub Metric", get_metric_list(config["data_type"], config["target_task"]))
        if st.button("Submit", key='submit_button', help='このボタンをクリックしてアクションを実行します'):
            print("submit")

    with tab3:
        st.selectbox("selectbox", ("select1", "select2"))


if __name__ == "__main__":
    load_dotenv(".env")
    main()
