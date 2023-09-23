import os

import streamlit as st


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


def get_model_list(data_type: str, target_task: str) -> list[str]:
    if data_type == "table":
        if target_task == "classification":
            return [
                "LogisticRegression",
                "RandomForestClassifier",
                "XGBClassifier",
                "LGBMClassifier",
            ]
        elif target_task == "regression":
            return [
                "LinearRegression",
                "RandomForestRegressor",
                "XGBRegressor",
                "LGBMRegressor",
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
