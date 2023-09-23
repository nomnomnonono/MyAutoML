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


def parameter_selection(model_name: str) -> dict[str, list]:
    with st.expander("Parameter Selection"):
        st.markdown(
            "##### When you input multiple values, you should join them by a half-width single space."
        )
        # table
        if model_name == "LogisticRegression":
            C = st.text_input(
                label="C (float)  :  Inverse of regularization strength",
                value="0.1 1.0 10",
            )
            random_state = st.text_input(
                "Random State (int)  :  Random number seed", value="42"
            )
            return {
                "C": list(map(float, C.split(" "))),
                "random_state": list(map(int, random_state.split(" "))),
            }
        elif model_name == "LinearRegression":
            random_state = st.text_input(
                "Random State (int)  :  Random number seed", value="42"
            )
            return {"random_state": list(map(int, random_state.split(" ")))}
        elif model_name[:12] == "RandomForest":
            n_estimators = st.text_input(
                label="n_estimators (int)  :  ", value="10 50 100 500"
            )
            max_depth = st.multiselect(
                "max_depth (int/None)  :  ", ["None", 5, 10], default=["None"]
            )
            min_samples_split = st.text_input(
                label="min_samples_split (int)  :  ", value="2"
            )
            min_samples_leaf = st.text_input(
                label="min_samples_leaf (int)  :  ", value="1"
            )
            max_features = st.multiselect(
                "max_features (str/None)  :  ",
                ["sqrt", "log2", "None"],
                default=["sqrt"],
            )
            if model_name[12:] == "Classifier":
                criterion = st.multiselect(
                    "criterion (str)  :  ",
                    ["gini", "entropy", "log_loss"],
                    default=["gini"],
                )
            elif model_name[12:] == "Regressor":
                criterion = st.multiselect(
                    "criterion (str)  :  ",
                    ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    default=["squared_error"],
                )
            else:
                raise ValueError(f"Invalid model: {model_name}")
            return {
                "n_estimators": list(map(int, n_estimators.split(" "))),
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": list(map(int, min_samples_split.split(" "))),
                "min_samples_leaf": list(map(int, min_samples_leaf.split(" "))),
                "max_features": max_features,
            }
        elif model_name == "XGBClassifier":
            pass
        elif model_name == "XGBRegressor":
            pass
        elif model_name == "LGBMClassifier":
            pass
        elif model_name == "LGBMRegressor":
            pass
        # text
        elif model_name == "TF-IDF + LogisticRegression":
            pass
        elif model_name == "BERT":
            pass
        elif model_name == "RoBERTa":
            pass
        elif model_name == "DEBERTa":
            pass
        elif model_name == "T5":
            pass
        elif model_name == "BertSum":
            pass
        # image
        elif model_name == "ResNet18":
            pass
        elif model_name == "ResNet34":
            pass
        elif model_name == "ResNet50":
            pass
        elif model_name == "EfficientNet":
            pass
        elif model_name == "EfficientNetV2":
            pass
        elif model_name == "MobileNetV2":
            pass
        elif model_name == "MobileNetV3":
            pass
        else:
            raise ValueError(f"Invalid model: {model_name}")
