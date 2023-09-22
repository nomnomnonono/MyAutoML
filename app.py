import glob
import json
import os
import subprocess
from io import BytesIO
from subprocess import PIPE

import streamlit as st
from app_utils import get_dataset_list, get_model_list, parameter_selection, upload
from components.train.utils import get_metric_list
from dotenv import load_dotenv
from google.cloud import storage
from mlflow_utils import update_mlflow

load_dotenv(".env")


def main():
    client = storage.Client()
    bucket = client.bucket(os.environ.get("DATA_BUCKET").lstrip("gs://"))

    with st.sidebar:
        st.markdown(
            f"""
        ### Vertex AI Pipelines
        [Dashboard Link](https://console.cloud.google.com/vertex-ai/pipelines/runs?hl=ja&project={os.environ.get('GCP_PROJECT_ID')})

        ### MLflow
        #### MLflowの起動
        ```bash
        $ make mlflow
        ```
        
        #### MLflowサーバーのURL
        http://127.0.0.1:5000/
        
        #### 実験結果をMLflowに反映
        """
        )
        if st.button("Update MLflow"):
            try:
                update_mlflow()
                st.text("Update Success")
            except Exception as e:
                print(e)
                st.text("Update Failed")

    st.title("MyAutoML")
    tab1, tab2, tab3 = st.tabs(["Upload Dataset", "Train Model", "Deploy Model"])

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
            if len(local_folder_path.split("-")) != 1:
                st.text("Invalid Folder Name: Do not use '-' in the folder name")
            else:
                try:
                    upload(bucket, os.path.join("data", local_folder_path))
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
        model_name = st.selectbox(
            "Model", get_model_list(config["data_type"], config["target_task"])
        )
        main_metric = st.selectbox(
            "Main Metric", get_metric_list(config["data_type"], config["target_task"])
        )
        machine_type = st.selectbox("Machine Type", ("n1-standard-4",))
        params = parameter_selection(model_name)
        params = str(params).replace(" ", "")
        if st.button("Submit", key="submit_button", help="このボタンをクリックしてアクションを実行します"):
            try:
                command = f"poetry run python pipeline.py --dataset {dataset} --data_type {config['data_type']} --target_task {config['target_task']} --model_name {model_name} --main_metric {main_metric} --machine_type {machine_type} --params {params} --is_train"
                proc = subprocess.run(command.split(" "), stdout=PIPE, stderr=PIPE)
                if len(proc.stdout.decode("utf-8")) == 0:
                    st.text(proc.stderr.decode("utf-8"))
                else:
                    st.text(proc.stdout.decode("utf-8"))
            except Exception as e:
                print(e)
                st.text("Train Failed")

    with tab3:
        st.selectbox("Select Model to Deploy", ("select1", "select2"))


if __name__ == "__main__":
    main()
