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
