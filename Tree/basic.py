import torch
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 计算模型总体评估指标
def calculate_metrics(true_labels, predictions):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predictions, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    return {
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }

# 统计子类的每一类指标
def calculate_evaluation_per_class(true_label, prediction):
    results = {}
    num_classes = true_label.shape[1]
    for class_index in range(num_classes):
        true_label_class = true_label[:, class_index]
        prediction_class = prediction[:, class_index]
        recall = recall_score(true_label_class, prediction_class, zero_division=0)
        precision = precision_score(true_label_class, prediction_class, zero_division=0)
        f1 = f1_score(true_label_class, prediction_class, zero_division=0)
        results[class_index] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    return results

# 统计4个大类的每一类的指标
def metric_big_per_class(true_label, prediction):
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]
    total = 0

    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        total += split_list[big_clss_id]

        predict_pre_list = np.array(predict_pre_list, dtype=np.int32)
        label_pre_list = np.array(label_pre_list, dtype=np.int32)

        recall = recall_score(label_pre_list, predict_pre_list, zero_division=0)
        precision = precision_score(label_pre_list, predict_pre_list, zero_division=0)
        f1 = f1_score(label_pre_list, predict_pre_list, zero_division=0)

        results[big_clss_id] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    return results

# 统计4大类的总体结果
def metric_big_class(true_label, prediction):
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]
    total = 0
    # 初始化用于存储拼接结果的矩阵
    prediction_4 = np.zeros((size[0], 0), dtype=np.int32)
    true_label_4 = np.zeros((size[0], 0), dtype=np.int32)

    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        total += split_list[big_clss_id]

        # 转换为NumPy数组，并将一维列表转换为一列
        predict_pre_list = np.array(predict_pre_list, dtype=np.int32).reshape(-1, 1)
        label_pre_list = np.array(label_pre_list, dtype=np.int32).reshape(-1, 1)

        # 按列拼接到现有矩阵
        prediction_4 = np.hstack((prediction_4, predict_pre_list))
        true_label_4 = np.hstack((true_label_4, label_pre_list))

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_label_4, prediction_4,
                                                                                 average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_label_4, prediction_4,
                                                                                 average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_label_4, prediction_4,
                                                                                          average='weighted')
    return {
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }


# 统计23类的总体的指标
def metric_23_class(true_label, prediction):
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]
    total = 0
    # 初始化用于存储拼接结果的矩阵
    prediction_23 = np.array(prediction, dtype=np.int32)
    true_label_23 = np.array(true_label, dtype=np.int32)

    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        total += split_list[big_clss_id]

        # 转换为NumPy数组，并将一维列表转换为一列
        predict_pre_list = np.array(predict_pre_list, dtype=np.int32).reshape(-1, 1)
        label_pre_list = np.array(label_pre_list, dtype=np.int32).reshape(-1, 1)

        # 按列拼接到现有矩阵
        prediction_23 = np.hstack((prediction_23, predict_pre_list))
        true_label_23 = np.hstack((true_label_23, label_pre_list))


    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_label_23, prediction_23, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_label_23, prediction_23, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_label_23, prediction_23, average='weighted')
    return {
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }



class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        label = [float(x) for x in label]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def read_tsv(file_path, label_num):
    df = pd.read_csv(file_path, delimiter='\t', header=0)
    labels = df.iloc[:, :label_num].values.tolist()
    texts = df.iloc[:, -1].tolist()  # 确保选取最后一列作为文本
    return texts, labels