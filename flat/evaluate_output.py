import warnings
import torch
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, ErnieForSequenceClassification, AdamW, BertForSequenceClassification
import pandas as pd
import numpy as np
import sys
from contextlib import redirect_stdout

# 保存控制台内容为txt文件
# 计算权重 输入损失函数
# 使用 micro f1 保存最佳模型

warnings.filterwarnings("ignore")

# 定义标签数量
label_num = 23
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
logits_values = [0.5]
train_batch_size = 16

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')
model = BertForSequenceClassification.from_pretrained(
    'Chinese-MentalBERT',
    num_labels=label_num)

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

def read_tsv(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=0)
    labels = df.iloc[:, :label_num].values.tolist()
    texts = df.iloc[:, -1].tolist()  # 确保选取最后一列作为文本
    return texts, labels

# 加载数据
val_texts, val_labels = read_tsv('test_data.tsv')
# 创建数据集和数据加载器
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=8)

all_best_f1_micro = 0

# 将控制台输出内容保存为txt文件
output_file = "output_evaluate_sigmoid.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        for logits_value in logits_values:
            for learning_rate in learning_rates:
                best_model_path = f"best_23_logits={logits_value}_lr={learning_rate}.pt"
                print("*************************************************************************")
                print(f"当前模型: {best_model_path}")
                print("*************************************************************************")
                model.load_state_dict(torch.load(best_model_path))
                # 设置设备
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                predict = np.zeros((0, label_num), dtype=np.int32)
                gt = np.zeros((0, label_num), dtype=np.int32)

                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    with torch.no_grad():
                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                        logits_np = logits.cpu().numpy()
                        predictions = np.where(logits_np >= logits_value, 1, 0)
                        predict = np.concatenate((predict, predictions))
                        gt = np.concatenate((gt, labels.cpu().numpy()))

                # np.savetxt('predict_ernie.tsv', predict.astype(int), delimiter='\t', fmt='%d')
                # np.savetxt('gt_ernie.tsv', gt.astype(int), delimiter='\t', fmt='%d')


                ############
                test_child_labels = gt[:, :19]
                test_parent_labels = gt[:, 19:]
                test_child_predictions = predict[:, :19]
                test_parent_predictions = predict[:, 19:]

                test_child_metrics = calculate_metrics(test_child_labels, test_child_predictions)
                test_parent_metrics = calculate_metrics(test_parent_labels, test_parent_predictions)
                test_metrics = calculate_metrics(gt, predict)

                if test_metrics['micro']['f1'] > all_best_f1_micro:
                    all_best_f1_micro = test_metrics['micro']['f1']
                    # 保存预测的结果
                    np.savetxt('gt_23.tsv', gt.astype(int), delimiter='\t', fmt='%d')
                    np.savetxt('predict_23.tsv', predict.astype(int), delimiter='\t', fmt='%d')
                    print(f"{best_model_path} 模型的真实结果gt_23和预测结果predict_23已保存！")

                print("\n统计总体的指标:")
                print("Parent Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(
                    test_parent_metrics['micro']['precision'] * 100,
                    test_parent_metrics['micro']['recall'] * 100,
                    test_parent_metrics['micro']['f1'] * 100))
                print("Child Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(
                    test_child_metrics['micro']['precision'] * 100,
                    test_child_metrics['micro']['recall'] * 100,
                    test_child_metrics['micro']['f1'] * 100))
                print("Overall Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(
                    test_metrics['micro']['precision'] * 100,
                    test_metrics['micro']['recall'] * 100,
                    test_metrics['micro']['f1'] * 100))

                parent_per_metrics = calculate_evaluation_per_class(test_parent_labels, test_parent_predictions)
                print("\n分别统计父类的指标:")
                for key, value in parent_per_metrics.items():
                    print(
                        f"{key + 1}: Precision: {value['precision'] * 100:.2f}, Recall: {value['recall'] * 100:.2f}, F1: {value['f1'] * 100:.2f}")

                child_per_metrics = calculate_evaluation_per_class(test_child_labels, test_child_predictions)
                print("\n分别统计子类的指标:")
                for key, value in child_per_metrics.items():
                    print(
                        f"{key + 1}: Precision: {value['precision'] * 100:.2f}, Recall: {value['recall'] * 100:.2f}, F1: {value['f1'] * 100:.2f}")

print("输出内容已保存为: ", output_file)


