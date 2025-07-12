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
label_num = 2
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
train_batch_size = 16

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')
model = BertForSequenceClassification.from_pretrained(
    'Chinese-MentalBERT',
    num_labels=label_num)

# 计算模型评估指标
def calculate_metrics(true_labels, predicted_labels):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
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

def calculate_evaluation_per_class(prediction, true_label):
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

def metric_big_class(true_labels, predicted_labels):
    predict_pre_list, label_pre_list = [], []
    # 统计总体大类的结果
    for sample_id in range(len(true_labels)):  # 按行样本遍历
        pre_result, label_result = False, False
        # 每一类的最终预测结果是所有子类结果的 按位与
        for sub_class in range(true_labels.shape[1]):  # 按列类别遍历，只要有一个1，结果就是1
            pre_result = pre_result or predicted_labels[sample_id][sub_class]
            label_result = label_result or true_labels[sample_id][sub_class]

        predict_pre_list.append(int(pre_result))
        label_pre_list.append(int(label_result))

    recall = recall_score(label_pre_list, predict_pre_list, zero_division=0)
    precision = precision_score(label_pre_list, predict_pre_list, zero_division=0)
    f1 = f1_score(label_pre_list, predict_pre_list, zero_division=0)

    return recall, precision, f1

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
    labels = df.iloc[:, 15: 15+label_num].values.tolist()  # label_num = 2，选取的是15-16列
    texts = df.iloc[:, -1].tolist()  # 确保选取最后一列作为文本
    return texts, labels

# 加载数据
val_texts, val_labels = read_tsv('test_data.tsv')
# 创建数据集和数据加载器
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=8)

all_best_f1_micro = 0

# 将控制台输出内容保存为txt文件
output_file = "output_evaluate.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        for learning_rate in learning_rates:
            best_model_path = f"best_subclass_3_lr={learning_rate}_batch={train_batch_size}.pt"
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
                    predictions = np.where(logits_np >= 0.5, 1, 0)
                    predict = np.concatenate((predict, predictions))
                    gt = np.concatenate((gt, labels.cpu().numpy()))

            # np.savetxt('predict_ernie.tsv', predict.astype(int), delimiter='\t', fmt='%d')
            # np.savetxt('gt_ernie.tsv', gt.astype(int), delimiter='\t', fmt='%d')

            metrics = calculate_metrics(gt, predict)

            if metrics['micro']['f1'] > all_best_f1_micro:
                all_best_f1_micro = metrics['micro']['f1']
                # 保存预测的结果
                np.savetxt('predict_sub3.tsv', predict.astype(int), delimiter='\t', fmt='%d')
                np.savetxt('gt_sub3.tsv', gt.astype(int), delimiter='\t', fmt='%d')
                print(f"{learning_rate} 模型的真实结果gt和预测结果predict已保存！")

            print(f"标签{label_num}的总体测试结果: \n")
            print("Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['micro']['precision'],
                                                                                metrics['micro']['recall'],
                                                                                metrics['micro']['f1']))
            print("Macro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['macro']['precision'],
                                                                                metrics['macro']['recall'],
                                                                                metrics['macro']['f1']))
            print("Weighted: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['weighted']['precision'],
                                                                                   metrics['weighted']['recall'],
                                                                                   metrics['weighted']['f1']))

            per_class_result = calculate_evaluation_per_class(predict, gt)
            print("分别统计每一类的指标:")
            for key, value in per_class_result.items():
                # print(f"{key}: Precision: {value['precision']}, Recall: {value['recall']}, F1: {value['f1']}")
                print(
                    f"{key}: Precision: {value['precision']:.4f}, Recall: {value['recall']:.4f}, F1: {value['f1']:.4f}")

            big_precision, big_recall, big_f1 = metric_big_class(gt, predict)
            print("统计这一大类的指标:")
            print(f"big_precision: {big_precision:.4f}, big_recall: {big_recall:.4f}, big_f1: {big_f1:.4f}")

print("输出内容已保存为: ", output_file)