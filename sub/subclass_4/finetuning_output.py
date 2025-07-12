import warnings
import torch
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
import pandas as pd
import numpy as np
import sys
from contextlib import redirect_stdout

# 保存控制台内容为txt文件
# 计算权重 输入损失函数
# 使用 micro f1 保存最佳模型

warnings.filterwarnings("ignore")
import random

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义标签数量
label_num = 2
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
train_batch_size = 16

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
    labels = df.iloc[:, 17: 17+label_num].values.tolist()  # label_num = 2，选取的是17-18列
    texts = df.iloc[:, -1].tolist()  # 确保选取最后一列作为文本
    return texts, labels

# 读取训练集和验证集
train_texts, train_labels = read_tsv('train_data_aug.tsv')
val_texts, val_labels = read_tsv('val_data.tsv')

# 使用 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')
model = BertForSequenceClassification.from_pretrained(
        'Chinese-MentalBERT', num_labels=label_num
    )
# 创建训练集和验证集的数据集对象
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)


# 将控制台输出内容保存为txt文件
output_file = "output_finetuning.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        for learning_rate in learning_rates:
            # 设置训练参数
            num_epochs = 40  # 训练轮数
            best_f1 = 0  # 最佳 F1 分数
            patience = 10  # 早停策略的耐心值，即多少个 epoch 没有改进后停止
            no_improvement_count = 0  # 没有改进的 epoch 计数器
            best_val_loss = float('inf')

            # 更新模型名字
            best_model_path = f"best_subclass_4_lr={learning_rate}_batch={train_batch_size}.pt"
            print("*************************************************************************")
            print(f"当前模型: {best_model_path}")
            print("*************************************************************************")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            optimizer = AdamW(model.parameters(), lr=learning_rate)

            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            for epoch in range(num_epochs):
                if no_improvement_count >= patience:
                    print("Early stopping triggered. Stopping training.")
                    break

                model.train()
                train_loss = 0
                for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                print(f"Average Training Loss: {train_loss / len(train_loader)}")

                model.eval()
                val_loss = 0
                predict = np.zeros((0, label_num), dtype=np.int32)
                gt = np.zeros((0, label_num), dtype=np.int32)

                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    with torch.no_grad():
                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                        loss = loss_fn(logits, labels)
                        val_loss += loss.item()

                        logits_np = logits.cpu().numpy()
                        predictions = np.where(logits_np >= 0.5, 1, 0)
                        predict = np.concatenate((predict, predictions))
                        gt = np.concatenate((gt, labels.cpu().numpy()))

                val_loss /= len(val_loader)
                print(f"Validation Loss: {val_loss}")

                metrics = calculate_metrics(gt, predict)
                print(f"Epoch {epoch + 1} :\n")
                print("Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['micro']['precision'],
                                                                                    metrics['micro']['recall'],
                                                                                    metrics['micro']['f1']))
                print("Macro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['macro']['precision'],
                                                                                    metrics['macro']['recall'],
                                                                                    metrics['macro']['f1']))
                print("Weighted: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['weighted']['precision'],
                                                                                       metrics['weighted']['recall'],
                                                                                       metrics['weighted']['f1']))

                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     no_improvement_count = 0
                #     torch.save(model.state_dict(), best_model_path)
                #     print(f"New best model saved at epoch {epoch + 1} with Validation Loss: {val_loss}")
                if metrics['micro']['f1'] > best_f1:
                    best_f1 = metrics['micro']['f1']
                    # 重置没有改进的计数器
                    no_improvement_count = 0
                    torch.save(model.state_dict(), best_model_path)  # 保存最佳模型状态
                    print(f"New best model saved at epoch {epoch + 1} with F1_micro: {metrics['micro']['f1']}")
                else:
                    no_improvement_count += 1
                print(f"Best F1_micro achieved: {best_f1}")
                # print(f"Best Validation Loss achieved: {best_val_loss}")

        print("训练完成")
print("输出内容已保存为: ", output_file)

