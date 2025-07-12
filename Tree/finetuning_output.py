import warnings
import torch
import basic
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, BertModel
import numpy as np
import sys
import random
from contextlib import redirect_stdout
from model import AdvancedFeatureExtractionModel

warnings.filterwarnings("ignore")

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 从 basic.py 导入参数
model_origin = basic.model_origin
save_base = basic.save_base
child_loss_select = basic.child_loss_select
parent_loss_select = basic.parent_loss_select
backbone_frozen = basic.backbone_frozen
learning_rate = basic.learning_rate
logit = basic.logit
label_num = basic.label_num
num_epochs = basic.num_epochs
patience = basic.patience
train_batch_size = basic.train_batch_size
weight_child = basic.weight_child
weight_parent = basic.weight_parent

# 读取训练集和验证集
train_texts, train_labels = basic.read_tsv('train_data.tsv', label_num)
val_texts, val_labels = basic.read_tsv('val_data.tsv', label_num)

# 转换标签为 NumPy 数组
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# 检查数据
print("Train labels shape:", train_labels.shape)
print("Train labels min/max:", train_labels.min(), train_labels.max())
print("Val labels shape:", val_labels.shape)
print("Val labels min/max:", val_labels.min(), val_labels.max())

# 使用 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT', max_length=128, truncation=True)

# 初始化自定义模型
base_model = BertModel.from_pretrained('Chinese-MentalBERT')
model = AdvancedFeatureExtractionModel(base_model, tokenizer, input_size=768, excel_file_path='decision_rules.csv')

# 根据 backbone_frozen 设置
for param in base_model.parameters():
    param.requires_grad = not backbone_frozen

# 创建训练集和验证集的数据集对象
train_dataset = basic.TextDataset(train_texts, train_labels, tokenizer)
val_dataset = basic.TextDataset(val_texts, val_labels, tokenizer)

# 保存输出到文件
best_model_path = f"{model_origin}_{save_base}_child={child_loss_select}_{weight_child}_parent={parent_loss_select}_{weight_parent}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
output_file = f"output_finetuning_{best_model_path}.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        print("*************************************************************************")
        print(f"Setting:\n model_origin: {model_origin}\n Num Epochs: {num_epochs}\n Save_base: {save_base}\n"
              f"Child Loss: {child_loss_select} weight_child: {weight_child}\n"
              f"Parent Loss: {parent_loss_select} weight_parent: {weight_parent}\n"
              f"Backbone Frozen: {backbone_frozen}\n Logits: {logit}\n Learning Rate: {learning_rate}\n Batch Size: {train_batch_size}")
        print("*************************************************************************")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_batch_size)

        best_f1 = 0
        no_improvement_count = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print("\n--------Epoch-------", epoch + 1)
            if no_improvement_count >= patience:
                print("Early stopping triggered. Stopping training.")
                break

            model.train()
            train_loss = 0
            all_train_labels = []
            all_train_predictions = []

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                child_labels = labels[:, :19]  # 多标签子类
                parent_labels = labels[:, 19:23]  # 多标签父类

                optimizer.zero_grad()
                child_probs, parent_probs, contrast_features, projected_parent = model(input_ids, attention_mask)
                probabilities = torch.cat((child_probs, parent_probs), dim=1)
                binary_outputs = (probabilities > logit).float()

                # 调试：检查概率范围
                print(f"Epoch {epoch + 1} Batch - Child probs min/max: {child_probs.min().item():.4f}, {child_probs.max().item():.4f}")
                print(f"Epoch {epoch + 1} Batch - Parent probs min/max: {parent_probs.min().item():.4f}, {parent_probs.max().item():.4f}")

                # 原始损失
                if child_loss_select == 'MSE':
                    child_loss_fn = torch.nn.MSELoss()
                elif child_loss_select == 'BCELoss':
                    child_loss_fn = torch.nn.BCELoss()
                elif child_loss_select == 'MultiLabelSoftMarginLoss':
                    child_loss_fn = torch.nn.MultiLabelSoftMarginLoss()
                else:
                    raise Exception('Undefined Child Loss!')

                if parent_loss_select == 'MSE':
                    parent_loss_fn = torch.nn.MSELoss()
                elif parent_loss_select == 'BCELoss':
                    parent_loss_fn = torch.nn.BCELoss()
                elif parent_loss_select == 'MultiLabelSoftMarginLoss':
                    parent_loss_fn = torch.nn.MultiLabelSoftMarginLoss()
                else:
                    raise Exception('Undefined Parent Loss!')

                child_train_loss = child_loss_fn(child_probs, child_labels)
                parent_train_loss = parent_loss_fn(parent_probs, parent_labels)
                base_loss = weight_child * child_train_loss + weight_parent * parent_train_loss

                # 层次化对比损失
                # 层次化对比损失
                batch_size = input_ids.size(0)
                contrast_parent_loss = 0
                contrast_child_loss = 0
                hier_loss = 0
                temperature = 0.1
                contrast_features = torch.nn.functional.normalize(contrast_features, dim=-1)
                projected_parent = torch.nn.functional.normalize(projected_parent, dim=-1)
                parent_dim = contrast_features.size(-1) // 2

                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            # Compute similarities
                            sim_parent = torch.cosine_similarity(contrast_features[i, :parent_dim].unsqueeze(0),
                                                                 contrast_features[j, :parent_dim].unsqueeze(0),
                                                                 dim=-1).item()
                            sim_child = torch.cosine_similarity(contrast_features[i, parent_dim:].unsqueeze(0),
                                                                contrast_features[j, parent_dim:].unsqueeze(0),
                                                                dim=-1).item()

                            # Compute Jaccard similarity (ensure scalar output)
                            intersection_child = (child_labels[i] * child_labels[j]).sum()
                            union_child = (
                                        child_labels[i] + child_labels[j] - (child_labels[i] * child_labels[j])).sum()
                            jaccard_child = intersection_child / union_child.clamp(min=1e-8)

                            intersection_parent = (parent_labels[i] * parent_labels[j]).sum()
                            union_parent = (parent_labels[i] + parent_labels[j] - (
                                        parent_labels[i] * parent_labels[j])).sum()
                            jaccard_parent = intersection_parent / union_parent.clamp(min=1e-8)

                            # Child node contrastive loss with strong/weak positives and negatives
                            weight_child = 1.0
                            if jaccard_child > 0.5:  # Strong positive (same subclass)
                                weight_child = 1.5
                                contrast_child_loss += weight_child * max(0, 0.5 - sim_child) ** 2
                            elif jaccard_parent > 0.5 and jaccard_child <= 0.5:  # Weak positive (same parent, different subclass)
                                weight_child = 1.0
                                contrast_child_loss += weight_child * max(0, 0.5 - sim_child) ** 2
                            elif jaccard_parent <= 0.5:  # Weak negative (different parent)
                                weight_child = 1.0
                                contrast_child_loss += weight_child * max(0, sim_child - 0.3) ** 2
                            elif sim_child > 0.7:  # Strong negative (semantically confusing)
                                weight_child = 1.5
                                contrast_child_loss += weight_child * max(0, sim_child - 0.3) ** 2

                            # Parent node contrastive loss with strong/weak positives and negatives
                            weight_parent = 1.0
                            if jaccard_parent > 0.5:  # Strong positive (same parent)
                                weight_parent = 1.5
                                contrast_parent_loss += weight_parent * max(0, 0.5 - sim_parent) ** 2
                            elif jaccard_parent <= 0.5:  # Weak negative (different parent)
                                weight_parent = 1.0
                                contrast_parent_loss += weight_parent * max(0, sim_parent - 0.3) ** 2
                            elif sim_parent > 0.7:  # Strong negative (semantically confusing)
                                weight_parent = 1.5
                                contrast_parent_loss += weight_parent * max(0, sim_parent - 0.3) ** 2

                    # Hierarchical consistency loss
                    hier_loss += torch.norm(contrast_features[i, parent_dim:] - projected_parent[i], p=2) ** 2

                # Normalize losses
                contrast_parent_loss /= (batch_size * (batch_size - 1))
                contrast_child_loss /= (batch_size * (batch_size - 1))
                hier_loss /= batch_size

                # Compute total contrastive loss
                contrast_loss = contrast_parent_loss + contrast_child_loss + 0.1 * hier_loss

                # 总损失
                total_loss = base_loss + 0.4 * contrast_loss                # total_loss = base_loss + 0.1 * contrast_parent_loss + 0.2 * contrast_child_loss + 0.1 * hier_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += total_loss.item()

                all_train_labels.append(labels.cpu().numpy())
                all_train_predictions.append(binary_outputs.cpu().numpy())

            # 计算并打印训练指标
            all_train_labels = np.concatenate(all_train_labels, axis=0)
            all_train_predictions = np.concatenate(all_train_predictions, axis=0)
            train_child_labels = all_train_labels[:, :19]
            train_parent_labels = all_train_labels[:, 19:23]
            train_child_predictions = all_train_predictions[:, :19]
            train_parent_predictions = all_train_predictions[:, 19:23]
            train_child_metrics = basic.calculate_metrics(train_child_labels, train_child_predictions)
            train_parent_metrics = basic.calculate_metrics(train_parent_labels, train_parent_predictions)
            train_metrics = basic.calculate_metrics(all_train_labels, all_train_predictions)

            print(f"\n[Epoch {epoch + 1} Training] Total Loss: {train_loss / len(train_loader):.4f}")
            print(f"Child Micro - Precision: {train_child_metrics['micro']['precision']:.4f}, "
                  f"Recall: {train_child_metrics['micro']['recall']:.4f}, "
                  f"F1: {train_child_metrics['micro']['f1']:.4f}")
            print(f"Parent Micro - Precision: {train_parent_metrics['micro']['precision']:.4f}, "
                  f"Recall: {train_parent_metrics['micro']['recall']:.4f}, "
                  f"F1: {train_parent_metrics['micro']['f1']:.4f}")
            print(f"Overall Micro - Precision: {train_metrics['micro']['precision']:.4f}, "
                  f"Recall: {train_metrics['micro']['recall']:.4f}, "
                  f"F1: {train_metrics['micro']['f1']:.4f}")

            # 验证部分
            model.eval()
            val_loss = 0
            all_val_labels = []
            all_val_predictions = []

            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                child_labels = labels[:, :19]
                parent_labels = labels[:, 19:23]

                with torch.no_grad():
                    child_probs, parent_probs, _, _ = model(input_ids, attention_mask)
                    probabilities = torch.cat((child_probs, parent_probs), dim=1)
                    binary_outputs = (probabilities > logit).float()

                    child_val_loss = child_loss_fn(child_probs, child_labels)
                    parent_val_loss = parent_loss_fn(parent_probs, parent_labels)
                    final_val_loss = weight_child * child_val_loss + weight_parent * parent_val_loss
                    val_loss += final_val_loss.item()

                    all_val_labels.append(labels.cpu().numpy())
                    all_val_predictions.append(binary_outputs.cpu().numpy())

            all_val_labels = np.concatenate(all_val_labels, axis=0)
            all_val_predictions = np.concatenate(all_val_predictions, axis=0)
            val_child_labels = all_val_labels[:, :19]
            val_parent_labels = all_val_labels[:, 19:23]
            val_child_predictions = all_val_predictions[:, :19]
            val_parent_predictions = all_val_predictions[:, 19:23]
            val_child_metrics = basic.calculate_metrics(val_child_labels, val_child_predictions)
            val_parent_metrics = basic.calculate_metrics(val_parent_labels, val_parent_predictions)
            val_metrics = basic.calculate_metrics(all_val_labels, all_val_predictions)

            val_loss /= len(val_loader)
            print(f"\n[Epoch {epoch + 1} Validation] Total Loss: {val_loss:.4f}")
            print(f"Child Micro - Precision: {val_child_metrics['micro']['precision']:.4f}, "
                  f"Recall: {val_child_metrics['micro']['recall']:.4f}, "
                  f"F1: {val_child_metrics['micro']['f1']:.4f}")
            print(f"Parent Micro - Precision: {val_parent_metrics['micro']['precision']:.4f}, "
                  f"Recall: {val_parent_metrics['micro']['recall']:.4f}, "
                  f"F1: {val_parent_metrics['micro']['f1']:.4f}")
            print(f"Overall Micro - Precision: {val_metrics['micro']['precision']:.4f}, "
                  f"Recall: {val_metrics['micro']['recall']:.4f}, "
                  f"F1: {val_metrics['micro']['f1']:.4f}")

            if save_base == 'best_loss':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved at epoch {epoch + 1} with Validation Loss: {val_loss}")
                else:
                    no_improvement_count += 1
            elif save_base == 'best_f1':
                if val_metrics['micro']['f1'] > best_f1:
                    best_f1 = val_metrics['micro']['f1']
                    no_improvement_count = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved at epoch {epoch + 1} with F1_micro: {val_metrics['micro']['f1']}")
                else:
                    no_improvement_count += 1
            else:
                raise Exception('Undefined save_base')

        print("Training Finished")

print("Training process save to: ", output_file)