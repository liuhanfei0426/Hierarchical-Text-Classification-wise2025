import torch
import torch.nn as nn
import pandas as pd

class process_decison_node(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(process_decison_node, self).__init__()
        self.process = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
        )
        self.decision = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        process = self.process(x)
        decision = self.decision(process)
        return process, decision

class decison_node(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(decison_node, self).__init__()
        self.decision = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decision = self.decision(x)
        return decision

class AdvancedFeatureExtractionModel(nn.Module):
    def __init__(self, base_model, tokenizer, input_size, excel_file_path):
        super(AdvancedFeatureExtractionModel, self).__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        # 读取 Excel 文件
        df = pd.read_csv(excel_file_path)
        self.tree_dict = df.groupby('Parent')['Child'].apply(list).to_dict()
        # 创建区域和节点模块
        for region, nodes in self.tree_dict.items():
            # 为每个区域添加处理决策节点
            self.add_module(region, process_decison_node(input_size))

            # 为每个子节点添加决策节点
            for node in nodes:
                self.add_module(node, decison_node(input_size))

        # 添加特征增强和投影层（恢复层次化对比功能）
        self.feature_enhancer = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(input_size)
        )
        self.parent_proj = nn.Linear(input_size, input_size)

    def forward(self, input_ids, attention_mask):
        # 从 BERT 模型提取 [CLS] token 的隐藏状态
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        child_decisions = []
        parent_decisions = []
        parent_features = []

        # 处理每个区域及其节点
        for region, nodes in self.tree_dict.items():
            region_module = getattr(self, region)
            region_process, region_decision = region_module(x)
            parent_decisions.append(region_decision)  # (batch_size, 1)
            parent_features.append(region_process)  # (batch_size, hidden_size)
            # 为每个子节点处理
            for node in nodes:
                node_module = getattr(self, node)
                node_decision = node_module(region_process)  # (batch_size, 1)
                # 层次约束：子节点决策受父节点决策加权
                node_final_decision = node_decision * region_decision
                child_decisions.append(node_final_decision)

        # 连接决策结果
        child_decisions = torch.cat(child_decisions, dim=1)  # (batch_size, num_child)
        parent_decisions = torch.cat(parent_decisions, dim=1)  # (batch_size, num_parent)
        parent_features = torch.stack(parent_features, dim=1)  # (batch_size, num_parent, hidden_size)
        parent_features = parent_features.mean(dim=1)  # (batch_size, hidden_size)

        # 特征增强和投影（恢复 contrast_features 和 projected_parent）
        enhanced_features = self.feature_enhancer(x)  # (batch_size, hidden_size)
        contrast_features = torch.cat((enhanced_features, parent_features), dim=-1)  # (batch_size, 2 * hidden_size)
        projected_parent = self.parent_proj(parent_features)  # (batch_size, hidden_size)

        return child_decisions, parent_decisions, contrast_features, projected_parent

if __name__ == '__main__':
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('zwzzz/Chinese-MentalBERT')
    base_model = BertModel.from_pretrained('zwzzz/Chinese-MentalBERT')

    # 冻结基础模型参数
    for param in base_model.parameters():
        param.requires_grad = False
    # 初始化自定义模型
    model = AdvancedFeatureExtractionModel(base_model, tokenizer, input_size=768, excel_file_path='decision_rules.csv')
    # 标记化输入文本
    texts = ["example text 1", "example text 2"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 前向传播
    child_decisions, parent_decisions, contrast_features, projected_parent = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    print('Model child output:', child_decisions)
    print('Model parent output:', parent_decisions)
    print('Contrast features shape:', contrast_features.shape)
    print('Projected parent shape:', projected_parent.shape)