import warnings
import torch
from transformers import BertTokenizer, BertModel
from contextlib import redirect_stdout
from model import AdvancedFeatureExtractionModel
import numpy as np
import basic

warnings.filterwarnings("ignore")

# 定义各种训练策略
model_origin_list = basic.model_origin_list
save_base_list = basic.save_base_list
loss_list = basic.loss_list
learning_rate_list = basic.learning_rate_list
logit_list = basic.logit_list

model_origin = basic.model_origin
save_base = basic.save_base
# loss_select = basic.loss_select
child_loss_select = basic.child_loss_select
parent_loss_select = basic.parent_loss_select

backbone_frozen = basic.backbone_frozen     # True
learning_rate = basic.learning_rate
logit = basic.logit

label_num = basic.label_num
num_epochs = basic.num_epochs  # 训练轮数
patience = basic.patience  # 早停策略的耐心值，即多少个 epoch 没有改进后停止
train_batch_size = basic.train_batch_size

weight_child = basic.weight_child
weight_parent = basic.weight_parent

# 使用 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')

# 初始化自定义模型
if model_origin == 'huggingface':
    base_model = BertModel.from_pretrained('Chinese-MentalBERT')
elif model_origin == 'pt':
    pt_model_path = 'pt_model.pt'
    base_model = BertModel.from_pretrained('Chinese-MentalBERT')
    # 不能用 因为我的微调模型是用BertForSequenceClassification训练的，比BertModel多一个分类器层，要先过滤。
    # base_model.load_state_dict(torch.load(pt_model_path))

    # state_dict = torch.load(pt_model_path, map_location=torch.device('cpu'))
    state_dict = torch.load(pt_model_path)
    # 过滤掉不需要的键
    def filter_keys(state_dictt, base_modell):
        model_keys = set(base_modell.state_dict().keys())
        # 只保留在base_model中出现的键
        filtered_state_dict = {k: v for k, v in state_dictt.items() if k in model_keys}
        return filtered_state_dict

    state_dict = filter_keys(state_dict, base_model)
    base_model.load_state_dict(state_dict, strict=False)

else:
    raise Exception('Undefined model_origin')

model = AdvancedFeatureExtractionModel(base_model, tokenizer, input_size=768, excel_file_path='decision_rules.csv')


# 读取测试集
test_texts, test_labels = basic.read_tsv('test_data.tsv', label_num)
# 创建测试集的数据集对象
test_dataset = basic.TextDataset(test_texts, test_labels, tokenizer)
test_loader = basic.DataLoader(test_dataset, batch_size=train_batch_size)


all_best_f1_micro = 0

# 将控制台输出内容保存为txt文件
best_model_path = f"{model_origin}_{save_base}_child={child_loss_select}_{weight_child}_parent={parent_loss_select}_{weight_parent}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
output_file = f"output_evaluate_{best_model_path}.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        #best_model_path = f"{save_base_list[0]}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
        # best_model_path = f"{save_base_list[0]}_logits={logit}_loss={loss_select}_lr={learning_rate}.pt"
        print("*************************************************************************")
        print(f"当前模型: {best_model_path}")
        print("*************************************************************************")
        model.load_state_dict(torch.load(best_model_path))
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        all_test_labels = []
        all_test_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                child_output = outputs[0]
                parent_output = outputs[1]

                # # 手动应用 sigmoid
                # child_output_sigmoid = torch.sigmoid(child_output)
                # parent_output_sigmoid = torch.sigmoid(parent_output)

                probabilities = torch.cat((child_output, parent_output), dim=1)
                binary_outputs = (probabilities > logit).float()

                # Collect labels and predictions for this batch
                all_test_labels.append(labels.cpu().numpy())
                all_test_predictions.append(binary_outputs.cpu().numpy())

        # Calculate metrics for the entire test set
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        all_test_predictions = np.concatenate(all_test_predictions, axis=0)
        test_child_labels = all_test_labels[:, :19]
        test_parent_labels = all_test_labels[:, 19:]
        test_child_predictions = all_test_predictions[:, :19]
        test_parent_predictions = all_test_predictions[:, 19:]

        test_child_metrics = basic.calculate_metrics(test_child_labels, test_child_predictions)
        test_parent_metrics = basic.calculate_metrics(test_parent_labels, test_parent_predictions)
        test_metrics = basic.calculate_metrics(all_test_labels, all_test_predictions)

        if test_metrics['micro']['f1'] > all_best_f1_micro:
            all_best_f1_micro = test_metrics['micro']['f1']
            # 保存预测的结果
            np.savetxt('gt_23.tsv', all_test_labels.astype(int), delimiter='\t', fmt='%d')
            np.savetxt('predict_23.tsv', all_test_predictions.astype(int), delimiter='\t', fmt='%d')
            print(f"{best_model_path} 模型的真实结果gt_23和预测结果predict_23已保存！")

        print("\n统计总体的指标:")
        print("Parent Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(test_parent_metrics['micro']['precision']*100,
                                                                                   test_parent_metrics['micro']['recall']*100,
                                                                                   test_parent_metrics['micro']['f1']*100))
        print("Child Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(test_child_metrics['micro']['precision']*100,
                                                                                  test_child_metrics['micro']['recall']*100,
                                                                                  test_child_metrics['micro']['f1']*100))
        print("Overall Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(test_metrics['micro']['precision']*100,
                                                                                    test_metrics['micro']['recall']*100,
                                                                                    test_metrics['micro']['f1']*100))

        parent_per_metrics = basic.calculate_evaluation_per_class(test_parent_labels, test_parent_predictions)
        print("\n分别统计父类的指标:")
        for key, value in parent_per_metrics.items():
             print(
                f"{key+1}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")


        child_per_metrics = basic.calculate_evaluation_per_class(test_child_labels, test_child_predictions)
        print("\n分别统计子类的指标:")
        for key, value in child_per_metrics.items():
             print(
                f"{key+1}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")


        print("\n###################按照原来的方法使用19子类标签计算指标:##################")
        # 统计4个大类的每一类的结果（不管子类预测的对不对）
        big_per_class_result = basic.metric_big_per_class(test_child_labels, test_child_predictions)
        print("\n分别统计4个父类的指标:（不管子类预测的对不对）")
        for key, value in big_per_class_result.items():
            print(
                f"{key}: Precision: {value['precision'] * 100:.2f}, Recall: {value['recall'] * 100:.2f}, F1: {value['f1'] * 100:.2f}")

        # 统计4个大类的总体结果：
        big_class_result = basic.metric_big_class(test_child_labels, test_child_predictions)
        print(f"\nParent nodes:")
        print(
            "Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(big_class_result['micro']['precision'] * 100,
                                                                          big_class_result['micro']['recall'] * 100,
                                                                          big_class_result['micro']['f1'] * 100))
        # print("Macro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(big_class_result['macro']['precision'] * 100, big_class_result['macro']['recall'] * 100, big_class_result['macro']['f1'] * 100))
        # print("Weighted: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(big_class_result['weighted']['precision'] * 100, big_class_result['weighted']['recall'] * 100, big_class_result['weighted']['f1'] * 100))

        # 统计23标签的总体结果：
        metrics_23 = basic.metric_23_class(test_child_labels, test_child_predictions)
        print(f"\nOverall nodes:")
        print("Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics_23['micro']['precision'] * 100,
                                                                            metrics_23['micro']['recall'] * 100,
                                                                            metrics_23['micro']['f1'] * 100))
        # print("Macro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics_23['macro']['precision'] * 100, metrics_23['macro']['recall'] * 100, metrics_23['macro']['f1'] * 100))
        # print("Weighted: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics_23['weighted']['precision'] * 100, metrics_23['weighted']['recall'] * 100, metrics_23['weighted']['f1'] * 100))

print("输出内容已保存为: ", output_file)


