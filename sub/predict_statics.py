import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def calculate_metrics(true_labels, predicted_labels):
    # 计算 micro 平均的精确率、召回率和 F1 分数
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')

    # 计算 macro 平均的精确率、召回率和 F1 分数
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')

    # 计算 weighted 平均的精确率、召回率和 F1 分数
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

def calculate_evaluation_per_class(true_label, prediction):
    results = {}
    # 获取类别总数
    num_classes = true_label.shape[1] if len(true_label.shape) > 1 else max(true_label.max(), prediction.max()) + 1
    # 对每个类别进行循环，计算每个类别的指标
    for class_index in range(num_classes):
        # 提取当前类别的标签
        true_label_class = true_label[:, class_index]
        prediction_class = prediction[:, class_index]
        # 计算recall, precision, 和 f1-score
        recall = recall_score(true_label_class, prediction_class, zero_division=0)
        # recall = metrics.recall_score(true_label_class, prediction_class, zero_division=0)
        precision = precision_score(true_label_class, prediction_class, zero_division=0)
        f1 = f1_score(true_label_class, prediction_class, zero_division=0)
        # # 统计 true_label_class 中是否全部为 0
        # all_zeros = np.all(prediction_class == 0)
        # if all_zeros:
        #     print("prediction_class 中全部为 0")
        # else:
        #     print("prediction_class 中不全为 0")

        # 保存每个类别的结果
        results[class_index] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    return results

def metric_big_class(true_label, prediction):
    # 分别统计4个大类的指标
    # (sample, class_num)
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]

    total = 0

    # 遍历四个大类
    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []

        # 遍历样本
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            # 每一类的最终预测结果是所有子类结果的 按位与
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        # 更新total初始值，即更新开始的列
        total = total + split_list[big_clss_id]

        predict_pre_list = np.array(predict_pre_list, dtype=np.int32)
        label_pre_list = np.array(label_pre_list, dtype=np.int32)

        # 计算recall, precision, 和 f1-score
        recall = recall_score(label_pre_list, predict_pre_list, zero_division=0)
        precision = precision_score(label_pre_list, predict_pre_list, zero_division=0)
        f1 = f1_score(label_pre_list, predict_pre_list, zero_division=0)
        # # 统计 true_label_class 中是否全部为 0
        # all_zeros = np.all(predict_pre_list == 0)
        # if all_zeros:
        #     print("predict_pre_list 中全部为 0")
        # else:
        #     print("predict_pre_list 中不全为 0")

        # 保存每个类别的结果
        results[big_clss_id] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    return results


def metric_big_class_total(true_label, prediction):
    # 分别统计4个大类的指标
    # (sample, class_num)
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]

    total = 0
    all_predict_pre_list = []
    all_label_pre_list = []

    # 遍历四个大类
    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []

        # 遍历样本
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            # 每一类的最终预测结果是所有子类结果的 按位与
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        # 更新total初始值，即更新开始的列
        total = total + split_list[big_clss_id]

        # predict_pre_list = np.array(predict_pre_list, dtype=np.int32)
        # label_pre_list = np.array(label_pre_list, dtype=np.int32)

        # 将结果追加到总列表中
        all_predict_pre_list.append(predict_pre_list)
        all_label_pre_list.append(label_pre_list)

    # 将列表转换为 NumPy 数组并按列拼接
    all_predict_pre_array = np.column_stack(all_predict_pre_list)
    all_label_pre_array = np.column_stack(all_label_pre_list)

    return calculate_metrics(all_label_pre_array, all_predict_pre_array)




def metric_23_class_total(true_label, prediction):
    # 分别统计4个大类的指标
    # (sample, class_num)
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]

    total = 0
    all_predict_pre_list = []
    all_label_pre_list = []

    # 遍历四个大类
    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []

        # 遍历样本
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            # 每一类的最终预测结果是所有子类结果的 按位与
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        # 更新total初始值，即更新开始的列
        total = total + split_list[big_clss_id]

        # predict_pre_list = np.array(predict_pre_list, dtype=np.int32)
        # label_pre_list = np.array(label_pre_list, dtype=np.int32)

        # 将结果追加到总列表中
        all_predict_pre_list.append(predict_pre_list)
        all_label_pre_list.append(label_pre_list)

    # 将列表转换为 NumPy 数组并按列拼接
    all_predict_pre_array = np.column_stack(all_predict_pre_list)
    all_label_pre_array = np.column_stack(all_label_pre_list)

    # 将 true_label 和 all_label_pre_array 按列拼接
    final_label_array = np.column_stack((true_label, all_label_pre_array))
    # 将 prediction 和 all_predict_pre_array 按列拼接
    final_prediction_array = np.column_stack((prediction, all_predict_pre_array))

    return calculate_metrics(final_label_array, final_prediction_array)






# 读取真实标签文件并按列合并
gt_files = ['gt_sub1.tsv', 'gt_sub2.tsv', 'gt_sub3.tsv', 'gt_sub4.tsv']
gt_list = [pd.read_csv(file, sep='\t') for file in gt_files]
gt = pd.concat(gt_list, axis=1)

# 读取预测结果文件并按列合并
predict_files = ['predict_sub1.tsv', 'predict_sub2.tsv', 'predict_sub3.tsv', 'predict_sub4.tsv']
predict_list = [pd.read_csv(file, sep='\t') for file in predict_files]
predict = pd.concat(predict_list, axis=1)

# 转换为 NumPy 数组
gt = gt.to_numpy()
predict = predict.to_numpy()

# 保存 NumPy 数组
np.save('gt.npy', gt)
np.save('predict.npy', predict)

metrics = calculate_metrics(gt, predict)
print(f"标签19的总体测试结果: \n")
print("Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['micro']['precision'],
                                                                    metrics['micro']['recall'],
                                                                    metrics['micro']['f1']))
print("Macro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['macro']['precision'],
                                                                    metrics['macro']['recall'],
                                                                    metrics['macro']['f1']))
print("Weighted: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metrics['weighted']['precision'],
                                                                       metrics['weighted']['recall'],
                                                                       metrics['weighted']['f1']))

per_class_result = calculate_evaluation_per_class(gt, predict)
print("分别统计每一类的指标:")
for key, value in per_class_result.items():
    print(f"{key}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")

big_class_result = metric_big_class(gt, predict)
print("分别统计四个大类的指标:")
for key, value in big_class_result.items():
    # print(f"{key}: Precision: {value['precision']:.4f}, Recall: {value['recall']:.4f}, F1: {value['f1']:.4f}")
    print(f"{key}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")


metric_big_class_total_result = metric_big_class_total(gt, predict)
print(f"四个大类的总体测试结果: \n")
print("Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metric_big_class_total_result['micro']['precision'],
                                                                    metric_big_class_total_result['micro']['recall'],
                                                                    metric_big_class_total_result['micro']['f1']))
print("Macro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metric_big_class_total_result['macro']['precision'],
                                                                    metric_big_class_total_result['macro']['recall'],
                                                                    metric_big_class_total_result['macro']['f1']))
print("Weighted: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metric_big_class_total_result['weighted']['precision'],
                                                                       metric_big_class_total_result['weighted']['recall'],
                                                                       metric_big_class_total_result['weighted']['f1']))


metric_23_class_total_result = metric_23_class_total(gt, predict)
print(f"23个类的总体测试结果: \n")
print("Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metric_23_class_total_result['micro']['precision'],
                                                                    metric_23_class_total_result['micro']['recall'],
                                                                    metric_23_class_total_result['micro']['f1']))
print("Macro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metric_23_class_total_result['macro']['precision'],
                                                                    metric_23_class_total_result['macro']['recall'],
                                                                    metric_23_class_total_result['macro']['f1']))
print("Weighted: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(metric_23_class_total_result['weighted']['precision'],
                                                                       metric_23_class_total_result['weighted']['recall'],
                                                                       metric_23_class_total_result['weighted']['f1']))
