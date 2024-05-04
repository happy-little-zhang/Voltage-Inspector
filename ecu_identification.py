import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re
import time
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pandas
import seaborn as sns
import random
import math
from pympler import asizeof

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from skrules import SkopeRules
from sklearn.linear_model import Perceptron
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
import pydotplus
import pygraphviz as pgv
from mlrl.boosting import Boomer


from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from my_model import *
from common import *

def ecu_identification_whole_vehicle():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    split_ratio = 0.5      # ratio of train dataset and test dataset

    current_ecu_num = 0

    whole_X_train = []
    whole_Y_train = []
    whole_X_test = []
    whole_Y_test = []

    # 遍历每个文件路径
    for file_path in file_paths:
        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        # 用于存储样本特征
        features_dataset = []

        vehicle_name = file_path
        print(vehicle_name)
        count = 0
        #if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        #if file_path == "Dacia Duster":
        #    print("vehicle:", file_path)
        #else:
        #    continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"


        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        file_names = os.listdir(file_names)
        #print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            #print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                header = next(reader)
                data = list(reader)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]

            differential = np.array(differential)

            index = np.arange(len(differential))
            #print(len(index))

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            if len(negative_indices) == 0:
                print("anomaly voltage: ", file)
                #continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                print("anomaly voltage: ", file)
                continue

            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            # Example usage
            data = differential
            #window_size = 20
            #iterations = 3
            #filtered_data = kz_filter(data, window_size, iterations)
            filtered_data = differential

            # 计算一阶导数
            first_derivative = np.gradient(filtered_data)

            threshold = 1.7
            segment_index = np.where((filtered_data > threshold) & (first_derivative >= 0))[0]

            seg_length = 200
            idx00 = segment_index[0]
            idx01 = idx00 + seg_length
            #idx02 = idx01 - 1
            segment = filtered_data[idx00:idx01]
            #segment = np.gradient(segment)
            feature.append(segment)
            #feature.append(filtered_data[idx02:idx01])

            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 1000:
            #    break

        # 输出结果
        # print(vehicle_name + " shape_features_dataset:")
        # print(features_dataset)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        #print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        # print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        # print(group_dataset)
        # for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset) * split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset) * split_ratio):])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        # print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        # print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        # merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        # print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)

        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        # merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #print("before current merged_train_dataset_y_numerical: ", merged_train_dataset_y_numerical)
        #print("before current Y_train: ", Y_train)
        #print("current_ecu_num: ", current_ecu_num)
        Y_train = Y_train + current_ecu_num
        Y_test = Y_test + current_ecu_num
        current_ecu_num = current_ecu_num + len(ecu_mapping)
        #print("current Y_train: ", Y_train)

        if len(whole_X_train) == 0:
            whole_X_train = X_train.copy()
            whole_Y_train = Y_train.copy()
            whole_X_test = X_test.copy()
            whole_Y_test = Y_test.copy()
        else:
            whole_X_train = np.concatenate([whole_X_train, X_train])
            whole_Y_train = np.concatenate([whole_Y_train, Y_train])
            whole_X_test = np.concatenate([whole_X_test, X_test])
            whole_Y_test = np.concatenate([whole_Y_test, Y_test])

    print("whole_X_train shape: ", whole_X_train.shape)
    print("whole_Y_train shape: ", whole_Y_train.shape)
    print("whole_Y_train : ", whole_Y_train)
    print("current_ecu_num: ", current_ecu_num)
    print("np.unique(whole_Y_train): ", np.unique(whole_Y_train))
    print("np.unique(whole_Y_test): ", np.unique(whole_Y_test))

    # mpn: models_performance_names
    models_performance_names = ["----------model----------", "train_num", "test_num", "train_time(s)", "test_time(s)",
                                "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
    # story the model performance
    models_performance_results = []

    # 创建模型并进行训练
    # 定义要尝试的模型列表
    models = [
        #DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10),
        #KNeighborsClassifier(),
        #LogisticRegression(max_iter=5000),
        #SVC(),
        #GaussianNB(),
        #MyNearestNeighbor(),
        #MyNearestNeighbor2(),
    ]

    # 循环遍历每个模型
    for model in models:
        # 创建模型实例
        model_name = model.__class__.__name__

        # 指定路径
        # base_path = "shape_results/identification/"
        # 子文件夹名称
        # folder_name = model_name
        # 拼接路径
        # folder_path = os.path.join(base_path, folder_name)
        # if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        # else:
        # print(f"Folder '{folder_path}' already exists. Skipping...")

        # 模型训练
        start_time = time.time()
        model.fit(whole_X_train, whole_Y_train)
        end_time = time.time()
        train_elapsed_time = end_time - start_time

        # 将模型保存到磁盘
        model_file_path = "model.pkl"
        joblib.dump(model, model_file_path)
        # 获取模型文件的大小
        model_file_size = os.path.getsize(model_file_path)

        # 加载模型
        model = joblib.load(model_file_path)
        # model_size = asizeof.asizeof(model)
        # 在测试集上进行预测
        # start_time
        start_time = time.time()

        y_pred = model.predict(whole_X_test)

        # end_time
        end_time = time.time()
        test_elapsed_time = end_time - start_time

        # 计算分类准确度
        # print("type(Y_test): ", type(Y_test))
        # print("type(y_pred): ", type(y_pred))
        # print("Y_test", Y_test)
        # print("y_pred", y_pred)
        accuracy = accuracy_score(whole_Y_test, y_pred)

        current_result = []
        current_result.append(model_name)
        current_result.append(len(whole_X_train))
        current_result.append(len(whole_X_test))
        current_result.append(round(train_elapsed_time, 6))
        current_result.append(round(test_elapsed_time, 6))
        current_result.append(round(train_elapsed_time / len(whole_X_train) / 1e-6, 6))
        current_result.append(round(test_elapsed_time / len(whole_X_test) / 1e-6, 6))
        current_result.append(model_file_size)
        # current_result.append(model_size)
        current_result.append(round(accuracy, 6))
        models_performance_results.append(current_result)

        cm_plt_flag = 0
        if cm_plt_flag == 1:
            # 计算混淆矩阵
            cm = confusion_matrix(whole_Y_test, y_pred)

            # 将混淆矩阵的每一行除以该行的总和，进行正则化
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print(cm_normalized)

            target_index = np.arange(current_ecu_num)
            target_names = target_index

            # colormap: 'viridis' 'coolwarm' 'RdYlBu' 'Greens' 'Blues' 'Oranges' 'Reds' 'YlOrBr'
            # 可视化混淆矩阵
            plt.figure(figsize=(16, 12))
            #sns.heatmap(cm, annot=True, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
            #sns.heatmap(cm_normalized, annot=False, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
            sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', annot_kws={"size": 5},
                        linewidths=.01, xticklabels=target_names, yticklabels=target_names, cbar=False)
            #plt.title("Whole vehicles")
            plt.xlabel('Predicted label')
            plt.ylabel('Actual label')
            # plt.grid(True, linestyle='--', linewidth=0.5)
            #plt.show()
            # figure_name = vehicle_name + ".jpg"
            savefig_path = "experiment_results/identification/overall.jpg"
            plt.savefig(savefig_path, dpi=300)
            plt.close()

    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)
    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)

def ecu_identification_whole_vehicle_ecuprint():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    split_ratio = 0.5      # ratio of train dataset and test dataset

    current_ecu_num = 0

    whole_X_train = []
    whole_Y_train = []
    whole_X_test = []
    whole_Y_test = []

    # 遍历每个文件路径
    for file_path in file_paths:
        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        # 用于存储样本特征
        features_dataset = []

        vehicle_name = file_path
        print(vehicle_name)
        count = 0
        #if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        #if file_path == "Dacia Duster":
        #    print("vehicle:", file_path)
        #else:
        #    continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"


        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        file_names = os.listdir(file_names)
        #print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            #print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                header = next(reader)
                data = list(reader)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]

            differential = np.array(differential)

            index = np.arange(len(differential))
            #print(len(index))

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            if len(negative_indices) == 0:
                print("anomaly voltage: ", file)
                continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                print("anomaly voltage: ", file)
                continue

            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            v_mean, v_max, t_bit, t_plat = ecu_print_feature_extract(differential, False)
            feature.append(v_mean)
            feature.append(v_max)
            feature.append(t_bit)
            feature.append(t_plat)

            features_dataset.append(feature)

            count = count + 1
            print(count)
            #if count > 3000:
            #    break

        # 输出结果
        # print(vehicle_name + " shape_features_dataset:")
        # print(features_dataset)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        #print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        # print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        # print(group_dataset)
        # for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset) * split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset) * split_ratio):])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        # print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        # print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1:] for row in merged_train_dataset_list]
        # merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        # print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)

        merged_test_dataset_x = [row[1:] for row in merged_test_dataset_list]
        # merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #print("before current merged_train_dataset_y_numerical: ", merged_train_dataset_y_numerical)
        #print("before current Y_train: ", Y_train)
        #print("current_ecu_num: ", current_ecu_num)
        Y_train = Y_train + current_ecu_num
        Y_test = Y_test + current_ecu_num
        current_ecu_num = current_ecu_num + len(ecu_mapping)
        #print("current Y_train: ", Y_train)

        if len(whole_X_train) == 0:
            whole_X_train = X_train.copy()
            whole_Y_train = Y_train.copy()
            whole_X_test = X_test.copy()
            whole_Y_test = Y_test.copy()
        else:
            whole_X_train = np.concatenate([whole_X_train, X_train])
            whole_Y_train = np.concatenate([whole_Y_train, Y_train])
            whole_X_test = np.concatenate([whole_X_test, X_test])
            whole_Y_test = np.concatenate([whole_Y_test, Y_test])

    print("whole_X_train shape: ", whole_X_train.shape)
    print("whole_Y_train shape: ", whole_Y_train.shape)
    print("whole_Y_train : ", whole_Y_train)
    print("current_ecu_num: ", current_ecu_num)
    print("np.unique(whole_Y_train): ", np.unique(whole_Y_train))
    print("np.unique(whole_Y_test): ", np.unique(whole_Y_test))

    # mpn: models_performance_names
    models_performance_names = ["----------model----------", "train_num", "test_num", "train_time(s)", "test_time(s)",
                                "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
    # story the model performance
    models_performance_results = []

    # 创建模型并进行训练
    # 定义要尝试的模型列表
    models = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(max_iter=5000),
        SVC(),
        GaussianNB(),
        #MyNearestNeighbor(),
        #MyNearestNeighbor2(),
    ]

    # 循环遍历每个模型
    for model in models:
        # 创建模型实例
        model_name = model.__class__.__name__

        # 指定路径
        # base_path = "shape_results/identification/"
        # 子文件夹名称
        # folder_name = model_name
        # 拼接路径
        # folder_path = os.path.join(base_path, folder_name)
        # if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        # else:
        # print(f"Folder '{folder_path}' already exists. Skipping...")

        # 模型训练
        start_time = time.time()
        model.fit(whole_X_train, whole_Y_train)
        end_time = time.time()
        train_elapsed_time = end_time - start_time

        # 将模型保存到磁盘
        model_file_path = "model.pkl"
        joblib.dump(model, model_file_path)
        # 获取模型文件的大小
        model_file_size = os.path.getsize(model_file_path)

        # 加载模型
        model = joblib.load(model_file_path)
        # model_size = asizeof.asizeof(model)
        # 在测试集上进行预测
        # start_time
        start_time = time.time()

        y_pred = model.predict(whole_X_test)

        # end_time
        end_time = time.time()
        test_elapsed_time = end_time - start_time

        # 计算分类准确度
        # print("type(Y_test): ", type(Y_test))
        # print("type(y_pred): ", type(y_pred))
        # print("Y_test", Y_test)
        # print("y_pred", y_pred)
        accuracy = accuracy_score(whole_Y_test, y_pred)

        current_result = []
        current_result.append(model_name)
        current_result.append(len(whole_X_train))
        current_result.append(len(whole_X_test))
        current_result.append(round(train_elapsed_time, 6))
        current_result.append(round(test_elapsed_time, 6))
        current_result.append(round(train_elapsed_time / len(whole_X_train) / 1e-6, 6))
        current_result.append(round(test_elapsed_time / len(whole_X_test) / 1e-6, 6))
        current_result.append(model_file_size)
        # current_result.append(model_size)
        current_result.append(round(accuracy, 6))
        models_performance_results.append(current_result)

        cm_plt_flag = 1
        if cm_plt_flag == 1:
            # 计算混淆矩阵
            cm = confusion_matrix(whole_Y_test, y_pred)

            # 将混淆矩阵的每一行除以该行的总和，进行正则化
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print(cm_normalized)

            target_index = np.arange(current_ecu_num)
            target_names = target_index

            # colormap: 'viridis' 'coolwarm' 'RdYlBu' 'Greens' 'Blues' 'Oranges' 'Reds' 'YlOrBr'
            # 可视化混淆矩阵
            plt.figure(figsize=(16, 9))
            #sns.heatmap(cm, annot=True, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
            sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                        linewidths=.01, xticklabels=target_names, yticklabels=target_names)
            plt.title("whole vehicles")
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            # plt.grid(True, linestyle='--', linewidth=0.5)
            plt.show()
            # figure_name = vehicle_name + ".jpg"
            # savefig_path = os.path.join(folder_path, figure_name)
            # plt.savefig(savefig_path, dpi=300)
            # plt.close()

    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)
    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)

def ecu_identification():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)

        if file_path == "Hyundai i20":
            print("vehicle:", file_path)
        else:
            continue

        # special folder formats, handled separately
        if file_path == "Ford Fiesta":
            #file_path = file_path + "/1_0min/"     #(识别精度不佳)
            file_path = file_path + "/2_10min/"

        # special folder formats, handled separately
        if file_path == "Honda Civic":
            #file_path = file_path + "/1_0min/"     #(识别精度不佳)
            file_path = file_path + "/ENVIRONMENTAL_5_30min_dynamic/"

        # 用于存储样本特征
        features_dataset = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                header = next(reader)
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            threshold = parameter_thresholds[vehicle_name]
            seg_length = parameter_length[vehicle_name]
            segment = segment_extract(data, threshold, seg_length)
            if len(segment) == 0:
                continue
            func = parameter_function[vehicle_name]
            if func == "gradient":
                segment = np.gradient(segment)
            feature.append(segment)
            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 6000:
            #    jump_flag = 1
            #    break

        # 输出结果
        # print(vehicle_name + "features_dataset:")
        # print(features_dataset)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)

        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        print("X_train.shape:", X_train.shape)

        # mpn: models_performance_names
        models_performance_names = ["----------model----------", "train_num", "test_num", "train_time(s)", "test_time(s)",
                                    "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
        # story the model performance
        models_performance_results = []


        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=10),
            #KNeighborsClassifier(),
            #LogisticRegression(max_iter=3000),
            #SVC(probability=True),
            #GaussianNB(),
        ]

        # 循环遍历每个模型
        for model in models:
            # 创建模型实例
            model_name = model.__class__.__name__

            # 指定路径
            base_path = "experiment_results/identification/"
            # 子文件夹名称
            folder_name = model_name
            # 拼接路径
            folder_path = os.path.join(base_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            else:
                print(f"Folder '{folder_path}' already exists. Skipping...")

            # 模型训练
            start_time = time.time()
            model.fit(X_train, Y_train)
            end_time = time.time()
            train_elapsed_time = end_time - start_time

            # 将模型保存到磁盘
            model_file_path = "model.pkl"
            joblib.dump(model, model_file_path)
            # 获取模型文件的大小
            model_file_size = os.path.getsize(model_file_path)

            # 加载模型
            model = joblib.load(model_file_path)
            # 在测试集上进行预测
            # start_time
            start_time = time.time()

            y_pred = model.predict(X_test)

            # end_time
            end_time = time.time()
            test_elapsed_time = end_time - start_time

            # 计算分类准确度
            #print("type(Y_test): ", type(Y_test))
            #print("type(y_pred): ", type(y_pred))
            #print("Y_test", Y_test)
            #print("y_pred", y_pred)
            accuracy = accuracy_score(Y_test, y_pred)
            current_result = []
            current_result.append(model_name)
            current_result.append(len(X_train))
            current_result.append(len(X_test))
            current_result.append(round(train_elapsed_time, 6))
            current_result.append(round(test_elapsed_time, 6))
            current_result.append(round(train_elapsed_time/len(X_train)/1e-6, 6))
            current_result.append(round(test_elapsed_time/len(X_test)/1e-6, 6))
            current_result.append(model_file_size)
            current_result.append(round(accuracy, 6))
            models_performance_results.append(current_result)

            cm_plt_flag = 1
            if cm_plt_flag == 1:
                # 计算混淆矩阵
                cm = confusion_matrix(Y_test, y_pred)

                # 将混淆矩阵的每一行除以该行的总和，进行正则化
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                # print(cm_normalized)

                target_index = np.arange(len(ecu_mapping))
                target_names = target_index

                # colormap: 'viridis' 'coolwarm' 'RdYlBu' 'Greens' 'Blues' 'Oranges' 'Reds' 'YlOrBr'
                # 可视化混淆矩阵
                plt.figure(figsize=(8, 6))
                #sns.heatmap(cm, annot=True, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                            linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                #plt_name = vehicle_name + " (" + model_name + ")"
                #plt.title(plt_name)
                plt.title(vehicle_name)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                #plt.grid(True, linestyle='--', linewidth=0.5)
                #plt.show()
                figure_name = vehicle_name + ".jpg"
                savefig_path = os.path.join(folder_path, figure_name)
                plt.savefig(savefig_path, dpi=300)
                plt.close()

        # print the comparison results
        str_size = []
        for i in range(len(models_performance_names)):
            str_size.append(len(models_performance_names[i]) + 5)
        #print("str_size", str_size)
        output_head = ""
        for i, mpn in enumerate(models_performance_names):
            #print(f"{mpn:>{str_size[i]}}")
            output_head += f"{mpn:<{str_size[i]}}"
        print(output_head)
        for model_result in models_performance_results:
            cc_str = ""
            for i, v in enumerate(model_result):
                cc_str += f"{str(v):<{str_size[i]}}"
            print(cc_str)


def ecu_print_feature_test():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)
        # if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        if file_path == "Dacia Duster":
            print("vehicle:", file_path)
        else:
            continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

        # 用于存储样本特征
        features_dataset = []

        # 存储遍历过的ID
        id_list = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]
            differential = np.array(differential)

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            if len(negative_indices) == 0:
                continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                continue

            v_mean, v_max, t_bit, t_plat = ecu_print_feature_extract(differential, False)

            feature.append(v_mean)
            feature.append(v_max)
            feature.append(t_bit)
            feature.append(t_plat)

            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 1000:
            #    jump_flag = 1
            #    break

            # 如果参数为空，跳过
            if not features_dataset:
                continue

        data = features_dataset
        data = data_reorganization(data)
        for row in data:
            print(row)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1:] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)


        merged_test_dataset_x = [row[1:] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #X_train = merged_train_dataset_x
        #Y_train = merged_train_dataset_y_numerical
        #X_test = merged_test_dataset_x
        #Y_test = merged_test_dataset_y_numerical

        # mpn: models_performance_names
        models_performance_names = ["----------model----------", "train_num", "test_num", "train_time(s)", "test_time(s)",
                                    "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
        # story the model performance
        models_performance_results = []


        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            #DecisionTreeClassifier(criterion="entropy", splitter="best", ),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=10),
            KNeighborsClassifier(),
            LogisticRegression(max_iter=3000),
            SVC(),
            GaussianNB(),
        ]

        # 循环遍历每个模型
        for model in models:
            # 创建模型实例
            model_name = model.__class__.__name__

            # 指定路径
            # base_path = "shape_results/identification/"
            # 子文件夹名称
            # folder_name = model_name
            # 拼接路径
            # folder_path = os.path.join(base_path, folder_name)
            #if not os.path.exists(folder_path):
            #    os.makedirs(folder_path)
            #else:
                #print(f"Folder '{folder_path}' already exists. Skipping...")

            # 模型训练
            start_time = time.time()
            model.fit(X_train, Y_train)
            end_time = time.time()
            train_elapsed_time = end_time - start_time

            # 将模型保存到磁盘
            model_file_path = "model.pkl"
            joblib.dump(model, model_file_path)

            # 获取模型文件的大小
            model_file_size = os.path.getsize(model_file_path)

            # 在测试集上进行预测
            # start_time
            start_time = time.time()

            y_pred = model.predict(X_test)

            # end_time
            end_time = time.time()
            test_elapsed_time = end_time - start_time


            # 计算分类准确度
            #print("type(Y_test): ", type(Y_test))
            #print("type(y_pred): ", type(y_pred))
            #print("Y_test", Y_test)
            #print("y_pred", y_pred)
            accuracy = accuracy_score(Y_test, y_pred)

            current_result = []
            current_result.append(model_name)
            current_result.append(len(X_train))
            current_result.append(len(X_test))
            current_result.append(round(train_elapsed_time, 6))
            current_result.append(round(test_elapsed_time, 6))
            current_result.append(round(train_elapsed_time/len(X_train)/1e-6, 6))
            current_result.append(round(test_elapsed_time/len(X_test)/1e-6, 6))
            current_result.append(model_file_size)
            current_result.append(round(accuracy, 6))
            models_performance_results.append(current_result)

            cm_plt_flag = 0
            if cm_plt_flag == 1:
                # 计算混淆矩阵
                cm = confusion_matrix(Y_test, y_pred)

                # 将混淆矩阵的每一行除以该行的总和，进行正则化
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                # print(cm_normalized)

                target_index = np.arange(len(ecu_mapping))
                target_names = target_index

                # colormap: 'viridis' 'coolwarm' 'RdYlBu' 'Greens' 'Blues' 'Oranges' 'Reds' 'YlOrBr'
                # 可视化混淆矩阵
                plt.figure(figsize=(8, 6))
                #sns.heatmap(cm, annot=True, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt= '.2f',
                            linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                plt.title(vehicle_name)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                #plt.grid(True, linestyle='--', linewidth=0.5)
                plt.show()
                #figure_name = vehicle_name + ".jpg"
                #savefig_path = os.path.join(folder_path, figure_name)
                #plt.savefig(savefig_path, dpi=300)
                #plt.close()

        # print the comparison results
        str_size = []
        for i in range(len(models_performance_names)):
            str_size.append(len(models_performance_names[i]) + 5)
        #print("str_size", str_size)
        output_head = ""
        for i, mpn in enumerate(models_performance_names):
            #print(f"{mpn:>{str_size[i]}}")
            output_head += f"{mpn:<{str_size[i]}}"
        print(output_head)
        for model_result in models_performance_results:
            cc_str = ""
            for i, v in enumerate(model_result):
                cc_str += f"{str(v):<{str_size[i]}}"
            print(cc_str)


def method_model_selection():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    # mpn: models_performance_names
    models_performance_names = ["-------vehicle-------", "--------model--------", "train_num", "test_num",
                                "train_time(s)", "test_time(s)",
                                "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
    # story the model performance
    models_performance_results = []

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)
        #if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        #if file_path == "Honda Civic":
        #    print("vehicle:", file_path)
        #else:
        #    continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

        # 用于存储样本特征
        features_dataset = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                header = next(reader)
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            #threshold = parameter_thresholds[vehicle_name]
            #seg_length = parameter_length[vehicle_name]
            threshold = 1.7
            seg_length = 200
            segment = segment_extract(data, threshold, seg_length)
            if len(segment) == 0:
                continue
            #func = parameter_function[vehicle_name]
            #if func == "gradient":
            #    segment = np.gradient(segment)
            #segment = calculate_curvature(segment)
            feature.append(segment)


            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 6000:
            #    jump_flag = 1
            #    break

        # 输出结果
        # print(vehicle_name + " shape_features_dataset:")
        # print(features_dataset)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)

        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        print("X_train.shape:", X_train.shape)

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=10),
            KNeighborsClassifier(),
            LogisticRegression(max_iter=3000),
            SVC(),
            GaussianNB(),
        ]

        # 循环遍历每个模型
        for model in models:
            # 创建模型实例
            model_name = model.__class__.__name__

            # 指定路径
            # base_path = "experiment_results/identification/"
            # 子文件夹名称
            # folder_name = model_name
            # 拼接路径
            # folder_path = os.path.join(base_path, folder_name)
            #if not os.path.exists(folder_path):
            #    os.makedirs(folder_path)
            #else:
                #print(f"Folder '{folder_path}' already exists. Skipping...")

            # 模型训练
            start_time = time.time()
            model.fit(X_train, Y_train)
            end_time = time.time()
            train_elapsed_time = end_time - start_time


            # 将模型保存到磁盘
            model_file_path = "model.pkl"
            joblib.dump(model, model_file_path)
            # 获取模型文件的大小
            model_file_size = os.path.getsize(model_file_path)

            # 加载模型
            model = joblib.load(model_file_path)
            #model_size = asizeof.asizeof(model)
            # 在测试集上进行预测
            # start_time
            start_time = time.time()

            y_pred = model.predict(X_test)

            # end_time
            end_time = time.time()
            test_elapsed_time = end_time - start_time


            # 计算分类准确度
            #print("type(Y_test): ", type(Y_test))
            #print("type(y_pred): ", type(y_pred))
            #print("Y_test", Y_test)
            #print("y_pred", y_pred)
            accuracy = accuracy_score(Y_test, y_pred)

            current_result = []
            current_result.append(vehicle_name)
            current_result.append(model_name)
            current_result.append(len(X_train))
            current_result.append(len(X_test))
            current_result.append(round(train_elapsed_time, 6))
            current_result.append(round(test_elapsed_time, 6))
            current_result.append(round(train_elapsed_time / len(X_train) / 1e-6, 6))
            current_result.append(round(test_elapsed_time / len(X_test) / 1e-6, 6))
            current_result.append(model_file_size)
            current_result.append(round(accuracy, 6) * 100)
            models_performance_results.append(current_result)

            cm_plt_flag = 0
            if cm_plt_flag == 1:
                # 计算混淆矩阵
                cm = confusion_matrix(Y_test, y_pred)

                # 将混淆矩阵的每一行除以该行的总和，进行正则化
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                # print(cm_normalized)

                target_index = np.arange(len(ecu_mapping))
                target_names = target_index

                # colormap: 'viridis' 'coolwarm' 'RdYlBu' 'Greens' 'Blues' 'Oranges' 'Reds' 'YlOrBr'
                # 可视化混淆矩阵
                plt.figure(figsize=(8, 6))
                #sns.heatmap(cm, annot=True, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.4f',
                            linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                plt_name = vehicle_name + " (" + model_name + ")"
                plt.title(plt_name)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                #plt.grid(True, linestyle='--', linewidth=0.5)
                plt.show()
                #figure_name = vehicle_name + ".jpg"
                #savefig_path = os.path.join(folder_path, figure_name)
                #plt.savefig(savefig_path, dpi=300)
                #plt.close()

    save_res_flag = True
    save_res_path = "experiment_results/identification/detect_res_model_selection.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


def method_parameter_selection():

    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    # mpn: models_performance_names
    models_performance_names = ["-------vehicle-------", "--------model--------", "train_num", "test_num",
                                "train_time(s)", "test_time(s)",
                                "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
    # story the model performance
    models_performance_results = []

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)
        # if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        if file_path == "Hyundai i20":
            print("vehicle:", file_path)
        else:
            continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

        # 用于存储样本特征
        features_dataset = []

        # 存储遍历过的ID
        id_list = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]
            differential = np.array(differential)

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            #if len(negative_indices) == 0:
            #    continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                continue

            #differential = sliding_average(differential, 10)
            feature.append(differential)

            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 1000:
            #    jump_flag = 1
            #    break

            # 如果参数为空，跳过
            # if not features_dataset:
            #     continue

        #data = features_dataset
        #for row in data:
        #    print(row)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)


        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #X_train = merged_train_dataset_x
        #Y_train = merged_train_dataset_y_numerical
        #X_test = merged_test_dataset_x
        #Y_test = merged_test_dataset_y_numerical

        # 创建模型并进行训练
        # 定义模型参数搜索空间
        #initial_threshold = [1.6, 1.7, 1.8, 1.9, 2.0]
        #segment_length = [100, 200, 300, 400, 500]
        space_segment_threshold = np.arange(1, 19, 1)/10       # Honda Civic 车辆不能超过1.85
        space_segment_length = np.arange(1, 7, 1)*50

        print("space_segment_threshold: ", space_segment_threshold)
        print("space_segment_length: ", space_segment_length)

        total_accuracy = []
        #for segment_threshold in space_segment_threshold:
        for segment_length in space_segment_length:
            sub_total_accuracy = []
            #for segment_length in space_segment_length:
            for segment_threshold in space_segment_threshold:
                #print("segment_threshold: ", segment_threshold)
                #print("segment_length: ", segment_length)

                model = VInspector(segment_threshold, segment_length)
                model_name = f"VI[{segment_threshold}, {segment_length}]"

                # 模型训练
                start_time = time.time()
                model.fit(X_train, Y_train)
                end_time = time.time()
                train_elapsed_time = end_time - start_time

                # 将模型保存到磁盘
                model_file_path = "model.pkl"
                joblib.dump(model, model_file_path)

                # 获取模型文件的大小
                model_file_size = os.path.getsize(model_file_path)

                # 在测试集上进行预测
                # start_time
                start_time = time.time()

                y_pred = model.predict(X_test)
                #y_pred = model.predict_joint_log_proba(X_test)
                #y_pred = model.predict_proba(X_test)

                # end_time
                end_time = time.time()
                test_elapsed_time = end_time - start_time


                # 计算分类准确度
                #print("type(Y_test): ", type(Y_test))
                #print("type(y_pred): ", type(y_pred))
                #print("Y_test", Y_test)
                #print("y_pred", y_pred)
                #print("y_pred")
                #for row in y_pred:
                #    print(row)

                accuracy = accuracy_score(Y_test, y_pred)

                current_result = []
                current_result.append(vehicle_name)
                current_result.append(model_name)
                current_result.append(len(X_train))
                current_result.append(len(X_test))
                current_result.append(round(train_elapsed_time, 6))
                current_result.append(round(test_elapsed_time, 6))
                current_result.append(round(train_elapsed_time/len(X_train)/1e-6, 6))
                current_result.append(round(test_elapsed_time/len(X_test)/1e-6, 6))
                current_result.append(model_file_size)
                current_result.append(round(accuracy, 6) * 100)
                models_performance_results.append(current_result)

                sub_total_accuracy.append(accuracy * 100)

            total_accuracy.append(sub_total_accuracy)

        fig_store_flag = False
        y = total_accuracy
        x = space_segment_threshold
        #for i, threshold in enumerate(space_segment_threshold):
        for i, segment_length in enumerate(space_segment_length):
            plt.plot(x, y[i], label=str(segment_length))
        plt.grid()
        plt.xticks(x)
        plt.xlabel("Threshold (V)")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        if fig_store_flag:
            figure_name = vehicle_name + ".jpg"
            savefig_path = "experiment_results/identification/parameter_selection/" + figure_name
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()

    save_res_flag = False
    save_res_path = "experiment_results/identification/optimization/detect_res_parameter_selection.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


def method_parameter_selection_gradient():

    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    # mpn: models_performance_names
    models_performance_names = ["-------vehicle-------", "--------model--------", "train_num", "test_num",
                                "train_time(s)", "test_time(s)",
                                "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
    # story the model performance
    models_performance_results = []

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)
        # if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        #if file_path == "Dacia Duster":
        #    print("vehicle:", file_path)
        #else:
        #    continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

        # 用于存储样本特征
        features_dataset = []

        # 存储遍历过的ID
        id_list = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]
            differential = np.array(differential)

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            #if len(negative_indices) == 0:
            #    continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                continue

            #differential = sliding_average(differential, 10)
            feature.append(differential)

            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 1000:
            #    jump_flag = 1
            #    break

            # 如果参数为空，跳过
            # if not features_dataset:
            #     continue

        #data = features_dataset
        #for row in data:
        #    print(row)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)


        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #X_train = merged_train_dataset_x
        #Y_train = merged_train_dataset_y_numerical
        #X_test = merged_test_dataset_x
        #Y_test = merged_test_dataset_y_numerical

        # 创建模型并进行训练
        # 定义模型参数搜索空间
        #initial_threshold = [1.6, 1.7, 1.8, 1.9, 2.0]
        #segment_length = [100, 200, 300, 400, 500]
        space_segment_threshold = np.arange(1, 19, 1)/10       # Honda Civic 车辆不能超过1.85
        space_segment_length = np.arange(1, 7, 1)*50

        print("space_segment_threshold: ", space_segment_threshold)
        print("space_segment_length: ", space_segment_length)

        total_accuracy = []
        #for segment_threshold in space_segment_threshold:
        for segment_length in space_segment_length:
            sub_total_accuracy = []
            #for segment_length in space_segment_length:
            for segment_threshold in space_segment_threshold:
                #print("segment_threshold: ", segment_threshold)
                #print("segment_length: ", segment_length)

                model = VInspectorGradient(segment_threshold, segment_length)
                model_name = f"VIG[{segment_threshold}, {segment_length}]"

                # 模型训练
                start_time = time.time()
                model.fit(X_train, Y_train)
                end_time = time.time()
                train_elapsed_time = end_time - start_time

                # 将模型保存到磁盘
                model_file_path = "model.pkl"
                joblib.dump(model, model_file_path)

                # 获取模型文件的大小
                model_file_size = os.path.getsize(model_file_path)

                # 在测试集上进行预测
                # start_time
                start_time = time.time()

                y_pred = model.predict(X_test)
                #y_pred = model.predict_joint_log_proba(X_test)
                #y_pred = model.predict_proba(X_test)

                # end_time
                end_time = time.time()
                test_elapsed_time = end_time - start_time


                # 计算分类准确度
                #print("type(Y_test): ", type(Y_test))
                #print("type(y_pred): ", type(y_pred))
                #print("Y_test", Y_test)
                #print("y_pred", y_pred)
                #print("y_pred")
                #for row in y_pred:
                #    print(row)

                accuracy = accuracy_score(Y_test, y_pred)

                current_result = []
                current_result.append(vehicle_name)
                current_result.append(model_name)
                current_result.append(len(X_train))
                current_result.append(len(X_test))
                current_result.append(round(train_elapsed_time, 6))
                current_result.append(round(test_elapsed_time, 6))
                current_result.append(round(train_elapsed_time/len(X_train)/1e-6, 6))
                current_result.append(round(test_elapsed_time/len(X_test)/1e-6, 6))
                current_result.append(model_file_size)
                current_result.append(round(accuracy, 6) * 100)
                models_performance_results.append(current_result)

                sub_total_accuracy.append(accuracy * 100)

            total_accuracy.append(sub_total_accuracy)

        fig_store_flag = True
        y = total_accuracy
        x = space_segment_threshold
        #for i, threshold in enumerate(space_segment_threshold):
        for i, segment_length in enumerate(space_segment_length):
            plt.plot(x, y[i], label=str(segment_length))
        plt.grid()
        plt.xticks(x)
        plt.xlabel("Threshold (V)")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        if fig_store_flag:
            figure_name = vehicle_name + "_gradient.jpg"
            savefig_path = "experiment_results/identification/parameter_selection/" + figure_name
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()

    save_res_flag = False
    save_res_path = "experiment_results/identification/optimization/detect_res_parameter_selection.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()

def method_comparison():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    # mpn: models_performance_names
    models_performance_names = ["-------vehicle-------", "--------model--------", "train_num", "test_num",
                                "train_time(s)", "test_time(s)", "per_train_time(us)", "per_test_time(us)", "model_size(byte)",
                                "accuracy", "precision", "recall", "f1_score"]
    # story the model performance
    models_performance_results = []

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)
        # if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        #if file_path == "Dacia Duster":
        #    print("vehicle:", file_path)
        #else:
        #    continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

        # 用于存储样本特征
        features_dataset = []

        # 存储遍历过的ID
        id_list = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]
            differential = np.array(differential)

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            #if len(negative_indices) == 0:
            #    continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                continue

            #differential = sliding_average(differential, 10)
            feature.append(differential)

            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 500:
            #    jump_flag = 1
            #    break

            # 如果参数为空，跳过
            # if not features_dataset:
            #     continue

        #data = features_dataset
        #for row in data:
        #    print(row)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)


        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #X_train = merged_train_dataset_x
        #Y_train = merged_train_dataset_y_numerical
        #X_test = merged_test_dataset_x
        #Y_test = merged_test_dataset_y_numerical

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            Baseline(),
            ECUPrint(),
            VInspectorLR(),
            BaselineOptimization(),
            ECUPrintOptimization(),
            VInspector(),
            VInspectorGradient(),
        ]

        # 循环遍历每个模型
        for model in models:
            # 创建模型实例
            model_name = model.__class__.__name__

            # 指定路径
            # base_path = "shape_results/identification/"
            # 子文件夹名称
            # folder_name = model_name
            # 拼接路径
            # folder_path = os.path.join(base_path, folder_name)
            #if not os.path.exists(folder_path):
            #    os.makedirs(folder_path)
            #else:
                #print(f"Folder '{folder_path}' already exists. Skipping...")

            # 模型训练
            start_time = time.time()
            model.fit(X_train, Y_train)
            end_time = time.time()
            train_elapsed_time = end_time - start_time

            # 将模型保存到磁盘
            model_file_path = "model.pkl"
            joblib.dump(model, model_file_path)

            # 获取模型文件的大小
            model_file_size = os.path.getsize(model_file_path)

            # 在测试集上进行预测
            # start_time
            start_time = time.time()

            y_pred = model.predict(X_test)
            #y_pred = model.predict_joint_log_proba(X_test)
            #y_pred = model.predict_proba(X_test)

            # end_time
            end_time = time.time()
            test_elapsed_time = end_time - start_time


            # 计算分类准确度
            #print("type(Y_test): ", type(Y_test))
            #print("type(y_pred): ", type(y_pred))
            #print("Y_test", Y_test)
            #print("y_pred", y_pred)
            #print("y_pred")
            #for row in y_pred:
            #    print(row)

            # "accuracy", "precision", "recall", "f1_score"
            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred, average="micro")
            recall = recall_score(Y_test, y_pred, average="micro")
            f1 = f1_score(Y_test, y_pred, average="micro")

            current_result = []
            current_result.append(vehicle_name)
            current_result.append(model_name)
            current_result.append(len(X_train))
            current_result.append(len(X_test))
            current_result.append(round(train_elapsed_time, 6))
            current_result.append(round(test_elapsed_time, 6))
            current_result.append(round(train_elapsed_time/len(X_train)/1e-6, 6))
            current_result.append(round(test_elapsed_time/len(X_test)/1e-6, 6))
            current_result.append(model_file_size)
            current_result.append(round(accuracy, 6))
            current_result.append(round(precision, 6))
            current_result.append(round(recall, 6))
            current_result.append(round(f1, 6))
            models_performance_results.append(current_result)

            cm_plt_flag = 0
            if cm_plt_flag == 1:
                # 计算混淆矩阵
                cm = confusion_matrix(Y_test, y_pred)

                # 将混淆矩阵的每一行除以该行的总和，进行正则化
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                # print(cm_normalized)

                target_index = np.arange(len(ecu_mapping))
                target_names = target_index

                # colormap: 'viridis' 'coolwarm' 'RdYlBu' 'Greens' 'Blues' 'Oranges' 'Reds' 'YlOrBr'
                # 可视化混淆矩阵
                plt.figure(figsize=(8, 6))
                #sns.heatmap(cm, annot=True, cmap='Blues', linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                           linewidths=.01, xticklabels=target_names, yticklabels=target_names)
                plt.title(vehicle_name)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                #plt.grid(True, linestyle='--', linewidth=0.5)
                plt.show()
                #figure_name = vehicle_name + ".jpg"
                #savefig_path = os.path.join(folder_path, figure_name)
                #plt.savefig(savefig_path, dpi=300)
                #plt.close()

    save_res_flag = True
    save_res_path = "experiment_results/identification/detect_res_comparison.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


def method_test_single():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    # mpn: models_performance_names
    models_performance_names = ["-------vehicle-------", "--------model--------", "train_num", "test_num",
                                "train_time(s)", "test_time(s)",
                                "per_train_time(us)", "per_test_time(us)", "model_size(byte)", "accuracy"]
    # story the model performance
    models_performance_results = []

    count = 0
    jump_flag = 0

    # 遍历每个文件路径
    for file_path in file_paths:

        if jump_flag == 1:
            break

        # print("vehicle:", file_path)
        # "Dacia Duster"
        # "Dacia Logan"
        # "Ford Ecosport"
        # "Ford Fiesta"  (environmental changes)
        # "Ford Kuga"
        # "Honda Civic"  (environmental changes)
        # "Hyundai i20"
        # "Hyundai ix35"
        # "John Deere Tractor"
        # "Opel Corsa"

        vehicle_name = file_path
        print(vehicle_name)
        # if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        if file_path == "Dacia Duster":
            print("vehicle:", file_path)
        else:
            continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

        # 用于存储样本特征
        features_dataset = []

        # 存储遍历过的ID
        id_list = []

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:

            if jump_flag == 1:
                break

            # print("file_name:", file_name)
            # 正则表达式匹配文件名
            # 统一的文件名格式,使用“|”将多个规则连到一起
            pattern = r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\].csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_\w+\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_Logan_\d+\.csv|' \
                      r'^\[\d+\]_\w+\_\d+\_DATA\d+\_extracted_ZERO_\[\d+\]001_002_003\.csv|' \
                      r'^\[\d+\]_\w+\_extracted_extracted_ZERO_\[\d+\]_cold_\d+\.csv'

            file_re_flag = 0
            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                file_re_flag = 1
            else:
                # 文件名格式不匹配，忽略该文件
                # print(f'Ignoring file: {file_name}')
                continue

            # 构建完整的文件路径
            # file = os.path.join(file_path, file_name)
            file = os.path.join(dataset_path, file_path, file_name)

            print(file)
            with open(file, 'r') as csv_file:
                lines = csv_file.readlines()

                # 读取特殊格式数据
                special_data = [line.strip() for line in lines[0:3]]
                # print(special_data)

                # if special_data[0] == '511':
                #    continue

                # 读取标准的CSV数据
                reader = csv.reader(lines[5:])
                data = list(reader)

            # 提取数据
            feature = []
            current_id = special_data[0]
            feature.append(current_id)

            # 提取数据列
            timestamp = [float(row[0]) for row in data]
            # index = np.arange(len(time))
            # channel_a = [float(row[1]) for row in data]
            # channel_b = [float(row[2]) for row in data]
            differential = [float(row[1]) - float(row[2]) for row in data]
            differential = np.array(differential)

            # 判断是否为完整的 one bit 数据，完整的one bit 逻辑曲线中具有正负偏导数
            # dominant,  differential = 0   V < 0.9
            # recessive, differential = 2
            # logical value
            logical_array = np.where(differential < 0.9, 0, 1)
            # one difference
            # diff = np.diff(logical_array)
            derivative = np.gradient(logical_array)
            positive_indices = np.where(derivative > 0)[0]
            negative_indices = np.where(derivative < 0)[0]
            # print(positive_indices, negative_indices)
            # 电压非凸形视为异常点，不做处理
            if len(negative_indices) == 0:
                continue

            # ack位电压,去除,异常样本
            if max(differential) > 2.8:
                continue

            #differential = sliding_average(differential, 10)
            feature.append(differential)

            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 1000:
            #    jump_flag = 1
            #    break

            # 如果参数为空，跳过
            # if not features_dataset:
            #     continue

        #data = features_dataset
        #for row in data:
        #    print(row)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # group the dataset by id
        id_list = [row[0] for row in features_dataset]
        unique_id_list = [x for i, x in enumerate(id_list) if x not in id_list[:i]]
        #print("unique_id_list:", unique_id_list)
        group_dataset = []
        for i in range(len(unique_id_list)):
            group_dataset.append([])
        for row in features_dataset:
            pos = unique_id_list.index(row[0])
            group_dataset[pos].append(row)
        #print(group_dataset)
        #for i in range(len(unique_id_list)):
        #   print(unique_id_list[i] + " data len:", len(group_dataset[i]))

        split_ratio = 0.5

        train_dataset = []
        test_dataset = []
        for i in range(len(unique_id_list)):
            sub_dataset = group_dataset[i]
            train_dataset.append(sub_dataset[:int(len(sub_dataset)*split_ratio)])
            test_dataset.append(sub_dataset[int(len(sub_dataset)*split_ratio):])

        #for i in range(len(unique_id_list)):
        #    print(unique_id_list[i] + " train_dataset len:", len(train_dataset[i]))
        #    print(unique_id_list[i] + " test_dataset len:", len(test_dataset[i]))

        #print("group_dataset[-1]: ", group_dataset[-1])
        #print("train_dataset[-1]: ", train_dataset[-1])
        #print("test_dataset[-1]: ", test_dataset[-1])

        merged_train_dataset_list = []
        for sublist in train_dataset:
            merged_train_dataset_list += sublist

        merged_test_dataset_list = []
        for sublist in test_dataset:
            merged_test_dataset_list += sublist
        #print("merged_train_dataset_list len: ", len(merged_train_dataset_list))
        #print("merged_test_dataset_list len: ", len(merged_test_dataset_list))

        random.shuffle(merged_train_dataset_list)
        random.shuffle(merged_test_dataset_list)

        # select the features
        merged_train_dataset_x = [row[1] for row in merged_train_dataset_list]
        #merged_train_dataset_y = [row[0] for row in merged_train_dataset_list]
        merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
        #print("merged_train_dataset_y_numerical", merged_train_dataset_y_numerical)


        merged_test_dataset_x = [row[1] for row in merged_test_dataset_list]
        #merged_test_dataset_y = [row[0] for row in merged_test_dataset_list]
        merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

        X_train = np.array(merged_train_dataset_x).astype(float)
        Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
        X_test = np.array(merged_test_dataset_x).astype(float)
        Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

        #X_train = merged_train_dataset_x
        #Y_train = merged_train_dataset_y_numerical
        #X_test = merged_test_dataset_x
        #Y_test = merged_test_dataset_y_numerical

        # 创建模型并进行训练
        # 定义模型参数搜索空间
        model = VInspector()
        model_name = f"VInspector"

        # 模型训练
        start_time = time.time()
        model.fit(X_train, Y_train)
        end_time = time.time()
        train_elapsed_time = end_time - start_time

        # 将模型保存到磁盘
        model_file_path = "model.pkl"
        joblib.dump(model, model_file_path)

        # 获取模型文件的大小
        model_file_size = os.path.getsize(model_file_path)

        # 在测试集上进行预测
        # start_time
        start_time = time.time()

        y_pred = model.predict(X_test)

        # end_time
        end_time = time.time()
        test_elapsed_time = end_time - start_time

        # 计算分类准确度
        accuracy = accuracy_score(Y_test, y_pred)

        current_result = []
        current_result.append(vehicle_name)
        current_result.append(model_name)
        current_result.append(len(X_train))
        current_result.append(len(X_test))
        current_result.append(round(train_elapsed_time, 6))
        current_result.append(round(test_elapsed_time, 6))
        current_result.append(round(train_elapsed_time / len(X_train) / 1e-6, 6))
        current_result.append(round(test_elapsed_time / len(X_test) / 1e-6, 6))
        current_result.append(model_file_size)
        current_result.append(round(accuracy, 6))
        models_performance_results.append(current_result)



    # print the comparison results
    str_size = []
    for i in range(len(models_performance_names)):
        str_size.append(len(models_performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(models_performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)
    for model_result in models_performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)



def main():

    #ecu_print_feature_test()               # reproduction of the ECUprint

    #ecu_identification_whole_vehicle()           # 10辆车一起的分类  固定参数 1.7 200 different
    #ecu_identification_whole_vehicle_ecuprint()  # 10辆车一起的分类  固定参数 1.7 200 different

    ecu_identification()                    # 精心挑选的参数，混淆矩阵存储
    #ecu_identification_with_feature()

    #method_model_selection()                # 固定阈值1.7和长度200，不同分类模型的比较
    #method_parameter_selection()           # VInspector模型固定，different不同阈值和长度下的检测比较
    #method_parameter_selection_gradient()  # VInspector模型固定，gradient不同阈值和长度下的检测比较
    #method_comparison()                    # VInspector(固定阈值和长度,固定模型)与baseline、ECUPrint方法比较
    #method_test_single()                   # VInspector 单独测试

if __name__ == '__main__':
    main()









