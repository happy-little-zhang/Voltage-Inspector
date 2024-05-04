import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re
import time
import sys
import joblib
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import random
import math

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from my_model import *
from common import *


def read_vehicle_data(dataset_path, file_path, file_names, para_vehicle, count_limits = 0):
    count = 0
    # 用于存储样本特征
    features_dataset = []

    file_names = os.listdir(file_names)
    # print(file_names)

    # 遍历每个文件名
    for file_name in file_names:
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

        threshold = parameter_thresholds[para_vehicle]
        seg_length = parameter_length[para_vehicle]
        segment = segment_extract(data, threshold, seg_length)
        if len(segment) == 0:
            continue
        func = parameter_function[para_vehicle]
        if func == "gradient":
            segment = np.gradient(segment)

        feature.append(segment)

        features_dataset.append(feature)
        # print("feature", feature)

        count = count + 1
        print(count)
        if count_limits and count > count_limits:
            break

    return features_dataset


def read_vehicle_data_with_ecu_print_feature(dataset_path, file_path, file_names, count_limits = 0):
    count = 0

    error_count = 0
    # 用于存储样本特征
    features_dataset = []

    file_names = os.listdir(file_names)
    # print(file_names)

    # 遍历每个文件名
    for file_name in file_names:
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
            error_count += 1
            continue

        # ack位电压,去除,异常样本
        if max(differential) > 2.8:
            error_count += 1
            continue

        v_mean, v_max, t_bit, t_plat = ecu_print_feature_extract(differential, False)
        feature.append(v_mean)
        feature.append(v_max)
        feature.append(t_bit)
        feature.append(t_plat)

        features_dataset.append(feature)
        # print("feature", feature)

        count = count + 1
        print(f"count: {count}, error_count: {error_count}")
        if count_limits and count > count_limits:
            break

    return features_dataset


def ecu_identification():
    # "Ford Fiesta"
    # "Honda Civic"
    target_vehicle_names = ["Ford Fiesta", "Honda Civic"]

    for target_vehicle_name in target_vehicle_names:
        #if target_vehicle_name == "Honda Civic":
        #    print("target_vehicle_name:", target_vehicle_name)
        #else:
        #    continue

        dataset_path = "ECUPrint_dataset/" + target_vehicle_name + "/"
        file_paths = os.listdir(dataset_path)
        print(file_paths)

        if target_vehicle_name == "Honda Civic":
            file_paths = file_paths[1:]
            print(file_paths)

        train_data = []
        test_data = []

        # 遍历每个文件路径
        for file_num, file_path in enumerate(file_paths):

            # print("file_path:", file_path)

            # "Ford Fiesta" (environmental changes)
            # "1_0min"
            # "2_10min"
            # "ENVIRONMENTAL_1_30min"
            # "ENVIRONMENTAL_2_60min"

            # "Honda Civic"  (environmental changes)
            # 1_0min
            # ENVIRONMENTAL_1_15min_static
            # ENVIRONMENTAL_2_30min_static
            # ENVIRONMENTAL_3_60min_static
            # ENVIRONMENTAL_4_15min_dynamic
            # ENVIRONMENTAL_5_30min_dynamic
            # ENVIRONMENTAL_6_60min_dynamic

            # 获取文件路径下的所有文件名
            file_names = os.path.join(dataset_path, file_path)
            print("file_names", file_names)

            # 第一个文件用于模型训练
            if file_num == 0:
                print(f"file_num: {file_num}, file_path: {file_path}")
                # feature_dataset = read_vehicle_data(dataset_path, file_path, file_names, target_vehicle_name, 3000)
                feature_dataset = read_vehicle_data(dataset_path, file_path, file_names, target_vehicle_name, 0)
                for sample in feature_dataset:
                    train_data.append(sample)

            else:
                #feature_dataset = read_vehicle_data(dataset_path, file_path, file_names, target_vehicle_name, 1000)
                feature_dataset = read_vehicle_data(dataset_path, file_path, file_names, target_vehicle_name, 0)
                test_data.append(feature_dataset)

        # read the clustering results
        clustering_save_path = "experiment_results/clustering/" + target_vehicle_name + ".csv"
        ecu_mapping = []
        with open(clustering_save_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if re.match(r'^\d', row[0]):
                    ecu_mapping.append(row[1])
        ecu_mapping = [eval(item) for item in ecu_mapping]
        print("ecu_mapping:", ecu_mapping)

        # 创建对应的标签列表
        train_labels = [row[0] for row in train_data]
        train_labels_y_numerical = [find_ecu(ecu_mapping, row_label) for row_label in train_labels]
        # 将字典中的样本数据转换为列表
        train_samples = [row[1] for row in train_data]
        train_samples = np.array(train_samples)
        print("train_samples.shape: ", train_samples.shape)

        X_train = np.array(train_samples).astype(float)
        Y_train = np.array(train_labels_y_numerical).astype(float)


        print("X_train.shape:", X_train.shape)

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            #DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=10),              # 耗时过高
            #TimeSeriesForest(random_state=2),      # 耗时过高
            #KNeighborsClassifier(),                # 内存太高
            #LogisticRegression(max_iter=3000),
            #SVC(),
            #GaussianNB(),
        ]

        # 循环遍历每个模型
        for model in models:
            # 创建模型实例
            model_name = model.__class__.__name__

            # 模型训练
            model.fit(X_train, Y_train)

            # step = 1000  # 指定分割步长
            # num_parts = len(test_data) // step  # 计算可分为几份
            # data_parts = [test_data[i * step:(i + 1) * step] for i in range(num_parts)]  # 使用切片操作分割数据
            # print(data_parts)

            detection_accuracy = []
            execution_times = []
            for batch_num, test_batch in enumerate(test_data):
                # 创建对应的标签列表
                test_labels = [row[0] for row in test_batch]
                test_labels_y_numerical = [find_ecu(ecu_mapping, row_label) for row_label in test_labels]
                # 将字典中的样本数据转换为列表
                test_samples = [row[1] for row in test_batch]
                test_samples = np.array(test_samples)
                # print("test_samples.shape: ", test_samples.shape)

                X_test = np.array(test_samples).astype(float)
                Y_test = np.array(test_labels_y_numerical).astype(float)

                start_time = time.time()
                y_pred = model.predict(X_test)
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time / 1e-6 / len(X_test))

                # 计算检测精度
                accuracy = accuracy_score(Y_test, y_pred)
                detection_accuracy.append(accuracy*100)

            average_execution_time = np.mean(execution_times)
            print(f"model_names: {model_name}, average_execution_time: {average_execution_time} us, accuracy: {detection_accuracy}")
            detection_accuracy = np.array(detection_accuracy)
            plt.plot(detection_accuracy, label=model_name)


        #plt.legend()
        plt.xlabel("Time periods")
        plt.ylabel("Accuracy (%)")
        xindex = np.arange(len(file_paths)-1)
        plt.xticks(xindex)
        plt.grid()

        fig_save_flag = True
        if fig_save_flag:
            base_path = "experiment_results/environment"
            figure_name = target_vehicle_name + ".jpg"
            savefig_path = os.path.join(base_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()

def ecu_identification_ecu_print():
    # "Ford Fiesta"
    # "Honda Civic"
    target_vehicle_name = "Honda Civic"

    dataset_path = "ECUPrint_dataset/" + target_vehicle_name + "/"
    file_paths = os.listdir(dataset_path)
    print(file_paths)

    train_data = []
    test_data = []

    # 遍历每个文件路径
    for file_num, file_path in enumerate(file_paths):

        # print("file_path:", file_path)

        # "Ford Fiesta" (environmental changes)
        # "1_0min"
        # "2_10min"
        # "ENVIRONMENTAL_1_30min"
        # "ENVIRONMENTAL_2_60min"

        # "Honda Civic"  (environmental changes)
        # 1_0min
        # ENVIRONMENTAL_1_15min_static
        # ENVIRONMENTAL_2_30min_static
        # ENVIRONMENTAL_3_60min_static
        # ENVIRONMENTAL_4_15min_dynamic
        # ENVIRONMENTAL_5_30min_dynamic
        # ENVIRONMENTAL_6_60min_dynamic

        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        print("file_names", file_names)

        # 第一个文件用于模型训练
        if file_num == 0:
            print(f"file_num: {file_num}, file_path: {file_path}")
            feature_dataset = read_vehicle_data_with_ecu_print_feature(dataset_path, file_path, file_names, 0)
            for sample in feature_dataset:
                train_data.append(sample)
        else:
            feature_dataset = read_vehicle_data_with_ecu_print_feature(dataset_path, file_path, file_names, 0)
            test_data.append(feature_dataset)

    # read the clustering results
    clustering_save_path = "experiment_results/clustering/" + target_vehicle_name + ".csv"
    ecu_mapping = []
    with open(clustering_save_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if re.match(r'^\d', row[0]):
                ecu_mapping.append(row[1])
    ecu_mapping = [eval(item) for item in ecu_mapping]
    print("ecu_mapping:", ecu_mapping)

    # 创建对应的标签列表
    train_labels = [row[0] for row in train_data]
    train_labels_y_numerical = [find_ecu(ecu_mapping, row_label) for row_label in train_labels]
    # 将字典中的样本数据转换为列表
    train_samples = [row[1:] for row in train_data]
    train_samples = np.array(train_samples)
    print("train_samples.shape: ", train_samples.shape)

    X_train = np.array(train_samples).astype(float)
    Y_train = np.array(train_labels_y_numerical).astype(float)


    print("X_train.shape:", X_train.shape)
    feature_reduction = 0
    if feature_reduction:
        # 创建决策树分类器
        clf = DecisionTreeClassifier()
        # 拟合数据
        clf.fit(X_train, Y_train)
        # 获取特征重要性评估值
        importance = clf.feature_importances_
        main_pos = np.where(importance != 0)
        print("main_pos:", main_pos)

        selected_X_train = X_train[:, main_pos[0]]
        X_train = selected_X_train

        print("X_train.shape:", X_train.shape)
        # X_test = X_test[:, main_pos]

    '''
    df = pd.DataFrame(X_train)
    # calculate the correlations
    correlations = df.corr()
    # plot the heatmap
    sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)
    # plot the clustermap
    sns.clustermap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)
    plt.show()
    '''
    # python 特征选择
    # https://zhuanlan.zhihu.com/p/348201771

    # 创建模型并进行训练
    # 定义要尝试的模型列表
    models = [
        # DecisionTreeClassifier(criterion="entropy", splitter="best", ),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(max_iter=3000),
        SVC(),
        GaussianNB(),
    ]

    # 循环遍历每个模型
    for model in models:
        # 创建模型实例
        model_name = model.__class__.__name__

        # 模型训练
        model.fit(X_train, Y_train)

        # step = 1000  # 指定分割步长
        # num_parts = len(test_data) // step  # 计算可分为几份
        # data_parts = [test_data[i * step:(i + 1) * step] for i in range(num_parts)]  # 使用切片操作分割数据
        # print(data_parts)

        detection_accuracy = []
        execution_times = []
        for batch_num, test_batch in enumerate(test_data):
            # 创建对应的标签列表
            test_labels = [row[0] for row in test_batch]
            test_labels_y_numerical = [find_ecu(ecu_mapping, row_label) for row_label in test_labels]
            # 将字典中的样本数据转换为列表
            test_samples = [row[1:] for row in test_batch]
            test_samples = np.array(test_samples)
            # print("test_samples.shape: ", test_samples.shape)

            X_test = np.array(test_samples).astype(float)
            Y_test = np.array(test_labels_y_numerical).astype(float)

            if feature_reduction:
                selected_X_test = X_test[:, main_pos[0]]
                X_test = selected_X_test

            y_pred = []

            step = 5
            for i in range(len(X_test) // step):
                if i == (len(X_test) // step) - 1:
                    single_x = X_test[i * step:]
                    #single_y = Y_test[i * step:]
                else:
                    single_x = X_test[i * step:(i + 1) * step]
                    #single_y = Y_test[i * step:(i + 1) * step]

                start_time = time.time()
                r = model.predict(single_x)
                if model_name == "GaussianNB" or model_name == "MyNearestNeighbor2":
                    model.partial_fit(single_x, r)
                end_time = time.time()
                y_pred.append(r)

                execution_time = end_time - start_time
                execution_times.append(execution_time / 1e-6 / len(single_x))

            y_pred = [element for sublist in y_pred for element in sublist]
            y_pred = np.array(y_pred)

            # 计算检测精度
            accuracy = accuracy_score(Y_test, y_pred)
            detection_accuracy.append(accuracy)

        average_execution_time = np.mean(execution_times)
        print(f"model_names: {model_name}, average_execution_time: {average_execution_time} us, accuracy: {detection_accuracy}")
        detection_accuracy = np.array(detection_accuracy)
        plt.plot(detection_accuracy, label=model_name)
    plt.legend()
    plt.show()





def main():
    ecu_identification()
    #ecu_identification_with_feature()
    #ecu_identification_ecu_print()



if __name__ == '__main__':
    main()









