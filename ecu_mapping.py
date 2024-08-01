import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

import seaborn as sns
import plotly.graph_objects as go
from common import *


def ecu_clustering():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    count = 0
    jump_flag = 0

    # Traverse each file path
    for file_path in file_paths:

        if jump_flag == 1:
            break

        print("vehicle:", file_path)
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
        if file_path == "Honda Civic":
            print("vehicle:", file_path)
        else:
            continue

        # special folder formats, handled separately
        if file_path == "Ford Fiesta":
            file_path = file_path + "/1_0min/"
            #file_path = file_path + "/2_10min/"

        # special folder formats, handled separately
        if file_path == "Honda Civic":
            #file_path = file_path + "/1_0min/"
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
            #if count > 3000:
            #    jump_flag = 1
            #    break

        # print(vehicle_name + " features_dataset:")
        # print(features_dataset)

        plt_sto_flag = 0
        if plt_sto_flag == 1:

            plt.clf()
            for current_id, segment in features_dataset:
                plt.plot(np.arange(len(segment)), segment)
            plt.show()
            # 指定路径
            #base_path = "experiment_results/segment_plt"
            #figure_name = vehicle_name + "_differential.jpg"
            #savefig_path = os.path.join(base_path, figure_name)
            #plt.savefig(savefig_path, dpi=300)
            #plt.close()

            plt.clf()
            for current_id, segment in features_dataset:
                plt.plot(np.arange(len(np.gradient(segment))), np.gradient(segment))
            plt.show()
            # 指定路径
            #base_path = "experiment_results/segment_plt"
            #figure_name = vehicle_name + "_gradient.jpg"
            #savefig_path = os.path.join(base_path, figure_name)
            #plt.savefig(savefig_path, dpi=300)
            #plt.close()

        # clustering
        train_data = features_dataset
        train_data = data_reorganization(train_data)    #average representation of samples

        #print(train_data)
        #print("len(data)", len(train_data))
        # 创建对应的标签列表
        labels = [row[0] for row in train_data]
        # 将字典中的样本数据转换为列表
        samples = [row[1] for row in train_data]
        samples = np.array(samples)
        print(samples.shape)
        # print("labels", labels)
        # print("samples", samples)

        # 定义参数范围(网格搜索获取最佳聚类结果)
        param_grid = {
            'eps': [i / 100.0 for i in range(1, 201, 1)],      # 数值100 ，一阶导数 [i / 100.0 for i in range(1, 201, 1)]
            #'eps': [1.64],  # 数值100 ，一阶导数 [i / 100.0 for i in range(1, 201, 1)]
            'min_samples': [1],
            'metric': [time_series_distance_metrics],
        }

        # 定义评估指标
        best_score = -1
        best_params = {}
        best_labels = []

        # 网格搜索
        for params in ParameterGrid(param_grid):
            dbscan = DBSCAN(**params)
            dbscan.fit(samples)

            # 排除没有聚类的情况,以及超出轮廓函数参数范围
            if len(set(dbscan.labels_)) > 1 and len(set(dbscan.labels_)) < len(labels) - 1:
                score = silhouette_score(samples, dbscan.labels_)

                # 更新最佳参数和聚类结果
                if score > best_score:
                    best_score = score
                    best_params = params

        # 使用最佳参数重新进行聚类
        best_dbscan = DBSCAN(**best_params)
        best_dbscan.fit(samples)
        best_labels = best_dbscan.labels_

        # 输出最佳参数、评估指标和聚类结果
        print("Best Parameters: ", best_params)
        print("Best Silhouette Score: ", best_score)
        print("Best Clustering Labels: ", best_labels)

        # 获取聚类结果
        cluster_labels = best_labels

        # 创建字典来存储每个簇的字符
        clusters = {}
        for current_id, cluster_label in zip(labels, cluster_labels):
            if cluster_label not in clusters:
                clusters[cluster_label] = [current_id]
            else:
                clusters[cluster_label].append(current_id)

        # 输出每个簇包含的字符
        for cluster_label, elements in clusters.items():
            if cluster_label == -1:
                print(f"Noise: Samples {elements}")
            else:
                print(f"Cluster {cluster_label}: Samples {elements}")

        # 将聚类结果和ID保存为CSV文件
        clustering_sto_flag = 0
        if clustering_sto_flag == 1:
            # 指定路径
            base_path = "experiment_results/clustering"
            file_name = vehicle_name + ".csv"
            savefile_path = os.path.join(base_path, file_name)
            with open(savefile_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['best_params', best_params])
                writer.writerow(['best_score', best_score])
                writer.writerow(['best_labels', best_labels])

                writer.writerow(['Cluster Label', 'IDs'])
                for cluster_label, cluster_chars in clusters.items():
                    writer.writerow([cluster_label, cluster_chars])

        # 常见的20个颜色列表
        common_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'lime', 'cyan',
                         'gold', 'darkblue', 'darkorange', 'darkgreen', 'darkred', 'darkviolet', 'sienna', 'magenta',
                         'slategray', 'olive']

        added_labels = set()
        sum_value = [np.sum(sample) for sample in samples]
        min_index = np.argmin(sum_value)
        refer = train_data[min_index]               # 获取处于最下方的参考曲线
        for i, row in enumerate(train_data):
            cc_id = row[0]
            recessive_data = row[1]

            diff = time_series_distance_metrics(row[1], refer[1])
            cluster_num = -1
            for cluster_label, elements in clusters.items():
                if cc_id in elements:
                    cluster_num = cluster_label
            label_name = "ECU " + str(cluster_num)
            pos_x = diff
            pos_y = i
            if label_name not in added_labels:
                plt.scatter(pos_x, pos_y, color=common_colors[cluster_num], label=label_name)
                added_labels.add(label_name)
            else:
                plt.scatter(pos_x, pos_y, color=common_colors[cluster_num])
            plt.annotate(cc_id, (pos_x, pos_y), textcoords="offset points", xytext=(10, -5), ha='center')
            #plt.annotate(cc_id, (pos_x, pos_y), textcoords="offset points", xytext=(20, -5), ha='center')
            #plt.plot([pos_x, pos_x + 1], [pos_y, pos_y], color=common_colors[cluster_num], linestyle='--')
        plt.xlabel("Related distance")
        plt.ylabel("ID")
        plt.legend()
        #plt.grid()
        #plt.show()
        plt.close()

        plt_save_flag = 0
        if plt_save_flag == 1:
            # 指定路径
            base_path = "experiment_results/clustering"
            figure_name = vehicle_name + ".jpg"
            savefig_path = os.path.join(base_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()

        dynamic_plot_flag = 0
        if dynamic_plot_flag == 1:
            id_list = []
            # 创建图表数据
            plt_data = []
            sub_data_range = []

            # plot 多图
            for row in train_data:
                # print(row)
                # 使用eval函数解析数据
                # 解析数据
                cc_id = row[0]
                voltage_data = row[1]
                # print(voltage_data)

                # 将浮点数转换为NumPy数组
                data_array = np.array(voltage_data, dtype=float)

                # print(data_array)
                # print(data_array.shape)
                index = np.arange(data_array.shape[0])
                # plt.plot(index, recessive_data, color='b', label='recessive_data')
                #plt.plot(index, data_array)

                # 相同ID画一次图就可以了
                if cc_id not in id_list:
                    id_list.append(cc_id)
                    plt_data.append(go.Scatter(x=index, y=data_array, name=cc_id))
                    # plt_data.append(go.Scatter(x=index, y=data_array))
                    sub_data_range.append(data_array.shape[0])

            # 交互动态界面
            # 创建图表数据
            # 创建布局
            layout = go.Layout(
                showlegend=True,
                updatemenus=[
                    {
                        'buttons': [
                            {
                                'method': 'update',
                                'label': label,
                                'args': [{'visible': [index == i for i in range(len(sub_data_range))]}]
                            } for index, label in enumerate([f'Curve {i + 1}' for i in range(len(sub_data_range))])
                        ],
                        'direction': 'down',
                        'showactive': True,
                    }
                ]
            )
            # 创建图表对象
            fig = go.Figure(data=plt_data, layout=layout)
            # 显示图表
            fig.show()


def ecu_clustering_validation():
    """
    plot the evidence for the Ford Kuga vehicle
    :return:
    """
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    count = 0
    jump_flag = 0

    # Traverse each file path
    for file_path in file_paths:

        if jump_flag == 1:
            break

        print("vehicle:", file_path)
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
        if file_path == "Ford Kuga":
            print("vehicle:", file_path)
        else:
            continue

        # special folder formats, handled separately
        if file_path == "Ford Fiesta":
            file_path = file_path + "/1_0min/"
            #file_path = file_path + "/2_10min/"

        # special folder formats, handled separately
        if file_path == "Honda Civic":
            #file_path = file_path + "/1_0min/"
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
            #if count > 3000:
            #    jump_flag = 1
            #    break

        # print(vehicle_name + " features_dataset:")
        # print(features_dataset)

        # clustering
        train_data = features_dataset
        train_data = data_reorganization(train_data)    #average representation of samples

        #print(train_data)
        #print("len(data)", len(train_data))
        # 创建对应的标签列表
        labels = [row[0] for row in train_data]
        # 将字典中的样本数据转换为列表
        samples = [row[1] for row in train_data]
        samples = np.array(samples)
        print(samples.shape)
        # print("labels", labels)
        # print("samples", samples)

        if vehicle_name == "Ford Kuga":
            plt.clf()
            fig1_ids = ['04A', '04B', '208', '388']
            for i, current_id in enumerate(labels):
                if current_id in fig1_ids:
                    current_y = samples[i]
                    plt.plot(np.arange(len(current_y)), current_y, label=current_id)
            #plt.show()
            #plt.grid()
            plt.legend()
            savefig_path = "experiment_results/clustering/validation/Ford Kuga_fig1.jpg"
            plt.savefig(savefig_path, dpi=300)
            plt.close()

            plt.clf()
            fig2_ids = ['04A', '208', '160', '170', '2E0']
            for i, current_id in enumerate(labels):
                if current_id in fig2_ids:
                    current_y = samples[i]
                    plt.plot(np.arange(len(current_y)), current_y, label=current_id)
            #plt.show()
            #plt.grid()
            plt.legend()
            savefig_path = "experiment_results/clustering/validation/Ford Kuga_fig2.jpg"
            plt.savefig(savefig_path, dpi=300)
            plt.close()


def ecu_clustering_search_optimization():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

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


        # 获取文件路径下的所有文件名
        file_names = os.path.join(dataset_path, file_path)
        # print(file_names)

        file_names = os.listdir(file_names)
        #print(file_names)

        # 用于存储样本特征
        features_dataset = []
        count = 0
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
                header = next(reader)
                data = list(reader)

            # 提取数据列
            feature = []
            current_id = special_data[0]
            feature.append(current_id)
            differential = [float(row[1]) - float(row[2]) for row in data]
            differential = np.array(differential)
            feature.append(differential)
            features_dataset.append(feature)
            # print("feature", feature)

            count = count + 1
            print(count)
            #if count > 1000:
            #    jump_flag = 1
            #    break

        # 输出结果
        # print(vehicle_name + " features_dataset:")
        # print(features_dataset)


        # 定义参数范围(网格搜索获取最佳聚类结果)
        global_param_grid = {
            'threshold': [i / 100.0 for i in range(170, 191, 5)],
            'segment_length': [i * 100.0 for i in range(1, 5, 1)],
        }

        # 定义评估指标
        global_best_score = -1
        global_best_params = {}
        global_best_labels = []
        final_eps = -1

        cluster_num_result = []

        # 网格搜索
        for global_params in ParameterGrid(global_param_grid):
            #print(global_params)
            #print(global_params['threshold'], global_params['segment_length'])
            temp_data = []

            threshold = global_params["threshold"]
            seg_length = int(global_params["segment_length"])

            for sample in features_dataset:
                sample_id = sample[0]
                sample_data = slice_extract(sample[1], threshold, seg_length)
                if len(sample_data) == 0:
                    continue
                temp_data.append([sample_id, sample_data])

            temp_data = data_reorganization(temp_data)

            # 创建对应的标签列表
            labels = [row[0] for row in temp_data]
            # 将字典中的样本数据转换为列表
            samples = [row[1] for row in temp_data]
            #samples = [row[1:] for row in train_data]
            samples = np.array(samples)
            # print("labels", labels)
            # print("samples", samples)

            # 定义参数范围(网格搜索获取最佳聚类结果)
            param_grid = {
                'eps': [i / 100.0 for i in range(1, 201, 1)],  # 数值100 ，一阶导数 [i / 100.0 for i in range(1, 201, 1)]
                'min_samples': [1],
                'metric': [time_series_distance_metrics],
            }

            # 定义评估指标
            best_score = -1
            best_params = {}
            best_labels = []

            # 网格搜索
            for params in ParameterGrid(param_grid):
                dbscan = DBSCAN(**params)
                # dbscan = DensityClustering(**params)
                dbscan.fit(samples)

                # 排除没有聚类的情况,以及超出轮廓函数参数范围
                if len(set(dbscan.labels_)) > 1 and len(set(dbscan.labels_)) < len(labels) - 1:
                    # score = silhouette_score(reduced_features, dbscan.labels_)
                    score = silhouette_score(samples, dbscan.labels_)
                    # score = davies_bouldin_score(scaled_data, dbscan.labels_)
                    # score = calinski_harabasz_score(scaled_data, dbscan.labels_)      # results is very terrible

                    # 更新最佳参数和聚类结果
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_labels = dbscan.labels_

            if best_score > global_best_score:
                global_best_score = best_score
                global_best_params = global_params
                global_best_labels = best_labels
                final_eps = best_params['eps']

            print(f"global_params: {global_params}, 'eps': {best_params['eps']}, "
                  f"best_score: {best_score}, best_clustering_num: {len(set(best_labels))}")

            cluster_num_result.append([global_params["segment_length"], global_params["threshold"], len(set(best_labels))])

        # 使用最佳参数重新进行聚类
        print(f"global_best_params: {global_best_params}, 'eps': {final_eps},"
              f"global_best_score: {global_best_score}, clustering_num: {len(set(global_best_labels))}")

        # 展示搜索过程
        #print("len(global_param_grid[threshold]): ", len(global_param_grid["threshold"]))
        #print("len(global_param_grid[segment_length])", len(global_param_grid["segment_length"]))
        #print("cluster_num_result: ", cluster_num_result)
        cluster_num_result = [element[2] for element in cluster_num_result]
        cluster_num_result = np.array(cluster_num_result)
        #rows = len(global_param_grid["segment_length"])
        cols = len(global_param_grid["threshold"])
        cluster_num_result = cluster_num_result.reshape(-1, cols)
        #print("cluster_num_result: ", cluster_num_result)

        xindex = np.arange(cols)
        xrange = global_param_grid["threshold"]
        curvelabel = global_param_grid["segment_length"]
        for i, s_len in enumerate(curvelabel):
            plt.plot(cluster_num_result[i], label=int(curvelabel[i]))
        plt.xticks(xindex, xrange)
        #print("xrange:", xrange)
        #plt.ylim([0, 15])
        plt.ylabel("Cluster num")
        plt.xlabel("Threshold (V)")
        plt.legend()
        plt.grid()

        fig_save_flag = True
        if fig_save_flag:
            base_path = "experiment_results/clustering/search_optimization"
            figure_name = vehicle_name + ".jpg"
            savefig_path = os.path.join(base_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()

        # 用于存储样本特征
        final_data = []

        threshold = global_best_params["threshold"]
        seg_length = int(global_best_params["segment_length"])

        for sample in features_dataset:
            sample_id = sample[0]
            sample_data = slice_extract(sample[1], threshold, seg_length)
            if len(sample_data) == 0:
                continue
            final_data.append([sample_id, sample_data])

        final_data = data_reorganization(final_data)

        # 创建对应的标签列表
        labels = [row[0] for row in final_data]
        # 将字典中的样本数据转换为列表
        samples = [row[1] for row in final_data]
        # samples = [row[1:] for row in train_data]
        samples = np.array(samples)
        # print("labels", labels)
        # print("samples", samples)

        # 定义参数范围(网格搜索获取最佳聚类结果)
        param_grid = {
            'eps': [final_eps],
            'min_samples': [1],
            'metric': [time_series_distance_metrics],
        }

        # 定义评估指标
        best_score = -1
        best_params = {}
        best_labels = []

        # 网格搜索
        for params in ParameterGrid(param_grid):
            dbscan = DBSCAN(**params)
            # dbscan = DensityClustering(**params)
            dbscan.fit(samples)

            # 排除没有聚类的情况,以及超出轮廓函数参数范围
            if len(set(dbscan.labels_)) > 1 and len(set(dbscan.labels_)) < len(labels) - 1:
                # score = silhouette_score(reduced_features, dbscan.labels_)
                score = silhouette_score(samples, dbscan.labels_)
                # score = davies_bouldin_score(scaled_data, dbscan.labels_)
                # score = calinski_harabasz_score(scaled_data, dbscan.labels_)      # results is very terrible

                # 更新最佳参数和聚类结果
                if score > best_score:
                    best_score = score
                    best_params = params
        best_dbscan = DBSCAN(**best_params)
        best_dbscan.fit(samples)
        best_labels = best_dbscan.labels_

        # 输出最佳参数、评估指标和聚类结果
        print("Best Parameters: ", best_params)
        print("Best Silhouette Score: ", best_score)
        print("Best Clustering Labels: ", best_labels)

        # 获取聚类结果
        cluster_labels = best_labels

        # 创建字典来存储每个簇的字符
        clusters = {}
        for current_id, cluster_label in zip(labels, cluster_labels):
            if cluster_label not in clusters:
                clusters[cluster_label] = [current_id]
            else:
                clusters[cluster_label].append(current_id)

        # 输出每个簇包含的字符
        for cluster_label, elements in clusters.items():
            if cluster_label == -1:
                print(f"Noise: Samples {elements}")
            else:
                print(f"Cluster {cluster_label}: Samples {elements}")

        # 常见的20个颜色列表
        common_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
                         'gold', 'darkblue', 'darkorange', 'darkgreen', 'darkred', 'darkviolet', 'sienna', 'magenta',
                         'slategray', 'lime']

        added_labels = set()
        refer = final_data[0]
        plt.clf()
        for i, row in enumerate(final_data):
            cc_id = row[0]
            recessive_data = row[1]

            diff = time_series_distance_metrics(row[1], refer[1])
            cluster_num = -1
            for cluster_label, elements in clusters.items():
                if cc_id in elements:
                    cluster_num = cluster_label
            label_name = "ECU " + str(cluster_num)
            pos_x = diff
            pos_y = i
            if label_name not in added_labels:
                plt.scatter(pos_x, pos_y, color=common_colors[cluster_num], label=label_name)
                added_labels.add(label_name)
            else:
                plt.scatter(pos_x, pos_y, color=common_colors[cluster_num])
            plt.annotate(cc_id, (pos_x, pos_y), textcoords="offset points", xytext=(10, -5), ha='center')
            #plt.annotate(cc_id, (pos_x, pos_y), textcoords="offset points", xytext=(20, -5), ha='center')
            #plt.plot([pos_x, pos_x + 1], [pos_y, pos_y], color=common_colors[cluster_num], linestyle='--')
        plt.xlabel("Related distance")
        plt.ylabel("ID")
        plt.legend()
        #plt.grid()
        if fig_save_flag:
            plt.close()
        else:
            plt.show()


def plotly_validation_example():
    dataset_path = "ECUPrint_dataset"
    file_paths = os.listdir(dataset_path)
    # print(file_paths)

    count = 0
    jump_flag = 0

    # Traverse each file path
    for file_path in file_paths:

        if jump_flag == 1:
            break

        print("vehicle:", file_path)
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
        if file_path == "Ford Kuga":
            print("vehicle:", file_path)
        else:
            continue

        # special folder formats, handled separately
        if file_path == "Ford Fiesta":
            file_path = file_path + "/1_0min/"
            #file_path = file_path + "/2_10min/"

        # special folder formats, handled separately
        if file_path == "Honda Civic":
            #file_path = file_path + "/1_0min/"
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
            #if count > 3000:
            #    jump_flag = 1
            #    break

        # print(vehicle_name + " features_dataset:")
        # print(features_dataset)

        plt_sto_flag = 0
        if plt_sto_flag == 1:

            plt.clf()
            for current_id, segment in features_dataset:
                plt.plot(np.arange(len(segment)), segment)
            plt.show()
            # 指定路径
            #base_path = "experiment_results/segment_plt"
            #figure_name = vehicle_name + "_differential.jpg"
            #savefig_path = os.path.join(base_path, figure_name)
            #plt.savefig(savefig_path, dpi=300)
            #plt.close()

            plt.clf()
            for current_id, segment in features_dataset:
                plt.plot(np.arange(len(np.gradient(segment))), np.gradient(segment))
            plt.show()
            # 指定路径
            #base_path = "experiment_results/segment_plt"
            #figure_name = vehicle_name + "_gradient.jpg"
            #savefig_path = os.path.join(base_path, figure_name)
            #plt.savefig(savefig_path, dpi=300)
            #plt.close()

        # clustering
        train_data = features_dataset
        train_data = data_reorganization(train_data)     # average representation of samples

        #print(train_data)
        #print("len(data)", len(train_data))
        # 创建对应的标签列表
        labels = [row[0] for row in train_data]
        # 将字典中的样本数据转换为列表
        samples = [row[1] for row in train_data]
        samples = np.array(samples)
        print(samples.shape)
        # print("labels", labels)
        # print("samples", samples)

        dynamic_plot_flag = 1
        if dynamic_plot_flag == 1:
            id_list = []
            # 创建图表数据
            plt_data = []
            sub_data_range = []

            # plot 多图
            for row in train_data:
                # print(row)
                # 使用eval函数解析数据
                # 解析数据
                cc_id = row[0]
                voltage_data = row[1]
                # print(voltage_data)

                # 将浮点数转换为NumPy数组
                data_array = np.array(voltage_data, dtype=float)

                # print(data_array)
                # print(data_array.shape)
                index = np.arange(data_array.shape[0])
                # plt.plot(index, recessive_data, color='b', label='recessive_data')
                #plt.plot(index, data_array)

                # 相同ID画一次图就可以了
                if cc_id not in id_list:
                    id_list.append(cc_id)
                    plt_data.append(go.Scatter(x=index, y=data_array, name=cc_id))
                    # plt_data.append(go.Scatter(x=index, y=data_array))
                    sub_data_range.append(data_array.shape[0])

            # 交互动态界面
            # 创建图表数据
            # 创建布局
            layout = go.Layout(
                showlegend=True,
                updatemenus=[
                    {
                        'buttons': [
                            {
                                'method': 'update',
                                'label': label,
                                'args': [{'visible': [index == i for i in range(len(sub_data_range))]}]
                            } for index, label in enumerate([f'Curve {i + 1}' for i in range(len(sub_data_range))])
                        ],
                        'direction': 'down',
                        'showactive': True,
                    }
                ]
            )
            # 创建图表对象
            fig = go.Figure(data=plt_data, layout=layout)
            # 显示图表
            fig.show()


def main():
    #ecu_clustering()                                  # 每辆车单独聚类
    #ecu_clustering_validation()                       # 聚类结果单独验证  plot the evidence for the Ford Kuga vehicle
    #ecu_clustering_search_optimization()              # 暴力遍历参数，得出最佳聚类结果

    plotly_validation_example()                        # validation example by plotly library


if __name__ == '__main__':
    main()









