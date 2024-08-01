from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dropout, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from keras.models import save_model
from keras.models import load_model
from keras.utils import to_categorical

import random
import re


anomaly_threshold = 0.95

parameter_function = {
    "Dacia Duster": "differential",
    "Dacia Logan": "differential",
    "Ford Ecosport": "differential",
    "Ford Fiesta": "gradient",         # gradient, differential
    "Ford Kuga": "differential",           # gradient, differential
    "Honda Civic": "gradient",         # gradient, differential
    "Hyundai i20": "differential",     # gradient, differential
    "Hyundai ix35": "gradient",    # gradient, differential
    "John Deere Tractor": "gradient",  # gradient, differential
    "Opel Corsa": "differential",
}

# Dacia Duster-optical, 'segment_length': 300.0, 'threshold': 1.85, 'eps': 0.59, 'score': 0.9099135142916587, 'cluster num': 3
# Dacia Logan-optical, 'segment_length': 100.0, 'threshold': 1.9, 'eps': 0.59, 'score': 0.9632335096952301, 'cluster num': 6
# Ford Ecosport-optical, 'segment_length': 200.0, 'threshold': 1.85, 'eps': 1.0, 'score': 0.9516835234756879, 'cluster num': 4
# Ford Fiesta-optical, 'segment_length': 400.0, 'threshold': 1.9, 'eps': 1.0, 'score': 0.8816763060535485, 'cluster num': 6 (不稳定)
# Ford Kuga-optical, 'segment_length': 300.0, 'threshold': 1.75, 'eps': 0.87, score: 0.8161411458340444, clustering_num: 12
# John Deere Tractor-optical, 'segment_length': 200.0, 'threshold': 1.9, 'eps': 1.64,global_best_score: 0.6954570686244338, clustering_num: 2
# Hyundai ix35-optical, 'segment_length': 100.0, 'threshold': 1.7, 'eps': 1.6, score: 0.8326926007851843, clustering_num: 2

# Dacia Logan-'segment_length':400.0, 'threshold': 1.9, 'eps': 1.61, 'cluster num': 7
# Ford Fiesta-'segment_length':400.0, 'threshold': 1.9, 'eps': 1.7, 'cluster num': 6
# Honda Civic-'segment_length': 400.0, 'threshold': 1.85, 'eps': 1.92, 'cluster num': 7
# John Deere Tractor, 'segment_length': 200.0, 'threshold': 1.9, 'eps': 1.64, clustering_num: 3
# Hyundai ix35, 'segment_length': 100.0, 'threshold': 1.85, 'eps': 0.48, clustering_num: 7

parameter_thresholds = {
    "Dacia Duster": 1.7,        # 1.7, 1.85
    "Dacia Logan": 1.7,         # 1.7
    "Ford Ecosport": 1.7,
    "Ford Fiesta": 1.7,         # 1.7
    "Ford Kuga": 1.8,           # 1.8
    "Honda Civic": 1.5,         # 1.5, 1.7
    "Hyundai i20": 1.8,         # 1.8, 1.7
    "Hyundai ix35": 1.85,        # 1.8, 1.7
    "John Deere Tractor": 1.75,  # 1.75
    "Opel Corsa": 1.8,          # 1.8
}

parameter_length = {
    "Dacia Duster": 100,        # 100, 300    optical: 300
    "Dacia Logan": 200,         # 100, 200
    "Ford Ecosport": 100,       # 100
    "Ford Fiesta": 200,         # 200
    "Ford Kuga": 200,           # 200
    "Honda Civic": 200,         # 200
    "Hyundai i20": 400,         # 200
    "Hyundai ix35": 200,
    "John Deere Tractor": 100,  # 100
    "Opel Corsa": 100,          # 100
}


def data_reorganization(data):
    '''
    # 根据样本特征值计算特征值mean or median
    average representation of samples
    :param data:
    :return:
    '''
    averages_dict = {}

    # 遍历每一行数据
    for row in data:
        key = row[0]  # 第一列作为字典的键
        values = row[1:]  # 其他列作为字典的值

        if key not in averages_dict:
            averages_dict[key] = []
        averages_dict[key].append(values)

    # 计算平均值
    for key in averages_dict:
        values = np.array(averages_dict[key])
        #averages_dict[key] = np.mean(values, axis=0)
        averages_dict[key] = np.median(values, axis=0)

    # 重组成一行
    result = [[key] + list(averages_dict[key]) for key in averages_dict]

    return result


def segment_extract(data, threshold, segment_length):
    # 提取数据列
    timestamp = [float(row[0]) for row in data]
    # index = np.arange(len(time))
    # channel_a = [float(row[1]) for row in data]
    # channel_b = [float(row[2]) for row in data]
    differential = [float(row[1]) - float(row[2]) for row in data]

    differential = np.array(differential)

    filtered_data = sliding_average(differential, 10)

    # 计算一阶导数
    first_derivative = np.gradient(filtered_data)

    segment_index = np.where((filtered_data > threshold) & (first_derivative >= 0))[0]
    #print("segment_index: ", segment_index)
    if len(segment_index) == 0:
        return []

    # ack位电压,去除,异常样本
    if max(differential) > 2.8:
        return []

    idx00 = segment_index[0]
    idx01 = idx00 + segment_length

    return filtered_data[idx00:idx01]


def slice_extract(data, threshold, segment_length):

    filtered_data = sliding_average(data, 10)

    # 计算一阶导数
    first_derivative = np.gradient(data)

    segment_index = np.where((data > threshold) & (first_derivative >= 0))[0]
    #print("segment_index: ", segment_index)
    if len(segment_index) == 0:
        return []

    # ack位电压,去除,异常样本
    if max(data) > 2.8:
        return []

    idx00 = segment_index[0]
    idx01 = idx00 + segment_length
    #idx02 = idx01-50

    return data[idx00:idx01]


def sliding_average(data, window_size):
    # 创建滑动窗口的权重数组
    weights = np.ones(window_size) / window_size

    # 使用边缘填充的方式计算滑动平均
    padding = (window_size - 1) // 2
    padded_data = np.pad(data, (padding, padding), mode='edge')
    averaged_data = np.convolve(padded_data, weights, mode='valid')

    return averaged_data


# find the ecu in the clustering result
def find_ecu(matrix, element):
    for i, sublist in enumerate(matrix):
        if element in sublist:
            return i
    return -1


def time_series_distance_metrics(x1, x2):

    # Manhattan Distance
    distance = l1_norm = np.linalg.norm(x1 - x2, ord=1)

    # 计算欧氏距离euclidean_dist
    #distance = np.linalg.norm(x1 - x2, ord=2)

    return distance


def time_series_distance_metrics2(x1, x2, ratio = 0.5):
    x1_gradient = np.gradient(x1)
    x2_gradient = np.gradient(x2)

    # Manhattan Distance
    distance1 = np.linalg.norm(x1 - x2, ord=1)

    distance2 = np.linalg.norm(x1_gradient - x2_gradient, ord=1)

    x_len = len(x1)
    amplification_factor = 100
    distance = ratio * distance1 + (1 - ratio) * amplification_factor * distance2

    #print(f"distance1: {distance1}, distance2: {distance2}, x_len: {x_len}")

    # 计算欧氏距离euclidean_dist
    #distance = np.linalg.norm(x1 - x2, ord=2)

    return distance


def energy(time_series_data):
    return np.sum(np.square(time_series_data))

def root_mean_square(time_series_data):
    #Root Mean Square, RMS

    return np.sum(np.square(time_series_data))

def downsample_array(arr, ratio):
    # 计算新的采样率
    new_sample_rate = int(1 / ratio)

    # 使用切片操作按照新的采样率截取数组
    downsampled_arr = arr[::new_sample_rate]

    return downsampled_arr


def ecu_print_feature_extract(data, plt_figure = False):
    data_len = len(data)
    #print(data_len, type(data_len))

    fixed_range = 150
    sigma = 0.002  # σ = 2ns = 0.002 us
    epsilon = 0.02  # epsilon = 20 mv = 0.02 v

    # 将浮点数转换为整数
    #mid_index = int(data_len / 2) - 1
    mid_index = int(np.floor(data_len / 2))
    start_index = mid_index - fixed_range
    end_index = mid_index + fixed_range
    #print(data_len, mid_index, start_index, end_index)

    v_mean = np.mean(data[start_index:end_index + 1])
    v_max = np.max(data[:start_index + 1])

    #t_bit_alpha_beta_index = np.where(np.abs(data) <= epsilon)
    #print("t_bit_alpha_beta_index", t_bit_alpha_beta_index)
    #t_bit_alpha = t_bit_alpha_beta_index[0][t_bit_alpha_beta_index[0] <= mid_index]
    #t_bit_beta = t_bit_alpha_beta_index[0][t_bit_alpha_beta_index[0] >= mid_index]
    min_left = np.min(data[:mid_index])
    min_right = np.min(data[mid_index:])
    t_bit_alpha_index = np.where(np.abs(data - min_left) <= epsilon)
    t_bit_beta_index = np.where(np.abs(data - min_right) <= epsilon)
    t_bit_alpha = t_bit_alpha_index[0][t_bit_alpha_index[0] <= mid_index]
    t_bit_beta = t_bit_beta_index[0][t_bit_beta_index[0] > mid_index]
    t_bit = (t_bit_beta[0] - t_bit_alpha[-1]) * sigma                     # min(beta - alpha) = min(beta) - max(alpha)

    t_plat_alpha_beta_index = np.where(np.abs(data - v_mean) <= epsilon)
    t_plat_alpha = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] <= mid_index]
    t_plat_beta = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] > mid_index]
    t_plat = (t_plat_beta[-1] - t_plat_alpha[0]) * sigma                  # max(beta - alpha) = max(beta) - min(alpha)

    #print(f"v_mean: {v_mean}, v_max: {v_max}, t_bit: {t_bit}, t_plat: {t_plat}")

    if plt_figure:
        plt.clf()
        index = np.arange(len(data))
        plt.plot(index, data, color="r", label='Differential')

        # 绘制水平直线 y=5
        plt.axhline(y=v_mean, color='grey', linestyle='--')
        plt.axhline(y=v_max, color='grey', linestyle='--')

        x_values = [t_bit_alpha[-1], t_bit_beta[0]]
        y_values = [data[t_bit_alpha[-1]], data[t_bit_beta[0]]]
        plt.plot(x_values, y_values, color="b", label="t_bit")
        # 绘制垂直直线 x=3
        plt.axvline(x=t_bit_alpha[-1], color='grey', linestyle='--')
        plt.axvline(x=t_bit_beta[0], color='grey', linestyle='--')

        x_values = [t_plat_alpha[0], t_plat_beta[-1]]
        y_values = [data[t_plat_alpha[0]], data[t_plat_beta[-1]]]
        plt.plot(x_values, y_values, color="g", label="t_plat")
        plt.axvline(x=t_plat_alpha[0], color='grey', linestyle='--')
        plt.axvline(x=t_plat_beta[-1], color='grey', linestyle='--')

        # 添加图例、标题和坐标轴标签
        plt.legend()
        plt.xlabel('index')
        plt.ylabel('Voltage (V)')
        # 显示图形
        plt.show()

    return v_mean, v_max, t_bit, t_plat
