import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import _tree
from sklearn.tree import export_text
from itertools import combinations, product
from sklearn.metrics import accuracy_score
from common import *


# baseline, feature derive from EASI
class Baseline:
    def __init__(self):
        self.model = None

    def feature_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        #data_len = len(sample)
        # print(data_len, type(data_len))

        # Initially, the first rising edge was detected by a voltage rise above 0.2V
        # the recording was made at a distance of 500 ns.

        start_point = 0.2
        record_distance = 500   # the recording was made at a distance of 500 ns.
        sampling_interval = 2   # sampling interval is 2ns = 0.002 us in the ECUPrint datatset
        #number_of_points = int(record_distance/sampling_interval)
        #number_of_points = int(2 * record_distance / sampling_interval)
        number_of_points = 500

        # 计算一阶导数
        first_derivative = np.gradient(data)

        segment_index = np.where((data > start_point) & (first_derivative >= 0))[0]
        # print("segment_index: ", segment_index)

        idx00 = segment_index[0]
        idx01 = idx00 + number_of_points
        rising_edge = data[idx00:idx01]

        #plt.clf()
        #index = np.arange(len(rising_edge))
        #plt.plot(index, rising_edge, color="r", label='Differential')
        #plt.legend()
        #plt.xlabel('index')
        #plt.ylabel('Voltage (V)')
        # 显示图形
        #plt.show()

        # feature extraction
        mean = np.mean(rising_edge)
        variance = np.var(rising_edge)
        power = np.sum(np.square(rising_edge))
        skewness = skew(rising_edge)
        kurt = kurtosis(rising_edge)
        maximum = np.max(rising_edge)
        len_rising_edge = len(rising_edge)
        plateau = len_rising_edge/4 * np.sum(rising_edge[int(len_rising_edge*3/4):])
        ratio_max_plat = maximum/plateau
        overshoot = maximum - plateau
        #rising_edge_fft = np.fft.fft(rising_edge)
        #rising_edge_fftfreq = np.fft.fftfreq(len_rising_edge, 2e-9)   # sampling interval is 2ns = 0.002 us in the ECUPrint datatset
        #irregularity = np.sum(np.square(rising_edge_fft[:-1] - rising_edge_fft[1:]))/np.sum(np.square(rising_edge_fft[:-1]))
        #centroid = (np.sum(rising_edge_fft * rising_edge_fftfreq))/(np.sum(rising_edge_fft))
        #flatness = np.sum(rising_edge_fft * (np.power(np.product(rising_edge_fft), 1/len(rising_edge_fft))/np.sum(rising_edge_fft)) )

        #return ratio_max_plat, skewness, plateau, kurt, overshoot, irregularity, centroid, flatness, mean, variance, power, maximum
        return ratio_max_plat, skewness, plateau, kurt, overshoot, mean, variance, power, maximum

    def fit(self, X, y):
        # feature extraction
        feature_data = [self.feature_extract(sample) for sample in X]

        clf = LogisticRegression(max_iter=3000)
        #clf = LogisticRegression()
        clf.fit(feature_data, y)
        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.feature_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred


class BaselineOptimization:
    def __init__(self):
        self.model = None

    def feature_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        #data_len = len(sample)
        # print(data_len, type(data_len))

        # Initially, the first rising edge was detected by a voltage rise above 0.2V
        # the recording was made at a distance of 500 ns.

        start_point = 0.2
        record_distance = 500   # the recording was made at a distance of 500 ns.
        sampling_interval = 2   # sampling interval is 2ns = 0.002 us in the ECUPrint datatset
        #number_of_points = int(record_distance/sampling_interval)
        #number_of_points = int(2 * record_distance / sampling_interval)
        number_of_points = 500

        # 计算一阶导数
        first_derivative = np.gradient(data)

        segment_index = np.where((data > start_point) & (first_derivative >= 0))[0]
        # print("segment_index: ", segment_index)

        idx00 = segment_index[0]
        idx01 = idx00 + number_of_points
        rising_edge = data[idx00:idx01]

        #plt.clf()
        #index = np.arange(len(rising_edge))
        #plt.plot(index, rising_edge, color="r", label='Differential')
        #plt.legend()
        #plt.xlabel('index')
        #plt.ylabel('Voltage (V)')
        # 显示图形
        #plt.show()

        # feature extraction
        mean = np.mean(rising_edge)
        variance = np.var(rising_edge)
        power = np.sum(np.square(rising_edge))
        skewness = skew(rising_edge)
        kurt = kurtosis(rising_edge)
        maximum = np.max(rising_edge)
        len_rising_edge = len(rising_edge)
        plateau = len_rising_edge/4 * np.sum(rising_edge[int(len_rising_edge*3/4):])
        ratio_max_plat = maximum/plateau
        overshoot = maximum - plateau
        #rising_edge_fft = np.fft.fft(rising_edge)
        #rising_edge_fftfreq = np.fft.fftfreq(len_rising_edge, 2e-9)   # sampling interval is 2ns = 0.002 us in the ECUPrint datatset
        #irregularity = np.sum(np.square(rising_edge_fft[:-1] - rising_edge_fft[1:]))/np.sum(np.square(rising_edge_fft[:-1]))
        #centroid = (np.sum(rising_edge_fft * rising_edge_fftfreq))/(np.sum(rising_edge_fft))
        #flatness = np.sum(rising_edge_fft * (np.power(np.product(rising_edge_fft), 1/len(rising_edge_fft))/np.sum(rising_edge_fft)) )

        #return ratio_max_plat, skewness, plateau, kurt, overshoot, irregularity, centroid, flatness, mean, variance, power, maximum
        return ratio_max_plat, skewness, plateau, kurt, overshoot, mean, variance, power, maximum

    def fit(self, X, y):
        # feature extraction
        feature_data = [self.feature_extract(sample) for sample in X]
        #clf = DecisionTreeClassifier()
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(feature_data, y)
        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.feature_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred


class ECUPrint:
    def __init__(self):
        self.model = None

    def feature_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        data_len = len(sample)
        # print(data_len, type(data_len))

        fixed_range = 150
        sigma = 0.002   # σ = 2ns = 0.002 us
        epsilon = 0.02  # epsilon = 20 mv = 0.02 v

        # mid_index = int(data_len / 2) - 1
        mid_index = int(np.floor(data_len / 2))
        start_index = mid_index - fixed_range
        end_index = mid_index + fixed_range
        # print(data_len, mid_index, start_index, end_index)

        v_mean = np.mean(data[start_index:end_index + 1])
        v_max = np.max(data[:start_index + 1])

        min_left = np.min(data[:mid_index])
        min_right = np.min(data[mid_index:])
        t_bit_alpha_index = np.where(np.abs(data - min_left) <= epsilon)
        t_bit_beta_index = np.where(np.abs(data - min_right) <= epsilon)
        t_bit_alpha = t_bit_alpha_index[0][t_bit_alpha_index[0] <= mid_index]
        t_bit_beta = t_bit_beta_index[0][t_bit_beta_index[0] > mid_index]
        t_bit = (t_bit_beta[0] - t_bit_alpha[-1]) * sigma  # min(beta - alpha) = min(beta) - max(alpha)

        t_plat_alpha_beta_index = np.where(np.abs(data - v_mean) <= epsilon)
        t_plat_alpha = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] <= mid_index]
        t_plat_beta = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] > mid_index]
        t_plat = (t_plat_beta[-1] - t_plat_alpha[0]) * sigma  # max(beta - alpha) = max(beta) - min(alpha)

        # print(f"v_mean: {v_mean}, v_max: {v_max}, t_bit: {t_bit}, t_plat: {t_plat}")

        return v_mean, v_max, t_bit, t_plat

    def fit(self, X, y):
        # feature extraction
        #print("dataset", X)
        feature_data = [self.feature_extract(sample) for sample in X]

        clf = LogisticRegression(max_iter=3000)
        #clf = LogisticRegression()
        clf.fit(feature_data, y)
        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.feature_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred


class ECUPrintOptimization:
    def __init__(self):
        self.model = None

    def feature_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        data_len = len(sample)
        # print(data_len, type(data_len))

        fixed_range = 150
        sigma = 0.002   # σ = 2ns = 0.002 us
        epsilon = 0.02  # epsilon = 20 mv = 0.02 v

        # mid_index = int(data_len / 2) - 1
        mid_index = int(np.floor(data_len / 2))
        start_index = mid_index - fixed_range
        end_index = mid_index + fixed_range
        # print(data_len, mid_index, start_index, end_index)

        v_mean = np.mean(data[start_index:end_index + 1])
        v_max = np.max(data[:start_index + 1])

        min_left = np.min(data[:mid_index])
        min_right = np.min(data[mid_index:])
        t_bit_alpha_index = np.where(np.abs(data - min_left) <= epsilon)
        t_bit_beta_index = np.where(np.abs(data - min_right) <= epsilon)
        t_bit_alpha = t_bit_alpha_index[0][t_bit_alpha_index[0] <= mid_index]
        t_bit_beta = t_bit_beta_index[0][t_bit_beta_index[0] > mid_index]
        t_bit = (t_bit_beta[0] - t_bit_alpha[-1]) * sigma  # min(beta - alpha) = min(beta) - max(alpha)

        t_plat_alpha_beta_index = np.where(np.abs(data - v_mean) <= epsilon)
        t_plat_alpha = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] <= mid_index]
        t_plat_beta = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] > mid_index]
        t_plat = (t_plat_beta[-1] - t_plat_alpha[0]) * sigma  # max(beta - alpha) = max(beta) - min(alpha)

        # print(f"v_mean: {v_mean}, v_max: {v_max}, t_bit: {t_bit}, t_plat: {t_plat}")

        return v_mean, v_max, t_bit, t_plat

    def fit(self, X, y):
        # feature extraction
        #print("dataset", X)
        feature_data = [self.feature_extract(sample) for sample in X]
        #clf = DecisionTreeClassifier()
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(feature_data, y)
        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.feature_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred


# voltage fragment inspector
class VInspector:
    def __init__(self, segment_threshold=1.7, segment_length=200):
        self.model = None
        self.segment_threshold = segment_threshold
        self.segment_length = segment_length
        self.start_point = 0
        self.end_point = 200

    def segment_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        # data_len = len(sample)
        # print(data_len, type(data_len))

        segment_threshold = self.segment_threshold
        segment_length = self.segment_length

        start_point = self.start_point
        end_point = self.end_point

        # 计算一阶导数
        first_derivative = np.gradient(data)

        segment_index = np.where((data > segment_threshold) & (first_derivative >= 0))[0]
        # print("segment_index: ", segment_index)

        idx00 = segment_index[0]
        idx01 = idx00 + segment_length
        voltage_segment = data[idx00:idx01]
        voltage_segment = voltage_segment[start_point:end_point]

        return voltage_segment

    def fit(self, X, y):
        # feature extraction
        feature_data = [self.segment_extract(sample) for sample in X]

        clf = RandomForestClassifier(n_estimators=10)
        #clf = DecisionTreeClassifier()
        clf.fit(feature_data, y)

        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.segment_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred


class VInspectorLR:
    def __init__(self, segment_threshold=1.7, segment_length=200):
        self.model = None
        self.segment_threshold = segment_threshold
        self.segment_length = segment_length
        self.start_point = 0
        self.end_point = 200

    def segment_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        # data_len = len(sample)
        # print(data_len, type(data_len))

        segment_threshold = self.segment_threshold
        segment_length = self.segment_length

        start_point = self.start_point
        end_point = self.end_point

        # 计算一阶导数
        first_derivative = np.gradient(data)

        segment_index = np.where((data > segment_threshold) & (first_derivative >= 0))[0]
        # print("segment_index: ", segment_index)

        idx00 = segment_index[0]
        idx01 = idx00 + segment_length
        voltage_segment = data[idx00:idx01]
        voltage_segment = voltage_segment[start_point:end_point]

        return voltage_segment

    def fit(self, X, y):
        # feature extraction
        feature_data = [self.segment_extract(sample) for sample in X]

        clf = LogisticRegression(max_iter=3000)
        clf.fit(feature_data, y)

        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.segment_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred

class VInspectorGradient:
    def __init__(self, segment_threshold=1.7, segment_length=200):
        self.model = None
        self.segment_threshold = segment_threshold
        self.segment_length = segment_length
        self.start_point = 0
        self.end_point = 200

    def segment_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        # data_len = len(sample)
        # print(data_len, type(data_len))

        segment_threshold = self.segment_threshold
        segment_length = self.segment_length

        start_point = self.start_point
        end_point = self.end_point

        # 计算一阶导数
        first_derivative = np.gradient(data)

        segment_index = np.where((data > segment_threshold) & (first_derivative >= 0))[0]
        # print("segment_index: ", segment_index)

        idx00 = segment_index[0]
        idx01 = idx00 + segment_length
        voltage_segment = first_derivative[idx00:idx01]
        voltage_segment = voltage_segment[start_point:end_point]

        return voltage_segment

    def fit(self, X, y):
        # feature extraction
        feature_data = [self.segment_extract(sample) for sample in X]

        clf = RandomForestClassifier(n_estimators=10)
        #clf = DecisionTreeClassifier()
        clf.fit(feature_data, y)

        self.model = clf

        return self

    def predict(self, X):

        # feature extraction
        feature_data = [self.segment_extract(sample) for sample in X]

        # load_model
        clf = self.model

        # predict
        y_pred = clf.predict(feature_data)

        return y_pred


# 一维卷积神经网络
class CNN1D:
    def __init__(self, segment_threshold=1.7, segment_length=200, epochs=30, batch_size=32, learning_rate=0.001):
        self.model = None
        self.segment_threshold = segment_threshold
        self.segment_length = segment_length
        self.start_point = 0
        self.end_point = 200

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def segment_extract(self, sample):
        data = sample
        data = sliding_average(data, 10)
        # data_len = len(sample)
        # print(data_len, type(data_len))

        segment_threshold = self.segment_threshold
        segment_length = self.segment_length

        start_point = self.start_point
        end_point = self.end_point

        # 计算一阶导数
        first_derivative = np.gradient(data)

        segment_index = np.where((data > segment_threshold) & (first_derivative >= 0))[0]
        # print("segment_index: ", segment_index)

        idx00 = segment_index[0]
        idx01 = idx00 + segment_length
        voltage_segment = data[idx00:idx01]
        voltage_segment = voltage_segment[start_point:end_point]

        return voltage_segment

    def fit(self, X, y):
        # feature extraction
        feature_data = np.array([self.segment_extract(sample) for sample in X])

        num_classes = len(np.unique(y))

        # 设置随机种子(确保结果可复现)
        random_seed = 77
        np.random.seed(random_seed)

        # 创建模型
        model = Sequential()
        # 第一个卷积层
        model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(feature_data.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))             # pool_size=2, kernel_size=3, filters=32
        model.add(BatchNormalization())
        # 第二个卷积层
        #model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        #model.add(BatchNormalization())

        model.add(Flatten())
        # 全连接层
        # model.add(Dense(64, activation='relu'))
        # Dropout层, 以防止过拟合
        # model.add(Dropout(0.5))
        #model.add(Dense(32, activation='relu'))
        # Dropout层, 以防止过拟合
        #model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # 模型信息汇总
        model.summary()

        # 模型编译
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # 模型训练
        # 由于标签数据是整数形式,我们需要将其转换为独热编码格式
        y_train_categorical = to_categorical(y, num_classes=num_classes)
        model.fit(feature_data[:, :, np.newaxis], y_train_categorical, epochs=self.epochs, batch_size=self.batch_size)

        self.model = model

        return self

    def predict(self, X):

        # feature extraction
        feature_data = np.array([self.segment_extract(sample) for sample in X])

        # load_model
        cnn_model = self.model

        # predict
        # 将独热编码格式的标签数据转换为整数形式
        y_test_categorical = cnn_model.predict(feature_data)
        y_pred = np.argmax(y_test_categorical, axis=1)

        return y_pred