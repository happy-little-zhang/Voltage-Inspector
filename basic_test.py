import matplotlib.pyplot as plt
import numpy as np

from my_model import *
from common import *
import time
import csv
import os


def read_interference_test():
    #filename = "ECUPrint_dataset\Ford Kuga\[001]_0B0_extracted_extracted_ZERO_[0].csv"
    filename = "ECUPrint_dataset\Ford Kuga\[040]_0B0_extracted_extracted_ZERO_[0].csv"

    #filename = "ECUPrint_dataset/Honda Civic/ENVIRONMENTAL_4_15min_dynamic/[069]_13C_extracted_extracted_ZERO_[5]_5min_1.csv"

    #electromagnetic interference
    #ECUPrint_dataset\Dacia Duster\[001]_511_extracted_extracted_ZERO_[9]_100.csv
    #ECUPrint_dataset\Ford Ecosport\[001]_04C_extracted_extracted_ZERO_[4].csv
    #ECUPrint_dataset\Ford Ecosport\[001]_04C_extracted_extracted_ZERO_[4].csv
    #ECUPrint_dataset\Ford Kuga\[001]_0B0_extracted_extracted_ZERO_[0].csv
    #"ECUPrint_dataset/Dacia Duster/[001]_1A5_extracted_extracted_ZERO_[1].csv"

    with open(filename, 'r') as file:
        lines = file.readlines()

        special_data = [line.strip() for line in lines[:3]]
        print(special_data)

        reader = csv.reader(lines[5:])
        data = list(reader)

        print(len(data))

        time = [float(row[0]) for row in data]
        channel_a = [float(row[1]) for row in data]
        channel_b = [float(row[2]) for row in data]
        differential = [float(row[1]) - float(row[2]) for row in data]
        index = np.arange(len(differential))

        plt.clf()
        plt.plot(index, channel_a, color="r", label='CAN High')
        plt.plot(index, channel_b, color="b", label='CAN Low')
        plt.legend()
        plt.title(special_data[0])
        plt.xlabel('Index')
        plt.ylabel('Voltage (V)')

        save_plt_flag = False

        if save_plt_flag:
            base_path = "experiment_results/test"
            #figure_name = "Interference.jpg"
            figure_name = "raw_voltage.jpg"
            savefig_path = os.path.join(base_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()

        plt.clf()
        plt.plot(index, differential, color="b", label='Differential')
        plt.legend()
        plt.title(special_data[0])
        plt.xlabel('Index')
        plt.ylabel('Voltage (V)')

        if save_plt_flag:
            base_path = "experiment_results/test"
            #figure_name = "Interference_cancellation.jpg"
            figure_name = "differential.jpg"
            savefig_path = os.path.join(base_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()


def read_single_sample():
    filename1 = "ECUPrint_dataset\Dacia Duster\[001]_1A5_extracted_extracted_ZERO_[0].csv"
    filename2 = "ECUPrint_dataset\Dacia Duster\[001]_244_extracted_extracted_ZERO_[0].csv"

    #filename1 = "ECUPrint_dataset\Dacia Logan\[001]_0C6_extracted_extracted_ZERO_[0]_Logan_1.csv"
    #filename1 = "ECUPrint_dataset\Dacia Logan\[001]_1B0_extracted_extracted_ZERO_[0]_Logan_1.csv"
    #filename1 = "ECUPrint_dataset\Dacia Logan\[001]_1F6_extracted_extracted_ZERO_[0]_Logan_1.csv"
    #filename1 = "ECUPrint_dataset\Dacia Logan\[001]_2A9_extracted_extracted_ZERO_[0]_Logan_1.csv"
    #filename2 = "ECUPrint_dataset\Dacia Logan\[001]_1A0_extracted_extracted_ZERO_[0]_Logan_1.csv"
    #filename2 = "ECUPrint_dataset\Dacia Logan\[001]_1F6_extracted_extracted_ZERO_[0]_Logan_1.csv"


    with open(filename1, 'r') as file:
        lines = file.readlines()

        special_data = [line.strip() for line in lines[:3]]
        print(special_data)
        current_id1 = special_data[0]

        reader = csv.reader(lines[5:])
        data = list(reader)

        print(len(data))

        time1 = [float(row[0]) for row in data]
        channel_a1 = [float(row[1]) for row in data]
        channel_b1 = [float(row[2]) for row in data]
        differential1 = [float(row[1]) - float(row[2]) for row in data]

    with open(filename2, 'r') as file:
        lines = file.readlines()

        special_data = [line.strip() for line in lines[:3]]
        print(special_data)
        current_id2 = special_data[0]

        reader = csv.reader(lines[5:])
        data = list(reader)

        print(len(data))

        time2 = [float(row[0]) for row in data]
        channel_a2 = [float(row[1]) for row in data]
        channel_b2 = [float(row[2]) for row in data]
        differential2 = [float(row[1]) - float(row[2]) for row in data]


    index = np.arange(len(differential1))
    plt.clf()
    plt.plot(index, differential1, color="r", label=current_id1)
    plt.plot(index, differential2, color="b", label=current_id2)

    #plt.Circle((750, 2.0), 0.1, color="g")
    pos = 700
    plt.scatter(pos, differential2[pos], s=2000, c='lime', marker='o')

    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Voltage (V)')

    save_plt_flag = False

    if save_plt_flag:
        base_path = "experiment_results/test"
        figure_name = "Corner_differences.jpg"
        savefig_path = os.path.join(base_path, figure_name)
        plt.savefig(savefig_path, dpi=300)
        plt.close()
    else:
        plt.show()


def my_test():
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
        #if file_path == "Dacia Duster" or file_path == "Dacia Logan":
        #    continue

        if file_path == "Ford Kuga":
            print("vehicle:", file_path)
        else:
            continue

        if file_path == "Ford Fiesta" or file_path == "Honda Civic":
            file_path = file_path + "/1_0min/"

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
                data = list(reader)
                print(len(data))

                # 提取数据列
                time = [float(row[0]) for row in data]
                channel_a = [float(row[1]) for row in data]
                channel_b = [float(row[2]) for row in data]
                differential = [float(row[1]) - float(row[2]) for row in data]

                # 清空画布
                plt.clf()
                # 绘制折线图
                plt.plot(time, channel_a, color="r", label='CAN High')
                plt.plot(time, channel_b, color="b", label='CAN Low')
                # 添加图例、标题和坐标轴标签
                #plt.legend()
                plt.title(special_data[0])
                plt.xlabel('Time (ms)')
                plt.ylabel('Voltage (V)')
                # 显示图形
                plt.show()

                # 清空画布
                plt.clf()
                # 绘制折线图
                plt.plot(time, differential, color="b", label='Differential')
                # 添加图例、标题和坐标轴标签
                #plt.legend()
                plt.title(special_data[0])
                plt.xlabel('Time (ms)')
                plt.ylabel('Voltage (V)')
                # 显示图形
                plt.show()

            count = count + 1
            print(count)
            if count > 1:
                jump_flag = 1
                break


def feature_extraction_overhead():

    features = [
        np.mean,
        np.std,
        np.var,
        skew,
        kurtosis,
        np.max,
        energy,
        root_mean_square,
        np.min,
    ]

    data_size = 50
    repeat_num = 10000
    print(f"data_size: {data_size}, repeat_num: {repeat_num}")
    for feature in features:
        execution_times = []
        for i in range(repeat_num):
            time_series_data = np.random.random(data_size)
            #print(time_series_data)

            start_time = time.time()

            value = feature(time_series_data)

            end_time = time.time()

            execution_time = end_time - start_time
            execution_times.append(execution_time/1e-6)

        average_execution_time = np.mean(execution_times)
        print(f"{feature.__name__} average_execution_time: {average_execution_time} us")


def smooth_filter_overhead():
    data_size = 2000
    repeat_num = 10000
    print(f"data_size: {data_size}, repeat_num: {repeat_num}")

    window_sizes = [5, 10, 20]

    for window_size in window_sizes:
        execution_times = []
        for i in range(repeat_num):
            time_series_data = np.random.random(data_size)
            # print(time_series_data)

            start_time = time.time()

            time_series_data = sliding_average(time_series_data, window_size)

            end_time = time.time()

            execution_time = end_time - start_time
            execution_times.append(execution_time / 1e-6)

        average_execution_time = np.mean(execution_times)
        print(f"window_size: {window_size}, average_execution_time: {average_execution_time} us")

def main():
    #read_interference_test()
    #read_single_sample()
    #my_test()

    #feature_extraction_overhead()
    smooth_filter_overhead()             # 测试滑动滤波的时间

if __name__ == '__main__':
    main()