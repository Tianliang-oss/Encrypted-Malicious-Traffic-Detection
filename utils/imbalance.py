import os
from scapy.all import *
from imblearn.over_sampling import SMOTE
from scapy.layers.inet import TCP, Ether, IP
import numpy as np


def count_samples_in_folders(folder_path):
    """
    统计每个子文件夹中样本的数量
    """
    sample_counts = {}
    for folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            sample_counts[folder] = len(os.listdir(folder_full_path))
    return sample_counts


def calculate_oversampling_amounts(sample_counts):
    """
    计算每个类别需要生成的样本数量
    """
    max_samples = max(sample_counts.values())
    oversampling_amounts = {}
    for class_name, count in sample_counts.items():
        oversampling_amounts[class_name] = max_samples - count
    return oversampling_amounts





if __name__ == "__main__":
    pcap_flow_dir = "E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/pcap_flow_dir/android"

    # 1. 统计每个子文件夹中样本的数量
    sample_counts = count_samples_in_folders(pcap_flow_dir)
    print("Sample counts:", sample_counts)

    # 2. 计算每个类别需要生成的样本数量
    oversampling_amounts = calculate_oversampling_amounts(sample_counts)
    print("Oversampling amounts:", oversampling_amounts)
