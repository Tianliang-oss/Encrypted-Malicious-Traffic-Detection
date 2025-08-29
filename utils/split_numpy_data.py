
"""
@Description: 分割numpy data，划分训练测试集合
"""

# -*- coding: utf-8 -*-
# @Time : 2023/5/17 15:47
# @Author :
# @Email :
# @File : split_all_data.py
"""
@Description: 按照训练与测试占比划分数据，存在本地文件夹里面
"""
from log.set_log import logger
import os
from sklearn.model_selection import train_test_split
import numpy as np


def split_data(pay_data, seq_data,label_data, train_size, npy_path):
    """
     将数据集拆分为训练集和测试集，并保存为 NumPy 文件

     Args:
     - pay_data: NumPy 数组，负载数据
     - seq_data: NumPy 数组，序列长度数据
     - label_data: NumPy 数组，标签数据
     - train_size: 浮点数，训练集所占比例
     - npy_path: 字符串，保存 NumPy 文件的路径

     Returns:
     无
     """
    pay_train, pay_test, seq_train,seq_test,label_train, label_test = train_test_split(pay_data, seq_data,label_data, train_size=train_size)
    os.makedirs(os.path.join(npy_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(npy_path, 'test'), exist_ok=True)
    np.save(os.path.join(npy_path, 'train/', 'pay_load.npy'), pay_train)
    np.save(os.path.join(npy_path, 'test/', 'pay_load.npy'), pay_test)
    np.save(os.path.join(npy_path, 'train/', 'ip_length.npy'), seq_train)
    np.save(os.path.join(npy_path, 'test/', 'ip_length.npy'), seq_test)
    np.save(os.path.join(npy_path, 'train/', 'label.npy'), label_train)
    np.save(os.path.join(npy_path, 'test/', 'label.npy'), label_test)
    logger.info("数据划分为训练集与测试集，训练集比例为{}".format(train_size))


def split_data_with_spiltCap(pay_data, seq_data, sta_data, label_data, train_size, npy_path):
    pay_train, pay_test, seq_train, seq_test, sta_train, sta_test, label_train, label_test = train_test_split(pay_data, seq_data, sta_data,label_data,
                                                                                         train_size=train_size)
    os.makedirs(os.path.join(npy_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(npy_path, 'test'), exist_ok=True)

    np.save(os.path.join(npy_path, 'train/', 'pay_load.npy'), pay_train)
    np.save(os.path.join(npy_path, 'test/', 'pay_load.npy'), pay_test)
    np.save(os.path.join(npy_path, 'train/', 'ip_length.npy'), seq_train)
    np.save(os.path.join(npy_path, 'test/', 'ip_length.npy'), seq_test)
    np.save(os.path.join(npy_path, 'train/', 'statistic.npy'), sta_train)
    np.save(os.path.join(npy_path, 'test/', 'statistic.npy'), sta_test)
    np.save(os.path.join(npy_path, 'train/', 'label.npy'), label_train)
    np.save(os.path.join(npy_path, 'test/', 'label.npy'), label_test)
    logger.info("数据划分为训练集与测试集，训练集比例为{}".format(train_size))


def split_data_with_splitCap1(pay_data, seq_data, sta_data, label_data, train_size, valid_size, npy_path):
    # Splitting data into train, validation, and test sets
    pay_train, pay_temp, seq_train, seq_temp, sta_train, sta_temp, label_train, label_temp = train_test_split(pay_data,
                                                                                                              seq_data,
                                                                                                              sta_data,
                                                                                                              label_data,
                                                                                                              train_size=train_size)
    pay_valid, pay_test, seq_valid, seq_test, sta_valid, sta_test, label_valid, label_test = train_test_split(pay_temp,
                                                                                                              seq_temp,
                                                                                                              sta_temp,
                                                                                                              label_temp,
                                                                                                              test_size=(
                                                                                                                          valid_size / (
                                                                                                                              1 - train_size)))

    # Creating directories
    os.makedirs(os.path.join(npy_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(npy_path, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(npy_path, 'test'), exist_ok=True)

    # Saving data arrays
    np.save(os.path.join(npy_path, 'train', 'pay_load.npy'), pay_train)
    np.save(os.path.join(npy_path, 'valid', 'pay_load.npy'), pay_valid)
    np.save(os.path.join(npy_path, 'test', 'pay_load.npy'), pay_test)

    np.save(os.path.join(npy_path, 'train', 'ip_length.npy'), seq_train)
    np.save(os.path.join(npy_path, 'valid', 'ip_length.npy'), seq_valid)
    np.save(os.path.join(npy_path, 'test', 'ip_length.npy'), seq_test)

    np.save(os.path.join(npy_path, 'train', 'statistic.npy'), sta_train)
    np.save(os.path.join(npy_path, 'valid', 'statistic.npy'), sta_valid)
    np.save(os.path.join(npy_path, 'test', 'statistic.npy'), sta_test)

    np.save(os.path.join(npy_path, 'train', 'label.npy'), label_train)
    np.save(os.path.join(npy_path, 'valid', 'label.npy'), label_valid)
    np.save(os.path.join(npy_path, 'test', 'label.npy'), label_test)

    # Logging
    logger.info(
        "数据划分为训练集、验证集和测试集，训练集比例为{}，验证集比例为{}，测试集比例为{}".format(train_size, valid_size, int(1 - train_size - valid_size)))