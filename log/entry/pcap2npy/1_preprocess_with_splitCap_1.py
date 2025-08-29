# -*- coding: utf-8 -*-
# @Time : 2024/3/10 10:18
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : 1_preprocess_with_splitCap_1.py
"""
@Description: 使用splitCap分流工具加统计特征提取
"""


import sys
import os

from preprocess.process_pcap_with_splitCap_1 import split_pcap_2_session, clipping, getPcapMesg, normalization
from utils.split_numpy_data import split_data, split_data_with_spiltCap, split_data_with_splitCap1

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from utils.set_config import setup_config

from log.set_log import logger

def main():
    yaml_path = r"E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/configuration/traffic_classification_configuration.yaml"
    cfg = setup_config(yaml_path) # 获取 config 文件
    logger.info("begin")
    process(
        pcap_path=cfg.preprocess.traffic_path,
        work_flow_data_dir=cfg.preprocess.splitCap_1.work_flow_path,
        tool_path=cfg.preprocess.splitCap_1.splitCap_exe_path,
        npy_path=cfg.preprocess.splitCap_1.datasets,
        train_size=cfg.preprocess.train_size,
        valid_size=cfg.preprocess.valid_size,
        threshold=cfg.preprocess.threshold,
        ip_length=cfg.preprocess.ip_length,
        n=cfg.preprocess.packet_num,
        m=cfg.preprocess.byte_num,
    )
    logger.info("over!")


def process(pcap_path, work_flow_data_dir, tool_path,
            npy_path, train_size, valid_size, threshold, ip_length, n, m):
    """
    1. 切割会话
    2. 提取特征
    3. 归一化
    4. 存储为npy文件
    5. 切分数据集
    """
    """
    以会话为单位切割pcap
pcap_flow_dir
│
└── android
    ├── file1.pcap
    ├── file2.pcap
    └── file3.pcap
转换为
pcap_flow_dir
│
└── android
    ├── file1
    │   ├── session1
    │   ├── session2
    │   └── ...
    ├── file2
    │   ├── session1
    │   ├── session2
    │   └── ...
    ├── file3
    │   ├── session1
    │   ├── session2
    │   └── ...
    └── ...

    """
    split_pcap_2_session(pcap_path, work_flow_data_dir, tool_path)

    """
       将子文件夹中的文件移动到上级目录，并删除子文件夹。
       pcap_flow_dir
   │
   ├── app1
   │   ├── flow1
   │   │   ├── file1.pcap
   │   │   ├── file2.pcap
   │   │   └── file3.pcap
   │   └── flow2
   │       ├── file4.pcap
   │       └── file5.pcap
   │
   └── app2
       ├── flow3
       │   ├── file6.pcap
       │   └── file7.pcap
       └── flow4
           └── file8.pcap
   转换为
   pcap_flow_dir
   │
   ├── app1
   │   ├── file1.pcap
   │   ├── file2.pcap
   │   ├── file3.pcap
   │   ├── file4.pcap
   │   └── file5.pcap
   │
   └── app2
       ├── file6.pcap
       ├── file7.pcap
       └── file8.pcap

       """
    clipping(work_flow_data_dir)



    """
该函数从给定的 pcap 文件夹中提取序列长度、前 packet_num 个报文的前 byte_num 字节以及统计特征，并将这些信息存储在列表中。

参数：
- pcap_folder：字符串，包含 pcap 文件的文件夹路径。
- threshold：整数，用于过滤流的最小长度阈值。
- ip_length：整数，限制 IP 长度的最大值。
- packet_num：整数，提取的包数量。
- byte_num：整数，每个包提取的字节数。

返回值：
- pay_list：numpy 数组，负载数据列表。
- seq_list：numpy 数组，序列长度列表。
- statistic_list：numpy 数组，统计特征列表。
- label_list：numpy 数组，标签列表。
   """
    pay, seq, sta, label = getPcapMesg(work_flow_data_dir, threshold, ip_length, n, m)
    # 数据归一化
    pay, seq, sta = normalization(pay, seq, sta)
    # 分割数据集
    split_data_with_spiltCap(pay, seq, sta, label, train_size, npy_path)
    #split_data_with_splitCap1(pay, seq, sta, label, train_size,valid_size,npy_path)
    logger.info("over!")

if __name__ == "__main__":
    main()
