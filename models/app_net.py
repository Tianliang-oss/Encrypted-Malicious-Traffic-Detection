
"""
@Description: APP_Net模型
"""
import sys

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.util.FeaturesCalc import FeaturesCalc
import torch
import torch.nn as nn
from models.Base_Model import BaseModel


# class APP_Net(BaseModel):
#     def __init__(self, input_size = 1, hidden_size = 256, num_layers = 2, bidirectional = True, num_classes=12):
#         super(APP_Net, self).__init__()
#         # rnn配置
#         self.bidirectional = bidirectional
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, bidirectional=bidirectional, batch_first=True)
#         self.fc0 = nn.Linear(hidden_size, num_classes)
#         self.fc1 = nn.Linear(hidden_size * 2, num_classes)
#
#         self.cnn_feature = nn.Sequential(
#             # 卷积层1
#             nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),  # (1,1024)->(32,1024)
#             nn.BatchNorm1d(32),  # 加上BN的结果
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (32,1024)->(32,342)
#
#             # 卷积层2
#             nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),  # (32,342)->(64,342)
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,342)->(64,114)
#         )
#         # 全连接层
#         self.cnn_classifier = nn.Sequential(
#             # 64*114
#             nn.Flatten(),
#             nn.Linear(in_features=64 * 114, out_features=1024),  # 784:88*64, 1024:114*64, 4096:456*64
#         )
#
#         self.cnn = nn.Sequential(
#             self.cnn_feature,
#             self.cnn_classifier,
#         )
#     # def __init__(self, input_size=1, hidden_size=256, num_layers=2, bidirectional=True, num_classes=12):
#     #     super(APP_Net, self).__init__()
#     #     # RNN配置
#     #     self.bidirectional = bidirectional
#     #     self.hidden_size = hidden_size
#     #     self.num_layers = num_layers
#     #     self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
#     #                         batch_first=True)
#     #     self.fc0 = nn.Linear(hidden_size, num_classes)
#     #     self.fc1 = nn.Linear(hidden_size * 2, num_classes)
#     #
#     #     # CNN卷积层部分
#     #     self.cnn_feature = nn.Sequential(
#     #         # 卷积层1
#     #         nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),  # (1,1024)->(32,1024)
#     #         nn.BatchNorm1d(32),  # 加上BN的结果
#     #         nn.ReLU(),
#     #         nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (32,1024)->(32,342)
#     #
#     #         # 卷积层2
#     #         nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),  # (32,342)->(64,342)
#     #         nn.BatchNorm1d(64),
#     #         nn.ReLU(),
#     #         nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,342)->(64,114)
#     #     )
#     #
#     #     # 全连接层部分
#     #     self.cnn_classifier = nn.Sequential(
#     #         nn.Flatten(),
#     #         nn.Linear(in_features=64 * 114, out_features=1024),  # 64*114
#     #         nn.BatchNorm1d(1024),  # 在全连接层后加上BN层
#     #         nn.ReLU(),  # 也可以增加ReLU激活函数
#     #     )
#     #
#     #     # 将CNN组合
#     #     self.cnn = nn.Sequential(
#     #         self.cnn_feature,
#     #         self.cnn_classifier,
#     #     )
#     #     def __init__(self, input_size=1, hidden_size=256, num_layers=2, bidirectional=True, num_classes=12):
#     #         super(APP_Net, self).__init__()
#     #         # rnn配置
#     #         self.bidirectional = bidirectional
#     #         self.hidden_size = hidden_size
#     #         self.num_layers = num_layers
#     #         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
#     #                             batch_first=True)
#     #         self.fc0 = nn.Linear(hidden_size, num_classes)
#     #         self.fc1 = nn.Linear(hidden_size * 2, num_classes)
#     #
#     #         self.cnn_feature = nn.Sequential(
#     #             # 卷积层1
#     #             nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),  # (1,1024)->(32,1024)
#     #             nn.BatchNorm1d(32),  # 加上BN的结果
#     #             nn.ReLU(),
#     #             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (32,1024)->(32,342)
#     #
#     #             # 卷积层2
#     #             nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),  # (32,342)->(64,342)
#     #             nn.BatchNorm1d(64),
#     #             nn.ReLU(),
#     #             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,342)->(64,114)
#     #         )
#     #         # 全连接层
#     #         self.cnn_classifier = nn.Sequential(
#     #             # 64*114
#     #             nn.Flatten(),
#     #             nn.Linear(in_features=64 * 114, out_features=1024),  # 784:88*64, 1024:114*64, 4096:456*64
#     #         )
#     #
#     #         self.cnn = nn.Sequential(
#     #             self.cnn_feature,
#     #             self.cnn_classifier,
#     #         )
#         self.rnn = nn.Sequential(
#             # (batch_size, seq_len, input_size)
#             nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True),
#         )
#         self.classifier_bi = nn.Sequential(
#             nn.Linear(in_features=1024 + hidden_size * 2, out_features=1024),
#             nn.Dropout(p=0.7),
#             nn.Linear(in_features=1024, out_features=num_classes)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=1024 + hidden_size, out_features=1024),
#             nn.Dropout(p=0.7),
#             nn.Linear(in_features=1024, out_features=num_classes)
#         )
#
#     def forward(self, x_payload, x_sequence,x_sta):
#         x_payload, x_sequence, x_sta = self.data_trans(x_payload, x_sequence, x_sta)
#         x_payload = self.cnn(x_payload)
#         x_sequence = self.rnn(x_sequence)
#         x_sequence = x_sequence[0][:, -1, :]
#         x = torch.cat((x_payload, x_sequence), 1)
#         if self.bidirectional == True:
#             x = self.classifier_bi(x)
#         else:
#             x = self.classifier(x)
#         return x,None
#
#     def data_trans(self,x_payload, x_sequence,x_sta):
#         # featuresCalc = FeaturesCalc(min_window_size=1)  # 初始化
#         # x_sta= featuresCalc.compute_features(packets_list=) # 计算特征
#         return x_payload, x_sequence,x_sta



class Enhanced_APP_Net(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_layers=3, bidirectional=True, num_classes=12):
        super(Enhanced_APP_Net, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM部分
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        # 添加线性层，将LSTM输出变为256维
        if bidirectional:
            self.lstm_to_256 = nn.Linear(hidden_size * 2, 256)
        else:
            self.lstm_to_256 = nn.Linear(hidden_size, 256)

        # CNN部分
        self.cnn_feature = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=25, stride=1, padding=12),  # (1,1024)->(64,1024)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,1024)->(64,342)

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, stride=1, padding=7),  # (64,342)->(128,342)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (128,342)->(128,114)

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3),  # (128,114)->(256,114)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (256,114)->(256,38)
        )

        self.cnn_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 38, out_features=2048),  # 根据卷积核和池化层计算
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.cnn = nn.Sequential(
            self.cnn_feature,
            self.cnn_classifier,
        )

        # 分类器部分
        # 计算输入特征的总维度
        input_features = 1024 + 256 + 26  # CNN 输出 1024 + LSTM 输出 256 + x_sta 的 26 个特征

        self.classifier_bi = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x_payload, x_sequence, x_sta):
        #print(f"x_sta shape: {x_sta.shape}")
        x_payload, x_sequence, x_sta = self.data_trans(x_payload, x_sequence, x_sta)

        # CNN模块处理payload数据
        x_payload = self.cnn(x_payload)
        #print(f"x_payload shape: {x_payload.shape}")  # 检查输出形状

        # LSTM模块处理sequence数据
        x_sequence, _ = self.lstm(x_sequence)
        x_sequence = x_sequence[:, -1, :]  # 只取最后时间步的输出
        # 新增：通过线性层将LSTM的输出映射到256维
        x_sequence = self.lstm_to_256(x_sequence)
        #print(f"x_squ shape: {x_sequence.shape}")  # 检查输出形状



        # 拼接CNN, LSTM, 静态特征的输出
        x = torch.cat((x_payload, x_sequence, x_sta), dim=1)

        # 分类器部分
        if self.bidirectional:
            x = self.classifier_bi(x)
        else:
            x = self.classifier(x)

        return x, None

    def data_trans(self, x_payload, x_sequence, x_sta):
        # 这里可以对x_payload, x_sequence和x_sta做一些数据预处理
        # featuresCalc = FeaturesCalc(min_window_size=1)  # 初始化
        # x_sta= featuresCalc.compute_features(packets_list=["Avg_syn_flag", "Avg_urg_flag", "Avg_fin_flag", "Avg_ack_flag", "Avg_psh_flag", "Avg_rst_flag", "Avg_DNS_pkt", "Avg_TCP_pkt",
        # "Avg_UDP_pkt", "Avg_ICMP_pkt", "Duration_window_flow", "Avg_delta_time", "Min_delta_time", "Max_delta_time", "StDev_delta_time",
        # "Avg_pkts_lenght", "Min_pkts_lenght", "Max_pkts_lenght", "StDev_pkts_lenght", "Avg_small_payload_pkt", "Avg_payload", "Min_payload",
        # "Max_payload", "StDev_payload", "Avg_DNS_over_TCP", "Num_pkts"]) # 计算特征
        return x_payload, x_sequence, x_sta





def app_net(model_path, pretrained=False, **kwargs):
    """
    CNN 1D model architecture

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Enhanced_APP_Net(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        #print(model)
    return model

# class Enhanced_APP_Ne(nn.Module):
#     def __init__(self, input_size=1, hidden_size=512, num_layers=3, bidirectional=True, num_classes=12):
#         super(Enhanced_APP_Ne, self).__init__()
#         self.bidirectional = bidirectional
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # LSTM部分
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
#                             bidirectional=bidirectional, batch_first=True)
#         # 添加线性层，将LSTM输出变为256维
#         if bidirectional:
#             self.lstm_to_256 = nn.Linear(hidden_size * 2, 256)
#         else:
#             self.lstm_to_256 = nn.Linear(hidden_size, 256)
#
#         # CNN部分
#         self.cnn_feature = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=64, kernel_size=25, stride=1, padding=12),  # (1,1024)->(64,1024)
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,1024)->(64,342)
#
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, stride=1, padding=7),  # (64,342)->(128,342)
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (128,342)->(128,114)
#
#             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3),  # (128,114)->(256,114)
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (256,114)->(256,38)
#         )
#
#         self.cnn_classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=256 * 38, out_features=2048),  # 根据卷积核和池化层计算
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#         )
#
#         self.cnn = nn.Sequential(
#             self.cnn_feature,
#             self.cnn_classifier,
#         )
#
#         # 处理x_sta静态特征
#         self.fc_sta = nn.Sequential(
#             nn.Linear(in_features=26, out_features=512),  # 26个输入特征
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#         )
#
#         # 分类器部分
#         self.classifier_bi = nn.Sequential(
#             nn.Linear(in_features=1024 + 256 + 26, out_features=1024),  # 输入特征
#             nn.ReLU(),  # 激活函数
#             nn.Dropout(p=0.5),  # 防止过拟合
#             nn.Linear(in_features=1024, out_features=512),  # 新增中间层，将维度降到 512
#             nn.ReLU(),  # 激活函数
#             nn.Dropout(p=0.5),  # 再次应用 Dropout
#             nn.Linear(in_features=512, out_features=num_classes)  # 输出层
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=1024 + 256 + 26, out_features=1024),  # 输入特征
#             nn.ReLU(),  # 激活函数
#             nn.Dropout(p=0.5),  # 防止过拟合
#             nn.Linear(in_features=1024, out_features=512),  # 新增中间层，将维度降到 512
#             nn.ReLU(),  # 激活函数
#             nn.Dropout(p=0.5),  # 再次应用 Dropout
#             nn.Linear(in_features=512, out_features=num_classes)  # 输出层
#         )
#
#     def forward(self, x_payload, x_sequence, x_sta):
#         x_payload, x_sequence, x_sta = self.data_trans(x_payload, x_sequence, x_sta)
#
#         # CNN模块处理payload数据
#         x_payload = self.cnn(x_payload)
#
#         # LSTM模块处理sequence数据
#         x_sequence, _ = self.lstm(x_sequence)
#         x_sequence = x_sequence[:, -1, :]  # 只取最后时间步的输出
#         # 新增：通过线性层将LSTM的输出映射到256维
#         x_sequence = self.lstm_to_256(x_sequence)
#
#         # 处理静态特征x_sta
#         #x_sta = self.fc_sta(x_sta)
#         #print(f"x_sta shape after fc_sta: {x_sta.shape}")
#         # 拼接CNN, LSTM, 静态特征的输出
#         x = torch.cat((x_payload, x_sequence, x_sta), dim=1)
#
#         # 分类器部分
#         if self.bidirectional:
#             x = self.classifier_bi(x)
#         else:
#             x = self.classifier(x)
#
#         return x,None
#
#     def data_trans(self, x_payload, x_sequence, x_sta):
#         # 这里可以对x_payload, x_sequence和x_sta做一些数据预处理
#         return x_payload, x_sequence, x_sta