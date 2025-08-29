
"""
@Description: 训练与测试模型
"""
import warnings
warnings.filterwarnings("ignore")
import sys

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn, optim
from log.set_log import logger
from utils.set_config import setup_config
# from models.cnn1d import cnn1d as train_model
from models.app_net import app_net as train_model
from train_valid.train import train_process
from train_valid.valid import valid_process
from dataloader.data_loader import data_loader
from dataloader.get_tensor import get_tensor_data

from utils.helper import adjust_learning_rate, save_checkpoint

from utils.evaluate_tools import display_model_performance_metrics

from torch.utils.tensorboard import SummaryWriter
import os
# logger = init_logger(log_path='/home/xl/TrafficClassificationPandemonium/log/log_file/train.log')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_pipeline():
    yaml_path = r"E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/configuration/traffic_classification_configuration.yaml"
    cfg = setup_config(yaml_path) # 获取 config 文件
    logger.info(cfg)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('是否使用 GPU 进行训练, {}'.format(device))
    torch.cuda.empty_cache()
    os.makedirs(cfg.train.model_dir, exist_ok=True)  # 这行代码创建了一个存储模型的目录，如果目录已存在则不会报错。
    model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name)  # 模型的路径 目录拼接名称 是否存在
    print(model_path)
    num_classes = len(cfg.test.label2index)  # 类别的数量
    print(num_classes)
    model = train_model(model_path, pretrained=cfg.test.pretrained, num_classes=num_classes).to(device)  # 定义模型)
    criterion_c = nn.CrossEntropyLoss()  # 分类用的损失函数
    criterion_r = nn.L1Loss()  # 重构误差的损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)  # 定义优化器


    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pay, seq_file=cfg.train.train_seq,
                               statistic_file=cfg.train.train_sta,
                               label_file=cfg.train.train_label,
                               batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    # valid_loader = data_loader(pcap_file=cfg.train.valid_pay, seq_file=cfg.train.valid_seq,
    #                           statistic_file=cfg.train.valid_sta,
    #                           label_file=cfg.train.valid_label,
    #                           batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    test_loader = data_loader(pcap_file=cfg.train.test_pay, seq_file=cfg.train.test_seq,
                              statistic_file=cfg.train.test_sta,
                              label_file=cfg.train.test_label,
                              batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    logger.info('成功加载数据集.')
    # 若只进行验证，验证后直接退出
    if cfg.test.evaluate:  # 是否只进行测试
        logger.info('进入测试模式.')
        prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r,0, device,
                                                 1)  # evaluate on validation set
        torch.cuda.empty_cache()  # 清除显存
        # 计算每个类别详细的准确率
        index2label = {j: i for i, j in cfg.test.label2index.items()}  # index->label 对应关系
        print(index2label)
        label_list = [index2label.get(i) for i in range(len(index2label))]  # 17 个 label 的标签
        pcap_data, seq_data, statistic_data, label_data = get_tensor_data(pcap_file=cfg.train.test_pay,
                                                                          seq_file=cfg.train.test_seq,
                                                                          statistic_file=cfg.train.test_sta,
                                                                          label_file=cfg.train.test_label)
        start_index = 0
        y_pred = None
        int_test_nums = len(test_loader) * (cfg.train.BATCH_SIZE - 1)
        int_test_nums = (int)(int_test_nums / 100) * 100  # 计算验证集样本的数量

        for i in list(range(100, int_test_nums + 100, 100)):  # 每次处理100个样本
            pay = pcap_data[start_index:i]
            seq = seq_data[start_index:i]
            sta = statistic_data[start_index:i]

            y_pred_batch, _ = model(pay.to(device), seq.to(device), sta.to(device))  # 在每次迭代中，从测试数据中获取一个批次的网络数据、序列数据和统计数据，然后将它们传递给模型进行预测。

            start_index = i

            # 将返回的预测结果 存到y_pred中去
            if y_pred == None:
                y_pred = y_pred_batch.cpu().detach()
            else:
                y_pred = torch.cat((y_pred, y_pred_batch.cpu().detach()), dim=0)
                print(y_pred.shape)

        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)  # 提取出y_pred中的最大值及其索引

        Y_data_label = [index2label.get(i.tolist()) for i in label_data]  # 将索引转换为具体的类别名称

        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]  # 将模型预测的索引 pred 转换为具体的类别名称。

        Y_data_label = Y_data_label[:int_test_nums]
        display_model_performance_metrics(true_labels=Y_data_label, predicted_labels=pred_label,confusion_path = cfg.test.confusion_path, classes=label_list)
        return

    best_prec1 = -float('inf')
    val_loss = float('inf')
    loss_writer = SummaryWriter(log_dir=os.path.join("E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/result/tensorboard", "loss"))
    acc_writer = SummaryWriter(log_dir=os.path.join("E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/result/tensorboard", "acc"))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)  # 或者使用StepLR调度器

    for epoch in range(cfg.train.epochs):
        torch.cuda.empty_cache()
        scheduler = adjust_learning_rate(optimizer, scheduler, epoch, best_prec1, val_loss)  # 动态调整学习率

        train_loss, train_acc = train_process(train_loader, model, 1, criterion_c, criterion_r, optimizer, epoch,
                                              device,
                                              2)  # train for one epoch
        prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r, epoch, device,
                                                 1)  # evaluate on validation set

        loss_writer.add_scalars("loss", {'train': train_loss, 'val': val_loss}, epoch)
        acc_writer.add_scalars("train_acc", {'train': train_acc, 'val': val_acc}, epoch)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # 保存最优的模型
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

    loss_writer.close()
    acc_writer.close()
    logger.info('Finished! (*￣︶￣)')


    # for epoch in range(cfg.train.epochs):
    #     adjust_learning_rate(optimizer, epoch, cfg.train.lr)  # 动态调整学习率
    #
    #     train_loss, train_acc = train_process(train_loader, model, 1, criterion_c, criterion_r, optimizer, epoch,
    #                                           device,
    #                                           2)  # train for one epoch
    #     prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r,epoch, device,
    #                                              1)  # evaluate on validation set
    #
    #     loss_writer.add_scalars("loss", {'train': train_loss, 'val': val_loss}, epoch)
    #     acc_writer.add_scalars("train_acc", {'train': train_acc, 'val': val_acc}, epoch)
    #
    #     # remember the best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #
    #     # 保存最优的模型
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': best_prec1,
    #         'optimizer': optimizer.state_dict()
    #     }, is_best, model_path)

    loss_writer.close()
    acc_writer.close()
    logger.info('Finished! (*￣︶￣)')


def train_pipeline1():
    yaml_path = r"E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/configuration/traffic_classification_configuration.yaml"
    cfg = setup_config(yaml_path)  # 获取 config 文件
    logger.info(cfg)

    # 强制使用CPU
    device = torch.device("cpu")
    logger.info('使用 CPU 进行训练')

    os.makedirs(cfg.train.model_dir, exist_ok=True)  # 创建模型存储目录
    model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name)
    print(model_path)

    num_classes = len(cfg.test.label2index)  # 类别数量
    model = train_model(model_path, pretrained=cfg.test.pretrained, num_classes=num_classes).to(device)  # 初始化模型
    criterion_c = nn.CrossEntropyLoss()  # 分类损失函数
    criterion_r = nn.L1Loss()  # 重构误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)  # 优化器

    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pay, seq_file=cfg.train.train_seq,
                               statistic_file=cfg.train.train_sta,
                               label_file=cfg.train.train_label,
                               batch_size=cfg.train.BATCH_SIZE)  # 获取训练数据
    test_loader = data_loader(pcap_file=cfg.train.test_pay, seq_file=cfg.train.test_seq,
                              statistic_file=cfg.train.test_sta,
                              label_file=cfg.train.test_label,
                              batch_size=cfg.train.BATCH_SIZE)  # 获取测试数据
    logger.info('成功加载数据集.')

    # 如果只进行测试
    if cfg.test.evaluate:
        logger.info('进入测试模式.')
        prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r, 0, device, 1)  # 评估
        # 清空显存
        #torch.cuda.empty_cache()

        index2label = {j: i for i, j in cfg.test.label2index.items()}
        print(index2label)
        label_list = [index2label.get(i) for i in range(len(index2label))]
        pcap_data, seq_data, statistic_data, label_data = get_tensor_data(pcap_file=cfg.train.test_pay,
                                                                          seq_file=cfg.train.test_seq,
                                                                          statistic_file=cfg.train.test_sta,
                                                                          label_file=cfg.train.test_label)

        start_index = 0
        y_pred = None
        int_test_nums = len(test_loader) * (cfg.train.BATCH_SIZE - 1)
        int_test_nums = (int)(int_test_nums / 100) * 100

        for i in list(range(100, int_test_nums + 100, 100)):
            pay = pcap_data[start_index:i]
            seq = seq_data[start_index:i]
            sta = statistic_data[start_index:i]

            y_pred_batch, _ = model(pay, seq, sta)  # 不需要.to(device)

            start_index = i

            # 将预测结果合并到y_pred中
            if y_pred is None:
                y_pred = y_pred_batch.cpu().detach()
            else:
                y_pred = torch.cat((y_pred, y_pred_batch.cpu().detach()), dim=0)

        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)

        Y_data_label = [index2label.get(i.tolist()) for i in label_data]
        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]

        Y_data_label = Y_data_label[:int_test_nums]
        display_model_performance_metrics(true_labels=Y_data_label, predicted_labels=pred_label,
                                          confusion_path=cfg.test.confusion_path, classes=label_list)
        return

    # 如果进行训练
    best_prec1 = -float('inf')
    val_loss = float('inf')
    loss_writer = SummaryWriter(log_dir=os.path.join(
        "E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/result/tensorboard",
        "loss"))
    acc_writer = SummaryWriter(log_dir=os.path.join(
        "E:/pythonProject/TrafficClassificationPandemonium-neww/TrafficClassificationPandemonium-main/result/tensorboard",
        "acc"))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    for epoch in range(cfg.train.epochs):
        scheduler = adjust_learning_rate(optimizer, scheduler, epoch, best_prec1, val_loss)  # 调整学习率

        train_loss, train_acc = train_process(train_loader, model, 1, criterion_c, criterion_r, optimizer, epoch,
                                              device, 2)
        prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r, epoch, device, 1)

        loss_writer.add_scalars("loss", {'train': train_loss, 'val': val_loss}, epoch)
        acc_writer.add_scalars("train_acc", {'train': train_acc, 'val': val_acc}, epoch)

        # 保存最优模型
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

    loss_writer.close()
    acc_writer.close()
    logger.info('Finished! (*￣︶￣)')


if __name__ == "__main__":
    print('111')
    train_pipeline()  # 用于测试
