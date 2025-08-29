

from utils.helper import AverageMeter, accuracy
from log.set_log import logger
import torch


def train_process(train_loader, model, alpha, criterion_c, criterion_r, optimizer, epoch, device, print_freq):
    """训练一个 epoch 的流程

    Args:
        train_loader (dataloader): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        epoch (int): 当前所在的 epoch
        device (torch.device): 是否使用 gpu
        print_freq ([type]): [description]
    """

    c_losses = AverageMeter()  # 在一个 train loader 中的 loss 变化
    r_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()  # 记录在一个 train loader 中的 accuracy 变化

    model.train()  # 切换为训练模型

    for i, (pay, seq, statistic, target) in enumerate(train_loader):
        # pay = pay.reshape(-1,256,1)
        #GPU
        pay = pay.to(device)
        seq = seq.to(device)
        statistic = statistic.to(device)
        target = target.to(device)

        # print(f"Input statistic shape: {statistic.shape}")
        # print(f"Input seq shape: {seq.shape}")
        # print(f"Input pay shape: {pay.shape}")

        output = model(pay, seq, statistic)  # 得到模型预测结果
        #print(pay, seq, statistic)
        classify_result, fake_rebuild = output

        loss_c = criterion_c(classify_result, target)  # 计算 分类的 loss
        if fake_rebuild != None:
            loss_r = criterion_r(statistic, fake_rebuild)  # 计算 重构 loss
            r_losses.update(loss_r.item(), pay.size(0))
        else:
            loss_r = 0
            alpha = 1
        loss = alpha * loss_c + loss_r  # 将两个误差组合在一起

        prec1 = accuracy(classify_result.data, target)
        c_losses.update(loss_c.item(), pay.size(0))
        losses.update(loss.item(), pay.size(0))
        top1.update(prec1[0].item(), pay.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1))
    return losses.val, top1.val


def train_process1(train_loader, model, alpha, criterion_c, criterion_r, optimizer, epoch, device, print_freq):
    """训练一个 epoch 的流程

    Args:
        train_loader (dataloader): 数据加载器
        model ([type]): 模型
        criterion_c ([type]): 分类损失函数
        criterion_r ([type]): 重建损失函数
        optimizer ([type]): 优化器
        epoch (int): 当前 epoch
        device (torch.device): 当前设备，cpu 或 gpu
        print_freq (int): 打印频率
    """

    c_losses = AverageMeter()  # 分类损失
    r_losses = AverageMeter()  # 重建损失
    losses = AverageMeter()  # 总损失
    top1 = AverageMeter()  # 精度

    model.train()  # 切换为训练模式

    for i, (pay, seq, statistic, target) in enumerate(train_loader):
        # 数据无需转移到GPU，保持在CPU上
        # 如果你需要进行任何类型的张量操作，保持它们在CPU上（不需要 `.to(device)`）

        # 在 CPU 上，移除 `.to(device)`
        # 这里的 pay, seq, statistic, target 不需要被转移到 GPU，因为已在 CPU 上
        output = model(pay.to(device), seq.to(device), statistic.to(device))  # 得到模型预测结果
        classify_result, fake_rebuild = output

        # 计算分类损失
        loss_c = criterion_c(classify_result, target)

        # 计算重建损失
        if fake_rebuild is not None:
            loss_r = criterion_r(statistic, fake_rebuild)
            r_losses.update(loss_r.item(), pay.size(0))  # 更新重建损失
        else:
            loss_r = 0
            alpha = 1  # 如果没有重建数据，设置 alpha = 1

        loss = alpha * loss_c + loss_r  # 总损失

        # 计算准确率
        prec1 = accuracy(classify_result.data, target)
        c_losses.update(loss_c.item(), pay.size(0))  # 更新分类损失
        losses.update(loss.item(), pay.size(0))  # 更新总损失
        top1.update(prec1[0].item(), pay.size(0))  # 更新准确率

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            # 打印每个训练步的信息
            logger.info(
                'Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1))

    # 返回每个 epoch 的损失和准确率
    return losses.val, top1.val
