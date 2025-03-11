import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet50, densenet121, efficientnet_v2_s, efficientnet_v2_m

from clm import CLM
from WeightedKappaLoss import WeightedKappaLoss
from my_dataset import MyDataSet
from utils import read_split_data
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import logging
from datetime import datetime

from sklearn.metrics import cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import seaborn as sns
from collections import Counter

# 修改每个基学习器的输出层为CLM结构
def create_clm_model(base_model, num_classes):
    # EfficientNetV2和DenseNet处理
    if hasattr(base_model, 'classifier'):
        if isinstance(base_model.classifier, nn.Sequential):
            # 保留原始结构，仅替换最后一层
            in_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Linear(in_features, 1)
        else:
            # 直接处理单个Linear层
            in_features = base_model.classifier.in_features
            base_model.classifier = nn.Linear(in_features, 1)

    # ResNet处理
    elif hasattr(base_model, 'fc'):
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, 1)

        # **适配 Swin Transformer (torchvision/timm 版本)**
    elif hasattr(base_model, 'head'):
        in_features = base_model.head.in_features
        base_model.head = nn.Linear(in_features, 1)

    # **适配 Vision Transformer (ViT)**
    elif hasattr(base_model, 'heads'):
        in_features = base_model.heads.head.in_features
        base_model.heads.head = nn.Linear(in_features, 1)

    # 组合CLM层
    return nn.Sequential(
        base_model,
        CLM(num_classes, "probit")
    )


# 梯度测试代码（添加到训练脚本中）
def check_grad_flow(model):
    grad_norms = [
        (name, param.grad.norm().item())
        for name, param in model.named_parameters()
        if param.grad is not None
    ]
    for name, norm in grad_norms:
        print(f"参数层 {name}: 梯度范数={norm:.4e}")
    return any(norm > 1e-6 for _, norm in grad_norms)


def main(args):
    # 初始化日志系统
    # log_file = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(message)s',
    #     handlers=[
    #         logging.FileHandler(log_file),
    #         logging.StreamHandler()
    #     ]
    # )
    # logger = logging.getLogger()
    # logger.info("训练参数: %s", vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label, class_indices = read_split_data(
        args.data_path)

    print(Counter(train_images_label)) 

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop([224, 224]),
            # transforms.RandomResizedCrop(img_size[num_model][0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            # transforms.Resize(img_size[num_model][1]),
            transforms.RandomResizedCrop([224, 224]),
            # transforms.CenterCrop(img_size[num_model][1]),
            transforms.CenterCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 获取加权采样器
    train_sampler = MyDataSet.get_weighted_sampler(train_images_label)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 创建三个基学习器
    # 修改 EfficientNetV2 的第一层卷积
    # model1 = efficientnet_v2_m(pretrained=False)
    # model1.features[0][0] = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)  # 修改输入通道为1
    # model1.classifier = nn.Linear(model1.classifier[1].in_features, args.num_classes)
    # model1 = model1.to(device)
    model1 = models.vit_b_16(pretrained=False)

    """
    修改为使用CLM激活层
    """
    # model1.fc = nn.Linear(model1.classifier[1].in_features, 1)  # EfficientNetV2
    model1 = create_clm_model(model1, args.num_classes).to(device)

    # 修改 ResNet50 的第一层卷积
    # model2 = resnet50(pretrained=False)
    # model2.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model2.fc = nn.Linear(model2.fc.in_features, args.num_classes)
    # model2 = model2.to(device)
    model2 = models.swin_b(pretrained=False)

    """
    修改为使用CLM激活层
    """
    # model2.fc = nn.Linear(model2.fc.in_features, 1)  # ResNet
    model2 = create_clm_model(model2, args.num_classes).to(device)

    # 修改 DenseNet121 的第一层卷积
    model3 = densenet121(pretrained=False)
    model3.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model3.classifier = nn.Linear(model3.classifier.in_features, args.num_classes)
    # model3 = model3.to(device)

    """
    修改为使用CLM激活层
    """
    # model3.classifier = nn.Linear(model3.classifier.in_features, 1)  # DenseNet
    model3 = create_clm_model(model3, args.num_classes).to(device)

    # 定义集成权重（可学习参数）
    ensemble_weights = nn.Parameter(torch.ones(3) / 3)  # 初始平均权重

    # 损失函数
    """
    修改为QWK损失函数
    """
    criterion_kappa = WeightedKappaLoss(args.num_classes, regression=False).to(device)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)  # 交叉熵损失
    lambda_ce = 0.7  # 控制交叉熵损失的影响

    # criterion = WeightedKappaLoss(args.num_classes, regression=False).to(device)

    # criterion = nn.CrossEntropyLoss()

    def train_one_epoch(models, optimizers, ensemble_optimizer, data_loader, device, epoch):
        for model in models:
            model.train()

        total_loss = 0.0
        total_mae = 0.0
        total_acc = 0.0
        total_samples = 0  
        batch_count = 0

        all_preds = []
        all_labels = []

        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 基学习器前向传播（保留梯度）
            base_probs = []
            for model in models:
                output = model(images)
                probs = F.softmax(output, dim=1)
                base_probs.append(probs)

            # 动态加权集成
            weights = F.softmax(ensemble_weights / 0.5, dim=0)
            ensemble_probs = sum(w * p for w, p in zip(weights, base_probs))

            # 计算损失
            qwk_loss = criterion_kappa(ensemble_probs, labels)
            ce_loss = criterion_ce(ensemble_probs, labels)
            total_loss_batch = (1-lambda_ce) * qwk_loss + lambda_ce * ce_loss  # **新损失函数**
            
            # qwk_loss = criterion(ensemble_probs, labels)
            # qwk_score = 1 - qwk_loss.item()  # 计算 QWK 评分

            # 反向传播
            for opt in optimizers:
                opt.zero_grad()
            ensemble_optimizer.zero_grad()

            total_loss_batch.backward()
            # qwk_loss.backward()

            # 检查第一个模型的梯度
            # if check_grad_flow(models[0]):
            #     print("梯度流动正常")
            # else:
            #     print("警告: 检测到梯度消失!")

            # 梯度统计（每10个batch打印一次）
            # if batch_idx % 10 == 0:
            #     grad_norms = [
            #         p.grad.norm().item()
            #         for model in models
            #         for p in model.parameters()
            #         if p.grad is not None
            #     ]
            #     avg_grad = sum(grad_norms)/len(grad_norms) if grad_norms else 0
            #     max_grad = max(grad_norms) if grad_norms else 0

            #     logger.info(
            #         "[Epoch %02d Batch %04d] 梯度统计: 平均=%.2e 最大=%.2e",
            #         epoch, batch_idx, avg_grad, max_grad
            #     )

            for opt in optimizers:
                opt.step()
            ensemble_optimizer.step()

            # 计算评估指标
            with torch.no_grad():
                pred_labels = torch.argmax(ensemble_probs, dim=1)
                mae = torch.abs(pred_labels.float() - labels.float()).mean()
                acc = (pred_labels == labels).float().mean()  # 计算准确率

            total_loss += total_loss_batch.item()
            # total_loss += qwk_loss.item()
            # total_qwk += qwk_score
            total_mae += mae.item()
            total_acc += acc.item() * labels.size(0)  # 累加准确率
            total_samples += labels.size(0)  # 累加样本数

            # 收集预测结果
            all_preds.append(pred_labels.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            batch_count += 1

        # 计算 sklearn QWK
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

        print("Current ensemble weights:", F.softmax(ensemble_weights, dim=0).detach().cpu().numpy())
        print(f"Min prob: {ensemble_probs.min().item()}, Max prob: {ensemble_probs.max().item()}")

        clm_layer = models[0][-1]  # 取 Sequential 里的最后一层（即 CLM）
        print(f"Epoch {epoch + 1} - CLM Output Range: min={clm_layer.last_min_prob:.6f}, max={clm_layer.last_max_prob:.6f}")

        print(f"Epoch {epoch + 1} - Train Loss: {total_loss / batch_count:.4f}")



        return (train_qwk,
                total_loss / batch_count,
                total_mae / batch_count,
                total_acc / total_samples)  # 返回平均准确率

    
    

    # 修改验证函数
    def evaluate(models, data_loader, device):
        
        global best_qwk, best_conf_matrix  # 让这两个变量可以跨函数修改
        best_qwk = -1  # 记录最高的 QWK
        best_conf_matrix = None  # 存储最佳混淆矩阵
        
        for model in models:
            model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_acc = 0.0
        total_samples = 0
        batch_count = 0
    
        all_preds = []
        all_labels = []


        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                # 获取每个基学习器的预测
                base_probs = []
                for model in models:
                    outputs = model(images)
                    base_probs.append(F.softmax(outputs, dim=1))

                # 动态加权集成
                weights = F.softmax(ensemble_weights, dim=0)
                ensemble_probs = sum(w * p for w, p in zip(weights, base_probs))

                ensemble_probs = torch.softmax(ensemble_probs / 0.1, dim=1)  # 让预测更自信


                # 计算 QWK 损失
                # batch_qwk = criterion(ensemble_probs, labels)
                # total_loss += batch_qwk.item()

                # **计算 QWK + 交叉熵损失**
                batch_qwk = criterion_kappa(ensemble_probs, labels)
                batch_ce = criterion_ce(ensemble_probs, labels)
                batch_loss = batch_qwk + lambda_ce * batch_ce  # **新损失函数**

                total_loss += batch_loss.item()
    
                # 计算准确率
                pred_labels = torch.argmax(ensemble_probs, dim=1)
                acc = (pred_labels == labels).float().mean()
                total_acc += acc.item() * labels.size(0)
                total_samples += labels.size(0)
    
                # 收集预测结果
                all_preds.append(pred_labels.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
                batch_count += 1

                # # 计算本批次的QWK损失
                # batch_qwk = criterion(ensemble_probs, labels)   # 计算 WeightedKappaLoss
                # total_loss += batch_qwk.item()

                # # batch_qwk_score = 1 - batch_qwk.item()  # 计算 QWK 评分
                # total_qwk += batch_qwk_score  # 修正为累加 QWK 评分
                # # total_qwk += batch_qwk.item()
                # batch_count += 1

                # # 计算准确率
                # pred_labels = torch.argmax(ensemble_probs, dim=1)
                # acc = (pred_labels == labels).float().mean()
                # total_acc += acc.item() * labels.size(0)
                # total_samples += labels.size(0)

                # # 收集预测结果
                # all_preds.append(pred_labels.cpu())
                # all_labels.append(labels.cpu())

        # 整体指标计算
        # preds = torch.cat(all_preds)
        # labels = torch.cat(all_labels)

        # mae = torch.mean(torch.abs(preds.float() - labels.float())).item()
        # avg_qwk = total_qwk / batch_count
        # avg_acc = total_acc / total_samples  # 计算平均准确率

        # sklearn_qwk = cohen_kappa_score(labels.cpu().numpy(), preds.cpu().numpy(), weights="quadratic")
        # print(f"Val QWK (sklearn): {sklearn_qwk:.4f} | Val QWK (model): {avg_qwk:.4f}")

        # return mae, avg_qwk, avg_acc


        # 计算 sklearn QWK
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        sklearn_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    
        mae = np.mean(np.abs(all_preds - all_labels))
        avg_acc = total_acc / total_samples

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_preds)

    # 记录最佳 QWK 和对应的混淆矩阵
        if sklearn_qwk > best_qwk:
            best_qwk = sklearn_qwk
            best_conf_matrix = conf_matrix  # 记录最佳混淆矩阵
    
        # print(f"Val QWK (sklearn): {sklearn_qwk:.4f}")
    
        return mae, sklearn_qwk, avg_acc

    # optimizer1 = optim.AdamW(
    #     model1.parameters(),
    #     lr=args.lr,  # 提高学习率
    #     weight_decay=args.weight_decay,
    #     betas=(0.9, 0.999)
    # )
    #
    # optimizer2 = optim.AdamW(
    #     model2.parameters(),
    #     lr=args.lr,  # 提高学习率
    #     weight_decay=args.weight_decay,
    #     betas=(0.9, 0.999)
    # )
    #
    # optimizer3 = optim.AdamW(
    #     model3.parameters(),
    #     lr=args.lr,  # 提高学习率
    #     weight_decay=args.weight_decay,
    #     betas=(0.9, 0.999)
    # )

    # 使用Adam优化器
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer3 = optim.Adam(model3.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 初始化优化器（加入集成权重优化）
    ensemble_optimizer = optim.Adam([ensemble_weights], lr=1e-5)

    # 学习率调度器
    scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler3 = lr_scheduler.CosineAnnealingLR(optimizer3, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 训练循环
    best_qwk = 0.0
    best_mae = 0.0
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_qwk, train_loss, train_mae, train_acc = train_one_epoch(
            [model1, model2, model3],
            [optimizer1, optimizer2, optimizer3],
            ensemble_optimizer,
            train_loader,
            device,
            epoch
        )

        val_mae, val_qwk, val_acc = evaluate(
            [model1, model2, model3],
            val_loader,
            device
        )

        # if val_mae > best_mae:
        #     best_mae = val_mae

        if val_acc > best_acc:
            best_acc = val_acc

        # 保存最佳模型
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                'models': [m.state_dict() for m in [model1, model2, model3]],
                'ensemble_weights': ensemble_weights,
            }, 'best_ensemble.pth')

        print(f'Epoch {epoch + 1}:')
        print(f'Train QWK: {train_qwk:.4f} | Train Loss: {train_loss:.4f}| Train MAE: {train_mae:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val QWK: {val_qwk:.4f} | Val MAE: {val_mae:.4f} | Val Acc: {val_acc:.4f}')

    print(f'BEST Val Acc: {val_acc:.4f}')

    print("Ensemble Weights:", F.softmax(ensemble_weights, dim=0).detach().cpu().numpy())


    # 训练结束后绘制最佳混淆矩阵
    if best_conf_matrix is not None:
        class_labels = [str(i) for i in range(args.num_classes)]  # 假设类别是 0, 1, 2, ..., num_classes-1
        plot_confusion_matrix(best_conf_matrix, class_labels, save_path="confusion_matrix.png")


    # 记录到TensorBoard
    # tb_writer.add_scalar("Train Loss", (train_loss1 + train_loss2 + train_loss3) / 3, epoch)
    # tb_writer.add_scalar("Train Accuracy", (train_acc1 + train_acc2 + train_acc3) / 3, epoch)
    # tb_writer.add_scalar("Val Accuracy", val_accuracy, epoch)
    # tb_writer.add_scalar("Learning Rate", optimizer1.param_groups[0]["lr"], epoch)


def plot_confusion_matrix(cm, class_labels, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Best Confusion Matrix")
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {os.path.abspath(save_path)}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data-path', type=str, default="gou-1w+")
    # parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)