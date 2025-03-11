# 使用说明
针对不平衡少标签数据的参照骨等级识别
## 目录说明
clm.py：针对有序分类任务的输出层  
WeightedKappaLoss：加权Kappa损失函数，适配有序分类任务  
train_WeightKappaLoss_clm：训练代码，损失函数采用WeightedKappaLoss与交叉熵损失的加权和
