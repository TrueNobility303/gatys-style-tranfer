import torch as t 
import torchvision as tv 

#定义均值和方差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#得到gram相似度矩阵
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    #对特征及其转置做batch上的矩阵乘法
    gram = t.bmm(features, features_t) / (ch * h * w)
    #计算channel之间的相似度
    return gram

#输入特征提取网络之前先做batch归一化
def batch_normalize(batch):
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    return (batch - mean) / std




