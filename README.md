# Neural Style Tranfer 

Neural style tranfer of Gatys methods in Paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)

Please refer to my report for details!

## Demo

![demo](https://github.com/TrueNobility303/vgg-image-style-tranfer/blob/master/results/arc.png)

# 神经网络风格迁移

## 摘要
用神经网络进行风格迁移，采用预训练的vgg提取图片特征，采取下采样（编码）结合上采样（解码）的方式转换图片，尽可能最小化生成图片的内容损失和风格损失 

使用imagenet作为训练数据集

## 文件说明
+ model.py 为神经网络模型
+ transform.py 定义了一些辅助工具函数
+ main.py 为主函数，主要实现训练和生成的过程
+ pdf 文件为报告展示
+ \content 文件夹下放要转换的图片，生成的图片应该尽可能有相似的内容
+ \style 下方定义风格的图片，生成的图片应该尽可能有相似的风格
+ \results 存放一些结果图
+ \dump 调试过程中产生的一些中间结果


## 示例

![demo](https://github.com/TrueNobility303/vgg-image-style-tranfer/blob/master/results/arc.png)


