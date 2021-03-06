from model import FeatureNet, TransformerNet
import torch as t
import torchvision as tv
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
from transform import gram_matrix
from transform import batch_normalize
import matplotlib.pyplot as plt  

device=t.device('cuda:3') if t.cuda.is_available() else t.device('cpu')

class TransModel():
    def __init__(self):
        self.transformer = TransformerNet().to(device)
        self.extracter = FeatureNet().eval().to(device)
        self.lr = 1e-3
        self.optimizer = t.optim.Adam(self.transformer.parameters(), self.lr)
        self.content_weight = 1e5
        self.style_weight = 1e10
        #style1 1e10
        #style2 1e11
        self.epoches = 3

    def train(self,dataloader,style):
        with t.no_grad():
            features_style = self.extracter(style)
            gram_style = [gram_matrix(y) for y in features_style]
            #Bx64x64 的channel相似度矩阵

        for epoch in range(self.epoches):
            for i, (x, _) in tqdm.tqdm(enumerate(dataloader)):
                
                self.optimizer.zero_grad()

                x = x.to(device)
                y = self.transformer(x)

                if i > 100:
                    break
                #y = batch_normalize(y)
                #x = batch_normalize(x)
                features_y = self.extracter(y)
                features_x = self.extracter(x)
            
                #使用relu2的值计算内容的损失
                content_loss = self.content_weight * F.mse_loss(features_y.relu2, features_x.relu2)
                
                gram_y = [gram_matrix(y) for y in features_y]
                style_loss = 0
                for i in range(len(gram_y)):
                    style_loss += F.mse_loss(gram_y[i], gram_style[i].expand_as(gram_y[i]))
                style_loss =  self.style_weight * style_loss 
                #print(style_loss)

                loss = content_loss + style_loss
                loss.backward()
                self.optimizer.step()

            if epoch % 1 == 0:
                plt.figure()
                
                origin_img = x.data.cpu()[1].permute(1,2,0)
                style_img = style.cpu()[0].permute(1,2,0)
                new_img  = y.data.cpu()[1].permute(1,2,0)
                
                plt.subplot(131)
                plt.imshow(origin_img)
                plt.xticks([]),plt.yticks([])
                plt.subplot(132)
                plt.imshow(style_img)
                plt.xticks([]),plt.yticks([])
                plt.subplot(133)
                plt.imshow(new_img)
                plt.xticks([]),plt.yticks([])

                plt.savefig('./dump/' + str(epoch) +'.png')
                plt.close()
                #path = './dump/' + str(epoch) +'.png'
                #tv.utils.save_image(y.data.cpu()[0].clamp(min=0, max=1), path)

    def stylise(self,style,content,save_path):
        plt.figure()
            
        origin_img = content.cpu()[0].permute(1,2,0)
        style_img = style.cpu()[0].permute(1,2,0)

        y = self.transformer(content) 
        new_img  = y.data.cpu()[0].permute(1,2,0)
        
        plt.subplot(131)
        plt.imshow(origin_img)
        plt.xticks([]),plt.yticks([])
        plt.subplot(132)
        plt.imshow(style_img)
        plt.xticks([]),plt.yticks([])
        plt.subplot(133)
        plt.imshow(new_img)
        plt.xticks([]),plt.yticks([])

        plt.savefig(save_path)
        plt.close()

    #def test(self,content,save_path):
    #    output = self.transformer(content)


img_size = 256
img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]
myTransform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(img_size),
        tv.transforms.CenterCrop(img_size),
        #tv.transforms.Normalize(mean=img_mean, std=img_std),
    ])

#使用imagenet的测试集作为数据集,共10000张图片
dataset = tv.datasets.ImageFolder("./tiny-imagenet-200/test", myTransform)
dataloader = DataLoader(dataset, 8)

style_image = tv.datasets.folder.default_loader('style1.jpg')
style = myTransform(style_image).unsqueeze(0).to(device)

content_image = tv.datasets.folder.default_loader('arc.jpg')
content = myTransform(content_image).unsqueeze(0).to(device)

trans = TransModel()
trans.train(dataloader,style)
trans.stylise(style,content,'./results/arc.png')