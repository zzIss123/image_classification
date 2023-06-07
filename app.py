import numpy as np
from flask import Flask,request, render_template
import numpy as np 
import json
from PIL import Image 
import matplotlib.pyplot as plt 

import torch 
import torchvision 
from torchvision import models,transforms

app = Flask(__name__)

use_predicted = True  #使用已经训练好的参数
net = models.vgg16(pretrained=use_predicted)
net.eval() #设置为推理模式

class BaseTransform():
    """
    调整图片的尺寸并对颜色进行标准化
    """
    def __init__(self,resize,mean,std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    def __call__(self,img):
        return self.base_transform(img)

resize=224
mean=(0.485,0.456,0.406)
std=(0.229,0.224,0.225)

class ILSVRCPredictor():
    def __init__(self,class_index):
        self.class_index = class_index 
    def predict_max(self,out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

#载入ILSVRC的分类标签信息，并保存在字典中
ILSVRC_class_index=json.load(open('./data/imagenet_class_index.json','r'))

#生成ILSVRCPredictor的实例
predictor = ILSVRCPredictor(ILSVRC_class_index)

#读取输入的图像
#1. 读取图片
image_file_path='./data/dog.jpg'
img = Image.open(image_file_path)  #[高][宽][RGB]
@app.route('/predict',methods=['POST'])
def predict():
    img_file_path = [str(x) for x in request.form.values()]
    img = Image.open(image_file_path)
    transform = BaseTransform(resize,mean,std)
    img_transform=transform(img)
    inputs = img_transform.unsqueeze_(0)
    # img_transform.shape
    out = net(inputs)
    return render_template('page.html',prediction_display_area=predictor.predict_max(out))


 
@app.route('/')
def home():
    return render_template('page.html')
 

if __name__ == "__main__":
    app.run(port=80,debug = True)
 
 