import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image


class ImageTransform():
    """
    图像的预处理类。训练时和推测时采用不同的处理方式
   对图像的大小进行调整，并将颜色信息标准化
    训练时采用 RandomResizedCrop 和 RandomHorizontalFlip 进行数据增强处理

    Attributes
    ----------
    resize : int
      指定调整尺寸后图片的大小
    mean : (R, G, B)
       各个颜色通道的平均值
    std : (R, G, B)
      各个颜色通道的标准偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)), #数据增强处理
                transforms.RandomHorizontalFlip(),  #数据增强处理
                transforms.ToTensor(), #转换为张量
                transforms.Normalize(mean, std)  #归一化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  #调整大小
                transforms.CenterCrop(resize),  #从图片中央截取resize × resize大小的区域
                transforms.ToTensor(), #转换为张量
                transforms.Normalize(mean, std)  #归一化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            指定预处理模式。
        """
        return self.data_transform[phase](img)


def make_datapath_list(phase="train"):
    """
    制作存储数据路径的列表。

    Parameters
    ----------
    phase : 'train' or 'val'
       指定训练数据还是验证数据

    Returns
    -------
    path_list : list
        存储通往数据的路径的列表
    """

    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = []  #把标改装数

    #保存在这里
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class HymenopteraDataset(data.Dataset):
    """
   蚂蚁和蜜蜂图片的Dataset类，继承自PyTorch的Dataset类

    Attributes
    ----------
    file_list : 列表
       列表中保存了图片路径
    transform : object
      预处理类的实例
    phase : 'train' or 'test'
      设定学习还是训练。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # 文件路径列表
        self.transform = transform  # 预处理类的实例
        self.phase = phase  #指定是train 还是val

    def __len__(self):
        '''返回图像的张数'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        获取预处理完毕的图片的张量数据和标签
        '''

        # 载入第index张图片
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高度][宽度][颜色RGB]

        #对图像进行预处理
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        #从文件名中抽出图像的标签
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

       #将标签转换为数字
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label
