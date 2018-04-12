import numpy as np
from PIL import Image
import os
import pandas as pd
import random

# 用python的yield 批读取处理图片
def readImageandLandmark(File, batch_size=32, rootpath='../../Tianchi_Landmark/croped_data/train/'):
    # File 是一个ndarry
    clothes = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
    while True:
        slice = np.array(random.sample(list(File), batch_size))

        # 处理图像
        x_train = []
        for i in slice:
            img = np.array(Image.open(os.path.join(rootpath,i[0])))
            x_train.append(img)

        x_train = np.array(x_train)
        x_train = x_train / 255

        # 处理坐标
        Anno = []
        for i, v in enumerate(slice):
            temp = []
            for j in v[2:]:
                temp.append(j.split('_'))
            Anno.append(temp)

        Anno = np.array(Anno, dtype='float32')

        # 解决计算NP的问题
        NP = []
        for i, v in enumerate(slice):
            if v[1] == clothes[0] or v[1] == clothes[1] or v[1] == clothes[2]:
                a = np.array(Anno[i][5:7][:, 0:2], dtype='float32')
            else:
                a = a = np.array(Anno[i][15:17][:, 0:2], dtype='float32')
            NP.append(computeNP(a))

        # 对坐标值进行归一化，落在[-1,1]之间
        Anno[:, :, 0:2] = (Anno[:, :, 0:2] - 256) / 256

        NP = np.array(NP)
        NP = np.repeat(NP, 3, axis=1)

        # 把NP 和 anno 堆放在一起
        y_train = []
        for i, v in enumerate(Anno):
            v = np.row_stack((v, NP[i]))
            y_train.append(v)
        y_train = np.array(y_train)

        # print(x_train.shape, y_train.shape)
        yield (x_train, y_train)
        
# 上衣、外套、连衣裙为两个腋窝点欧式距离，裤子和半身裙为两个裤头点的欧式距离

def computeNP(a):
    NP = np.sqrt(np.sum(np.square(a[0, :2] - a[1, :2]), axis=-1, keepdims=True))
    return NP