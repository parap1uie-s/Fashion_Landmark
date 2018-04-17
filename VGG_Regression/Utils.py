import numpy as np
from PIL import Image
import os
import pandas as pd
import random

# 用python的yield 批读取处理图片
# def readImageandLandmark(annoPath, batch_size=32, rootpath='../../Tianchi_Landmark/croped_data/train/'):
    

#     while True:
#         slice = np.array(random.sample(list(File), batch_size))

#         # 处理图像
#         x_train = []
#         for i in slice:
#             img = np.array(Image.open(os.path.join(rootpath,i[0])))
#             x_train.append(img)

#         x_train = np.array(x_train)
#         x_train = x_train / 255

#         # 处理坐标
#         Anno = []
#         for i, v in enumerate(slice):
#             temp = []
#             for j in v[2:]:
#                 temp.append(j.split('_'))
#             Anno.append(temp)

#         Anno = np.array(Anno, dtype='float32')

#         # 解决计算NP的问题
#         NP = []
#         for i, v in enumerate(slice):
#             if v[1] == clothes[0] or v[1] == clothes[1] or v[1] == clothes[2]:
#                 a = np.array(Anno[i][5:7][:, 0:2], dtype='float32')
#             else:
#                 a = a = np.array(Anno[i][15:17][:, 0:2], dtype='float32')
#             NP.append(computeNP(a))

#         # 对坐标值进行归一化，落在[-1,1]之间
#         Anno[:, :, 0:2] = (Anno[:, :, 0:2] - 256) / 256

#         NP = np.array(NP)
#         NP = np.repeat(NP, 3, axis=1)

#         # 把NP 和 anno 堆放在一起
#         y_train = []
#         for i, v in enumerate(Anno):
#             v = np.row_stack((v, NP[i]))
#             y_train.append(v)
#         y_train = np.array(y_train)

#         # print(x_train.shape, y_train.shape)
#         yield (x_train, y_train)
        
# 上衣、外套、连衣裙为两个腋窝点欧式距离，裤子和半身裙为两个裤头点的欧式距离

def computeNP(a):
    NP = np.sqrt(np.sum(np.square(a[0, :2] - a[1, :2]), axis=-1, keepdims=True))
    return NP

def readImageandLandmark(annoPath, batch_size=32, rootpath='../../Tianchi_Landmark/croped_data/train/'):
    csv_handle = pd.read_csv(annoPath)
    clothes = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']

    data_num = len(csv_handle)

    offset = 5
    resize_shape = 256
    origin_shape = 512
    resize_ratio = resize_shape / origin_shape

    while True:
        x_train = []
        y_train = []
        ind = np.random.choice(data_num, batch_size, replace=False)
        choiced_data = csv_handle.iloc[ind,:]

        for row in choiced_data.iterrows():
            r = row[1]
            y1, x1, y2, x2 = r.loc[['y1','x1','y2','x2']].values.tolist()

            split_axis = r.ix[2:-4].str.split(pat='_', expand=True)
            split_axis.columns = ['x', 'y', 'vis']

            if y1 == x1 == y2 == x2 == 0:
                # no crop
                Img_object = Image.open(os.path.join(rootpath,r['image_id'])).resize((resize_shape,resize_shape),Image.ANTIALIAS)

                split_axis.loc[split_axis['vis']!='-1','x'] = np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='float32') * resize_ratio
                split_axis.loc[split_axis['vis']!='-1','y'] = np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='float32') * resize_ratio
            else:
                # crop
                y1 = max(y1 - offset, 0)
                x1 = max(x1 - offset, 0)
                y2 = min(y2 + offset, 512)
                x2 = min(x2 + offset, 512)

                Img_object = Image.open(os.path.join(rootpath,r['image_id'])).crop(x1,y1,x2,y2).resize((resize_shape,resize_shape),Image.ANTIALIAS)

                split_axis.loc[split_axis['vis']!='-1','x'] = (np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='float32') - x1) * (resize_shape / (x2 - x1))
                split_axis.loc[split_axis['vis']!='-1','y'] = (np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='float32') - y1) * (resize_shape / (y2 - y1))

            x_train.append(np.array(Img_object))

            # compute NP
            if r['image_category'] in clothes[0:3]:
                NP = computeNP(np.array(split_axis[['x','y']].loc[5:7].values))
            else:
                NP = computeNP(np.array(split_axis[['x','y']].loc[15:17].values))
            NP = np.array(NP).repeat(3,axis=0)
            y_train.append( np.row_stack((np.array(split_axis.values),NP)) )

        x_train = np.array(x_train) / 255.0
        y_train = np.array(y_train)

        yield x_train, y_train


