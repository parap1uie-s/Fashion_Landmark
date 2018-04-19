import numpy as np
from PIL import Image
import os
import pandas as pd
import random

# 用python的yield 批读取处理图片
def computeNP(a):
    NP = np.sqrt(np.sum(np.square(a[0, :2] - a[1, :2]), axis=-1, keepdims=True))
    return NP

def readImageandLandmark(annoPath, batch_size=32, rootpath='../../Tianchi_Landmark/croped_data/train/', data_augment=True):
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
                y2 = min(y2 + offset, origin_shape)
                x2 = min(x2 + offset, origin_shape)

                Img_object = Image.open(os.path.join(rootpath,r['image_id'])).crop((x1,y1,x2,y2)).resize((resize_shape,resize_shape),Image.ANTIALIAS)

                split_axis.loc[split_axis['vis']!='-1','x'] = (np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='float32') - x1) * (resize_shape / (x2 - x1))
                split_axis.loc[split_axis['vis']!='-1','y'] = (np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='float32') - y1) * (resize_shape / (y2 - y1))

            if data_augment:
                aug =  random.randint(0,2)
                # 水平翻转
                if aug == 1:
                    Img_object = Img_object.transpose(Image.FLIP_LEFT_RIGHT)
                    split_axis.loc[split_axis['vis']!='-1','x'] = resize_shape - split_axis.loc[split_axis['vis']!='-1','x']
                elif aug == 2:
                    Img_object = Img_object.transpose(Image.FLIP_TOP_BOTTOM)
                    split_axis.loc[split_axis['vis']!='-1','y'] = resize_shape - split_axis.loc[split_axis['vis']!='-1','y']

            x_train.append(np.array(Img_object))

            # compute NP
            if r['image_category'] in clothes[0:3]:
                NP = computeNP(np.array(split_axis[['x','y']].iloc[5:7].values,dtype='float32'))
            else:
                NP = computeNP(np.array(split_axis[['x','y']].iloc[15:17].values,dtype='float32'))
            NP = np.array(NP).repeat(3,axis=0)
            y_train.append( np.row_stack((  np.array(split_axis.values, dtype='float32') ,NP)) )

        x_train = np.array(x_train) / 255.0
        y_train = np.array(y_train)

        yield x_train, y_train


