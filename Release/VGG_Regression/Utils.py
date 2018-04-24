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
    origin_shape = 512

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
                Img_object = Image.open(os.path.join(rootpath,r['image_id']))
                split_axis.loc[split_axis['vis']!='-1','x'] = np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='float32')
                split_axis.loc[split_axis['vis']!='-1','y'] = np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='float32')
            else:
                # crop
                y1 = max(y1 - offset, 0)
                x1 = max(x1 - offset, 0)
                y2 = min(y2 + offset, origin_shape)
                x2 = min(x2 + offset, origin_shape)

                Img_object, hori_pad, vert_pad = pad_img(Image.open(os.path.join(rootpath,r['image_id'])).crop((x1,y1,x2,y2)), origin_shape)

                split_axis.loc[split_axis['vis']!='-1','x'] = (np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='float32') - x1) + hori_pad
                split_axis.loc[split_axis['vis']!='-1','y'] = (np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='float32') - y1) + vert_pad

            if data_augment:
                aug =  random.randint(0,2)
                # 水平翻转
                if aug == 1:
                    Img_object = Img_object.transpose(Image.FLIP_LEFT_RIGHT)
                    split_axis.loc[split_axis['vis']!='-1','x'] = origin_shape - split_axis.loc[split_axis['vis']!='-1','x']
                elif aug == 2:
                    Img_object = Img_object.transpose(Image.FLIP_TOP_BOTTOM)
                    split_axis.loc[split_axis['vis']!='-1','y'] = origin_shape - split_axis.loc[split_axis['vis']!='-1','y']

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

def pad_img(Img, longer_side):
    # longer_side = max(Img.size)
    horizontal_padding = (longer_side - Img.size[0]) / 2
    vertical_padding = (longer_side - Img.size[1]) / 2
    Img = Img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            Img.size[0] + horizontal_padding,
            Img.size[1] + vertical_padding
        )
    )

    return Img, horizontal_padding, vertical_padding
