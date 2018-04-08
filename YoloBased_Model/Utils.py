import numpy as np
from PIL import Image
import random
import os
import pandas as pd

# darknet最后输出的卷积图尺寸是(16,16)
# 512 / 16 = 32

def YoloBasedGenerator(datalist, point_num, grid_length = 32, batch_size = 16, val_ratio = 0.9, data_type='train'):
    rootpath = '../../Tianchi_Landmark/croped_data/train/'
    # datalist = [(filename, (x1,y1,v1,grid),(x2,y2,v2,grid)...(x24,y24,v24,grid)) , (filename......)]
    
    if data_type is "train":
        datalist = datalist[0:round(len(datalist) * val_ratio)]
    else:
        datalist = datalist[round(len(datalist) * val_ratio):]
    total_num = len(datalist)

    while True:
        point_groundtruth = np.zeros((batch_size, 16, 16, point_num, 27 ))
        # Imgs = np.zeros((batch_size, 512, 512, 3))
        # point_groundtruth = []
        Imgs = []
        ind = np.random.choice(total_num, batch_size, replace=False)

        for count,i in enumerate(ind):
            grid_counter = 256*[0]
            fileName = datalist[i][0]
            Imgs.append( np.array(Image.open( os.path.join(rootpath,fileName) )) )
            for key, point in enumerate(datalist[i][1:]):
                if int(point[2]) == -1:
                    continue
                grid_num = int(point[3])
                grid_w = int(grid_num % 16)
                grid_h = int(grid_num / 16)
                k = [0] * 24
                k[key] = 1
                try:
                    point_groundtruth[count, grid_w, grid_h, grid_counter[grid_num]] = [point[2], point[0], point[1]] + k
                except Exception as e:
                    print(e)
                    continue
                
                grid_counter[grid_num] += 1
        Imgs = np.array(Imgs) / 255.0
        yield [Imgs, point_groundtruth], np.zeros((batch_size))

# 将原始的csv文件中的坐标，转换为24组x,y,v及格子编号
def ConvertToDatalist(datafile, destFile='train_new.csv', grid_length=32):
    assert os.path.isfile(datafile), '文件不存在！'

    save_handle = open(destFile, 'w+', encoding="UTF-8")
    csv_handle = pd.read_csv(datafile)

    for row in csv_handle.iterrows():
        r = row[1]
        fileName = r['image_id']
        split_axis = r.ix[2:].str.split(pat='_', expand=True)
        split_axis.columns = ['x', 'y', 'vis']

        split_axis['grid'] = split_axis.loc[split_axis['vis']!='-1','x'].astype('float').div(grid_length).astype('int') \
        + split_axis.loc[split_axis['vis']!='-1','y'].astype('float').div(grid_length).astype('int') * 16
        split_axis = split_axis.fillna('-1')

        split_axis.loc[split_axis['vis']!='-1','x'] = split_axis.loc[split_axis['vis']!='-1','x'].astype('float').mod(grid_length)
        split_axis.loc[split_axis['vis']!='-1','y'] = split_axis.loc[split_axis['vis']!='-1','y'].astype('float').mod(grid_length)

        res = [fileName] + [r['image_category']] + [','.join(i) for i in split_axis.astype('str').values.tolist()]
        save_handle.write(','.join(res) + "\n")
        save_handle.flush()
    save_handle.close()

def GetDatalist(datafile):
    assert os.path.isfile(datafile), '文件不存在！'
    csv_handle = pd.read_csv(datafile)

    res = []
    for row in csv_handle.iterrows():
        temp = []
        r = row[1]
        fileName = r[0]
        temp.append(fileName)

        split_axis = r.ix[2:].values.tolist()
        for ind in range(0,len(split_axis),4):
            temp.append((split_axis[ind],split_axis[ind+1],split_axis[ind+2],split_axis[ind+3]))
        res.append(temp)
    return res
