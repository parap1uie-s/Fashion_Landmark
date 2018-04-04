import os
import numpy as np
from PIL import Image
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_img(Img, savePath, fileName):
    W,H = Img.size

    Img = Img.resize((512, 512),Image.ANTIALIAS)
    Img.save(os.path.join(savePath,fileName))

    W_ratio = 512.0 / W
    H_ratio = 512.0 / H
    return W_ratio, H_ratio

def read_anno(annoPath, savePath, loadPath):
    csv_handle = pd.read_csv(annoPath)
    save_handle = open(os.path.join(savePath,"Annotations") + '/train.csv', 'w+', encoding="UTF-8")
    save_handle.write( ','.join(csv_handle.columns)+"\n")

    for row in csv_handle.iterrows():
        r = row[1]
        fileName = r['image_id']
        Img = Image.open(os.path.join(loadPath, fileName))
        W_ratio, H_ratio = resize_img(Img, savePath, fileName)
        split_axis = r.ix[2:].str.split(pat='_', expand=True)
        split_axis.columns = ['x', 'y', 'vis']

        split_axis.loc[split_axis['vis']!='-1','x'] = np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='float32') * W_ratio
        split_axis.loc[split_axis['vis']!='-1','y'] = np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='float32') * H_ratio

        res = [fileName] + [r['image_category']] + ['_'.join(i) for i in split_axis.astype('str').values.tolist()]
        save_handle.write(','.join(res) + "\n")
        save_handle.flush()
    save_handle.close()

def read_test(annoPath, savePath, loadPath):
    csv_handle = pd.read_csv(annoPath)
    save_handle = open(savePath + '/test.csv', 'w+', encoding="UTF-8")
    save_handle.write( ','.join(csv_handle.columns) +"W_ratio,H_ratio" +"\n")

    for row in csv_handle.iterrows():
        r = row[1]
        fileName = r['image_id']
        Img = Image.open(os.path.join(loadPath, fileName))
        W_ratio, H_ratio = resize_img(Img, savePath, fileName)

        res = [fileName] + [r['image_category']] + [str(W_ratio), str(H_ratio)]
        save_handle.write(','.join(res) + "\n")
        save_handle.flush()
    save_handle.close()

if __name__ == '__main__':
    read_anno('../data/train/Annotations/train.csv', 'train/', '../data/train/')
    # read_test('../data/test/test.csv', 'test/', '../data/test/')
