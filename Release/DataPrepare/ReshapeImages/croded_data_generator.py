import os
import numpy as np
from PIL import Image
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_img(Img, savePath, fileName):
    # longer_side = max(Img.size)
    longer_side = 512
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
    Img.save(os.path.join(savePath,fileName))

    return horizontal_padding, vertical_padding

def read_anno(annoPath, savePath, loadPath):
    csv_handle = pd.read_csv(annoPath)
    save_handle = open(os.path.join(savePath,"Annotations") + '/train.csv', 'w+', encoding="UTF-8")
    save_handle.write( ','.join(csv_handle.columns)+"\n")

    for row in csv_handle.iterrows():
        r = row[1]
        fileName = r['image_id']
        Img = Image.open(os.path.join(loadPath, fileName))
        hori_pad, vert_pad = pad_img(Img, savePath, fileName)
        split_axis = r.ix[2:].str.split(pat='_', expand=True)
        split_axis.columns = ['x', 'y', 'vis']
        split_axis.loc[split_axis['vis']!='-1','x'] = np.array(split_axis[split_axis['vis']!='-1']['x'], dtype='int32') + hori_pad
        split_axis.loc[split_axis['vis']!='-1','y'] = np.array(split_axis[split_axis['vis']!='-1']['y'], dtype='int32') + vert_pad

        res = [fileName] + [r['image_category']] + ['_'.join(i) for i in split_axis.astype('str').values.tolist()]
        save_handle.write(','.join(res) + "\n")
        save_handle.flush()
    save_handle.close()

def read_test(annoPath, savePath, loadPath):
    csv_handle = pd.read_csv(annoPath)
    save_handle = open(savePath + '/test.csv', 'w+', encoding="UTF-8")
    save_handle.write( ','.join(csv_handle.columns) +"hori_pad,vert_pad" +"\n")

    for row in csv_handle.iterrows():
        r = row[1]
        fileName = r['image_id']
        Img = Image.open(os.path.join(loadPath, fileName))
        hori_pad, vert_pad = pad_img(Img, savePath, fileName)

        res = [fileName] + [r['image_category']] + [str(hori_pad), str(vert_pad)]
        save_handle.write(','.join(res) + "\n")
        save_handle.flush()
    save_handle.close()


if __name__ == '__main__':
    read_anno('../data/train/Annotations/train.csv', 'train/', '../data/train/')
    read_test('../data/test/test.csv', 'test/', '../data/test/')
