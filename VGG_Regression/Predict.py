from Model import *
from Utils import *
import os
import pandas as pd
import numpy as np
from PIL import Image

def exec_res(predict_result, offset_x, offset_y, hori_pad_new, vert_pad_new, crop_pos):
    res = predict_result[0]

    prob_max = np.zeros((24))
    position = np.ones((24,3)) * -1

    if crop_pos is None:
        # no crop
        for i in range(24):
            cache = res[i]
            position[i] = np.maximum(cache - [offset_x, offset_y],0).astype(int).tolist() + [1]
    else:
        y1, x1, y2, x2 = crop_pos
        # crop
        for i in range(24):
            cache = res[i]
            cache[0] = cache[0] - hori_pad_new + x1
            cache[1] = cache[1] - vert_pad_new + y1
            position[i] = np.maximum(cache - [offset_x, offset_y],0).astype(int).tolist() + [1]

    return position

def img_type_filter(points, img_type):
    fit = {
    'blouse':[0,1,2,3,4,5,6,9,10,11,12,13,14],
    'dress':[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,17,18],
    'outwear':[0, 1, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14],
    'skirt':[15,16,17,18],
    'trousers':[15,16,19,20,21,22,23]
    }
    pad = {
    'blouse':{0:[202,101,1], 1:[288,100,1],2:[244,133,1],3:[145,120,1],4:[318,119,1],5:[164,211,1],6:[324,211,1],9:[154,327,1],
    10:[121,322,1],11:[316,323,1],12:[349,319,1],13:[162,371,1],14:[325,371,1]},
    'dress':{0:[202,80,1],1:[274,80,1],2:[238,112,1],3:[152,85,1],4:[276,85,1],5:[179,156,1],6:[297,156,1],7:[131,152,1],
    8:[201,152,1],9:[127,177,1],10:[109,173,1],11:[227,176,1],12:[253,173,1],17:[148,433,1],18:[326,434,1]},
    'outwear':{0:[207,90,1],1:[271,90,1],3:[157,117,1],4:[319,117,1],5:[165,192,1],6:[311,191,1],7:[32,40,1],
    8:[51,40,1],9:[153,308,1],10:[123,307,1],11:[309,305,1],12:[338,305,1],13:[150,406,1],14:[326,406,1]},
    'skirt':{15:[173,119,1],16:[304,119,1],17:[125,388,1],18:[349,388,1]},
    'trousers':{15:[170,119,1],16:[301,119,1],19:[236,248,1],20:[218,416,1],21:[142,405,1],22:[252,416,1],23:[328,406,1]}
    }
    assert img_type in fit.keys(), 'invalid img type'
    fit = fit[img_type]
    filted_points = np.ones((24,3)) * -1

    for key,point in enumerate(points):
        if key not in fit:
            continue
        if point[0] == point[1] == point[2] == -1:
            filted_points[key] = pad[img_type][key]
        else:
            filted_points[key] = point
    return filted_points

if __name__ == '__main__':
    input_shape = (512, 512, 3)

    model = change_vgg19(input_shape)
    rootpath = '../../Tianchi_Landmark/croped_data/test/'
    filepath = os.path.join(rootpath,'test.csv')
    csv_handle = pd.read_csv(filepath)
    dest_file = open('result.csv','w+',encoding='utf-8')

    dest_file.write('image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out' + '\n')

    model.load_weights("vggbased_weight.h5", by_name=True)

    for row in csv_handle.iterrows():
        r = row[1]
        
        img_type = r[1]
        hori_pad = r[2]
        vert_pad = r[3]
        y1, x1, y2, x2 = r.loc[['y1','x1','y2','x2']].values.tolist()
        
        if y1 == x1 == y2 == x2 == 0:
            # no crop
            img = np.expand_dims(np.array( Image.open(os.path.join(rootpath,r['image_id'])) ),axis=0) / 255.0
            points = exec_res(model.predict(img), hori_pad, vert_pad, None)
        else:
            img_object = Image.open(os.path.join(rootpath,r['image_id'])).crop((x1,y1,x2,y2))
            img_object, hori_pad_new, vert_pad_new = pad_img(img_object, 512)

            img = np.expand_dims(np.array( img_object ),axis=0) / 255.0
            points = exec_res(model.predict(img), hori_pad, vert_pad, hori_pad_new, vert_pad_new, (y1, x1, y2, x2))

        res_line = [r[0],img_type]

        
        points = img_type_filter(points, img_type)
        assert len(points) == 24
        for p in range(0,len(points)):
            res_line.append( '_'.join([ str(points[p][0].astype(int)),str(points[p][1].astype(int)),str(points[p][2].astype(int)) ]) )
        dest_file.write(','.join(res_line) + '\n')


