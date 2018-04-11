from Model import *
from Utils import *
import os
import pandas as pd
import numpy as np
from PIL import Image

def exec_res(predict_result, config, offset_x, offset_y):
    res = predict_result[0]

    prob_max = np.zeros((24))
    position = np.ones((24,3)) * -1
    # point_classes = point_classes + 3
    grad_w, grad_h, point_num, point_classes = res.shape
    for h in range(grad_h):
        for w in range(grad_w):
            for p in range(point_num):
                cache = res[w,h,p]
                if cache[0] < config['vis_threshold']:
                    continue
                # 预测的是哪一个点
                point_index = np.argmax(cache[3:])
                if prob_max[point_index] < cache[point_index+3]:
                    temp_x = round( (cache[1] + config['grid_length']) * w - offset_x)
                    temp_y = round( (cache[2] + config['grid_length']) * h - offset_y)
                    if temp_x < 0 or temp_y < 0:
                        continue
                    prob_max[point_index] = cache[point_index+3]
                    position[point_index, 0] = temp_x
                    position[point_index, 1] = temp_y
                    position[point_index, 2] = round(cache[0])
    return position.astype(int)

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
    width = 512
    height = 512
    point_num = 10
    point_classes = 24

    config = {'vis_threshold':0, 'grid_length':32}

    model = YoloBasedModel(width, height, point_num, point_classes, phase='evaluate')
    rootpath = '../../Tianchi_Landmark/croped_data/test/'
    filepath = os.path.join(rootpath,'test.csv')
    csv_handle = pd.read_csv(filepath)
    dest_file = open('result.txt','w+',encoding='utf-8')

    dest_file.write('image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,\
        shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,\
        cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,\
        hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out' + '\n')

    if os.path.exists("yolobased_weight.h5"):
        model.load_weights("yolobased_weight.h5", by_name=True)

    for row in csv_handle.iterrows():
        r = row[1]
        img = np.expand_dims(np.array( Image.open( os.path.join(rootpath,r[0]) )),axis=0)
        img_type = r[1]
        hori_pad = r[2]
        vert_pad = r[3]
        
        res_line = [r[0],img_type]

        points = exec_res(model.predict(img), config, hori_pad, vert_pad)
        points = img_type_filter(points, img_type)
        assert len(points) == 24
        for p in range(0,len(points)):
            res_line.append( '_'.join([ str(points[p][0].astype(int)),str(points[p][1].astype(int)),str(points[p][2].astype(int)) ]) )
        dest_file.write(','.join(res_line) + '\n')


