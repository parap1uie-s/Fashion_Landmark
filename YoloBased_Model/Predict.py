from Model import *
from Utils import *
import os
import pandas as pd
import numpy as np
from PIL import Image

width = 512
height = 512
point_num = 10
point_classes = 24

config = {'vis_threshold':0.5, 'grid_length':32}

model = YoloBasedModel(width, height, point_num, point_classes, phase='predict')
rootpath = '../../Tianchi_Landmark/croped_data/test/'
filepath = os.path.join(rootpath,'test.csv')
read_file_handle = pd.read_csv(filepath)
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
    assert len(points) == 72
    for p in range(0,72,3):
        res_line.append( '_'.join([ points[p],points[p+1],points[p+2] ]) )
    dest_file.write(','.join(res_line) + '\n')

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
                # if cache[0] < config['vis_threshold']:
                #     continue
                # 预测的是哪一个点
                point_index = np.argmax(cache[3:])
                if prob_max[point_index] < cache[point_index+3]:
                    prob_max[point_index] = cache[point_index+3]
                    position[point_index, 0] = round( (cache[1] + config['grid_length']) * w - offset_x)
                    position[point_index, 1] = round( (cache[2] + config['grid_length']) * h - offset_y)
                    position[point_index, 2] = round(cache[0])
    return position.astype(int)

def img_type_filter(points, img_type):
    fit = {
    'blouse':[]
    'dress':[]
    'outwear':[]
    'skirt':[]
    'trousers':[]
    }
    pad = {
    'blouse':{}
    'dress':{}
    'outwear':{}
    'skirt':{}
    'trousers':{}
    }
    assert img_type in fit.keys(), 'invalid img type'
    fit = fit[img_type]
    filted_points = np.ones((24,3)) * -1

    for key,point in enumerate(points):
        if key in fit:
            continue
        if point[0] == point[1] == point[2] == -1:
            filted_points[key] = pad[img_type][key]
        else:
            filted_points[key] = point
    return filted_points
