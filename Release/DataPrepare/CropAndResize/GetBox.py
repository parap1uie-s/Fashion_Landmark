import os
import numpy as np
import tensorflow as tf
import coco
import utils
import model as modellib
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    DEVICE = "/GPU:0"
    TEST_MODE = "inference"

    config = coco.CocoConfig()
    COCO_DIR = "path to COCO dataset"  # TODO: enter value here
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    weights_path = COCO_MODEL_PATH
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # csv_handle_train = pd.read_csv(os.path.join(ROOT_DIR, "../../Tianchi_Landmark/croped_data/train/Annotations/train.csv"))
    csv_handle_test = pd.read_csv(os.path.join(ROOT_DIR, "../../Tianchi_Landmark/croped_data/test/test.csv"))

    # Train Images
    # save_handle = open('train.csv', 'w+', encoding="UTF-8")
    # save_handle.write( ','.join(csv_handle_train.columns)+",y1,x1,y2,x2\n")
    # for row in csv_handle_train.iterrows():
    #     r = row[1]
    #     fileName = r['image_id']
    #     Img = np.array(plt.imread(os.path.join("../../Tianchi_Landmark/croped_data/train/", fileName)))
    #     results = model.detect([Img], verbose=1)
    #     seg_res = results[0]

    #     split_axis = r.ix[2:].str.split(pat='_', expand=True)
    #     split_axis.columns = ['x', 'y', 'vis']

    #     point_x = np.mean(split_axis.loc[split_axis['vis']!='-1','x'].astype('float32').values.tolist())
    #     point_y = np.mean(split_axis.loc[split_axis['vis']!='-1','y'].astype('float32').values.tolist())

    #     y1, x1, y2, x2 = 0,0,0,0
    #     for k, box in enumerate(seg_res['rois']):
    #         if seg_res['class_ids'][k] != 1:
    #             continue
    #         if point_x < box[1] or point_x > box[3] or point_y < box[0] or point_y > box[2]:
    #             continue
    #         (y1, x1, y2, x2) = box

    #     res = r.astype('str').values.tolist() + [str(y1), str(x1), str(y2), str(x2)]
    #     save_handle.write(','.join(res) + "\n") 
    # save_handle.close()

    # Test Images
    save_handle = open('test.csv', 'w+', encoding="UTF-8")
    save_handle.write( ','.join(csv_handle_test.columns)+",y1,x1,y2,x2\n")
    for row in csv_handle_test.iterrows():
        r = row[1]
        fileName = r['image_id']
        Img = np.array(plt.imread(os.path.join("../../Tianchi_Landmark/croped_data/test/", fileName)))
        results = model.detect([Img], verbose=1)
        seg_res = results[0]

        y1, x1, y2, x2 = 0,0,0,0
        for k, box in enumerate(seg_res['rois']):
            if seg_res['class_ids'][k] != 1:
                continue
            (y1, x1, y2, x2) = box

        res = r.astype('str').values.tolist() + [str(y1), str(x1), str(y2), str(x2)]
        save_handle.write(','.join(res) + "\n") 
    save_handle.close()