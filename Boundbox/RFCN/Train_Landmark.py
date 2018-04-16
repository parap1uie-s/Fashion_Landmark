from Data_generator import data_generator
from Model.Model import RFCN_Model
import os
import pandas as pd
import Utils 
from Config import Config
import pickle
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

############################################################
#  Config
############################################################

class RFCNNConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Landmark"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    C = 1 + 5  # background + 2 tags
    NUM_CLASSES = C
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 300
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    RPN_NMS_THRESHOLD = 0.5
    POOL_SIZE = 7

############################################################
#  Dataset
############################################################

class FashionDataset(Utils.Dataset):
    # count - int, images in the dataset
    def initDB(self, count, start = 0):
        self.start = start
        # self.rootpath = '../../Tianchi_Landmark/croped_data/train'
        self.rootpath = '../../croped_data/train'

        csv_handle = pd.read_csv(os.path.join(self.rootpath,'Annotations/train.csv'))
        # all_images = csv_handle['image_id'].tolist()
        csv_handle = shuffle(csv_handle, random_state=1)

        assert start+count < len(csv_handle), 'the number of start must less than image num'
        assert count < len(csv_handle), 'the number of count must less than image num'

        classes = csv_handle['image_category'].unique().tolist()

        # Add classes
        self.classes = {}
        for k,c in enumerate(classes):
            self.add_class("Landmark",k+1,c)
            self.classes[k+1] = c

        k = 0
        for row in csv_handle.iloc[start:start+count].iterrows():
            r = row[1]
            split_axis = r.ix[2:].str.split(pat='_', expand=True)
            split_axis.columns = ['x', 'y', 'vis']
            split_axis.loc[split_axis['vis']!='-1','x'] = np.array(split_axis[split_axis['vis']!='-1']['x'])
            split_axis.loc[split_axis['vis']!='-1','y'] = np.array(split_axis[split_axis['vis']!='-1']['y'])

            point_x = np.array(split_axis.loc[split_axis['vis']!='-1','x'].tolist(),dtype='float32')
            point_y = np.array(split_axis.loc[split_axis['vis']!='-1','y'].tolist(),dtype='float32')

            x1 = point_x.min()
            x2 = point_x.max()
            y1 = point_y.min()
            y2 = point_y.max()

            self.add_image(source="Landmark",image_id=k, 
                path=os.path.join(self.rootpath, r['image_id']), width=512, height=512, bboxes=(y1,x1,y2,x2), category=r['image_category'])
            k += 1

    # read image from file and get the 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        # tempImg = image.img_to_array( image.load_img(info['path']) )
        tempImg = np.array(Image.open( info['path'] ))
        return tempImg

    def get_keys(self, d, value):
        return [k for k,v in d.items() if v == value]

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        bboxes = [info['bboxes']]
        labels = [self.get_keys(self.classes, info['category'])]
        return np.array(bboxes), np.array(labels)

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    config = RFCNNConfig()
    dataset_train = FashionDataset()
    dataset_train.initDB(30000)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FashionDataset()
    dataset_val.initDB(1000, start=30000)
    dataset_val.prepare()

    model = RFCN_Model(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs") )

    # model.load_weights("~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
    model.load_weights("mask_rcnn_fashion_0184.h5", by_name=True, exclude=['score_map_class_1',
        'score_map_class_2','score_map_class_3','score_map_class_4','score_map_class_5','score_map_class_6',
        'score_map_class_7','score_map_class_8','score_map_class_0','score_map_regr_1','score_map_regr_2',
        'score_map_regr_3','score_map_regr_4','score_map_regr_5','score_map_regr_6','score_map_regr_7','score_map_regr_8','score_map_regr_0'])
    
    try:
        model_path = model.find_last()[1]
        model.load_weights(model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Not checkpoint founded')
        
    # *** This training schedule is an example. Update to your needs ***

    # # Training - Stage 1
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

    # # Training - Stage 2
    # # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=80,
    #             layers='all')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=480,
                layers='all')