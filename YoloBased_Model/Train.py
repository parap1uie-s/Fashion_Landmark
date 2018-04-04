from Model import *
from Utils import *
import os
import random
random.seed(1)

width = 512
height = 512
point_num = 5
point_classes = 24
model = YoloBasedModel(width, height, point_num, point_classes, phase='train')

if os.path.exists("yolobased_weight.h5"):
    model.load_weights("yolobased_weight.h5", by_name=True)

datalist = GetDatalist('train_new.csv')
random.shuffle(datalist)

epoch = 1000
for e in range(epoch):
    model.fit_generator(
        YoloBasedGenerator(datalist, point_num, grid_length = 32, batch_size = 16, val_ratio = 0.9, data_type='train'), 
        steps_per_epoch=256, 
        epochs=1, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=2,
        validation_data=YoloBasedGenerator(datalist, point_num, grid_length = 32, batch_size = 16, val_ratio = 0.9, data_type='val'), 
        validation_steps=5)
    model.save("yolobased_weight.h5")