from Model import *
from Utils import *
from Losses import *
import os

input_shape = (512, 512, 3)
batch_size = 2
model = change_vgg16(input_shape)
weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model.load_weights(weight_path, by_name=True, skip_mismatch=True)

if os.path.exists("vggbased_weight.h5"):
    model.load_weights("vggbased_weight.h5", by_name=True)

optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, clipnorm=5.0)
model.compile(optimizer=optimizer, loss=NEloss)

annoPath = '../../Tianchi_Landmark/croped_data/train/Annotations/train.csv'
csv_handle = pd.read_csv(annoPath)

file = np.array(csv_handle)

epoch = 1000
for e in range(epoch):
    model.fit_generator(
        readImageandLandmark(file, batch_size), 
        steps_per_epoch=256, 
        epochs=1, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=2)
        # validation_data=YoloBasedGenerator(datalist, point_num, grid_length = 32, batch_size = batch_size, val_ratio = 0.9, data_type='val'), 
        # validation_steps=5)
    model.save_weights("vggbased_weight.h5")





for i in range(10):
    model.fit_generator(readImageandLandmark(file, batch_size), steps_per_epoch=100, epochs=10,use_multiprocessing=True,max_queue_size=100,workers=2)
    model.save_weights('model.h5') 