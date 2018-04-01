import os
import random
import h5py
import keras
import numpy as np
from PIL import Image
import pandas as pd
# from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten, Reshape
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model
import tensorflow as tf
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# 一张图片对应72个点 24 * 3 = 72
clothes = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']


# 上衣、外套、连衣裙为两个腋窝点欧式距离，裤子和半身裙为两个裤头点的欧式距离

def computeNP(a):
    NP = np.sqrt(np.sum(np.square(a[0, :2] - a[1, :2]), axis=-1, keepdims=True))
    return NP

# 用python的yield 批读取处理图片
def readImageandLandmark(File, batch_size=32):
    # File 是一个ndarry

    while True:
        slice = np.array(random.sample(list(File), batch_size))

        # 处理图像
        x_train = []
        for i in slice:
            img = np.array(Image.open(i[0]))
            x_train.append(img)

        x_train = np.array(x_train)
        x_train = x_train / 255

        # 处理坐标
        Anno = []
        for i, v in enumerate(slice):
            temp = []
            for j in v[2:]:
                temp.append(j.split('_'))
            Anno.append(temp)

        Anno = np.array(Anno, dtype='float32')

        # 解决计算NP的问题
        NP = []
        for i, v in enumerate(slice):
            if v[1] == clothes[0] or v[1] == clothes[1] or v[1] == clothes[2]:
                a = np.array(Anno[i][5:7][:, 0:2], dtype='float32')
            else:
                a = a = np.array(Anno[i][15:17][:, 0:2], dtype='float32')
            NP.append(computeNP(a))

        # 对坐标值进行归一化，落在[-1,1]之间
        Anno[:, :, 0:2] = (Anno[:, :, 0:2] - 256) / 256

        NP = np.array(NP)
        NP = np.repeat(NP, 3, axis=1)

        # 把NP 和 anno 堆放在一起
        y_train = []
        for i, v in enumerate(Anno):
            v = np.row_stack((v, NP[i]))
            y_train.append(v)
        y_train = np.array(y_train)

        # print(x_train.shape, y_train.shape)
        yield (x_train, y_train)


def change_vgg16(input_shape):
    # 改变vgg最后一层，直接回归一个72的预测值
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # 使用1x1卷积
    x = Conv2D(128, (1, 1), activation='relu', padding='same', name='block5_conv4')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x_final_dense = Dense(4096, activation='relu', name='fc2')(x)

    # regress block
    x_landmark = Dense(48, name='prediction_landmark')(x_final_dense)
    x = Reshape((24, 2))(x_landmark)

    # x = [Dense(3, activation='softmax', name='p%d' % (i + 1))(x_final_dense) for i in range(24)]
    output = [x]
    model = Model(img_input, outputs=output, name='change_vgg16')

    return model


# 损失函数为官网的NE只考虑v = 1的情况
# 要考虑不同衣服对应的归一化参数不同
# loss是一个tensor运算

# y_true(batchsize,25,3)(25 = 24 + 1:1代表每一张图片的归一化参数，参考比赛的公式定义)
with tf.device('/cpu:0'):
    def NEloss(y_true, y_pred):
        NP = y_true[:, 24, 0]
        # NP = K.flatten(NP)
        # NP, _ = tf.unique(NP)

        NP = K.reshape(NP, [-1, 1, 1])

        # 把坐标值弄出来
        y_true = y_true[:, 0:24, :]
        print(y_true)

        # vis = tf.shape(y_true[:,:,2])

        # 得到真实坐标的可见性，然后将其转换为0和1的值
        vis = y_true[:, :, 2]
        y = K.ones(shape=(1))  # 下行equal中的y必须是tensor，就先声明一个
        vis = K.equal(vis, y)  # 结果是bool类型
        vis = K.cast(vis, dtype='float32')  # 转变数据类型，成为0和1
        print(vis)

        # 得到不含v的坐标值
        y_true_no_v = y_true[:, :, 0:2]

        # 坐标轴归一到[-1,1],预测时记得还原
        # y_true_no_v = (y_true_no_v - 256) / 256
        y_pred = (y_pred - 256) / 256

        # 每一个元素进行重复，0,1 -> 0,0,1,1
        vis = K.reshape(K.repeat_elements(vis, 2, axis=1), [-1, 24, 2])

        # 根据v得到可计算的点
        temp_t = tf.multiply(y_true_no_v, vis)
        temp_p = tf.multiply(y_pred, vis)

        temp_r = K.reshape(K.sum(K.square(temp_t - temp_p), axis=2), [-1, 24, 1]) / NP
        # 介于0-1之间，进行开根运算要比原数大
        # temp_r = K.reshape(tf.sqrt(K.sum(K.square(temp_t - temp_p), axis=2)), [-1, 24, 1]) / NP

        # 归一化距离，变换维度，方面下面统计非0元素的个数，降了一维（变成(batch_size,24)
        value = temp_r[:, :, 0]

        count = K.maximum(tf.count_nonzero(value, dtype='float32'), y)

        neloss = K.sum(temp_r) / count

        # 出现nan是梯度爆炸
        return neloss

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = change_vgg16(input_shape)

    model.summary()
    # plot_model(model, to_file='change_vgg16.png')
    weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weight_path, by_name=True, skip_mismatch=True)

    start_time = time.time()
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer=sgd, loss=NEloss)

    annoPath = 'Annotations/train.csv'
    csv_handle = pd.read_csv(annoPath)

    file = np.array(csv_handle)
    batch_size = 16

    for i in range(10):
        parallel_model.fit_generator(readImageandLandmark(file, batch_size), steps_per_epoch=100, epochs=10)
        model.save_weights('model_2.h5')

