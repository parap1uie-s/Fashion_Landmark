from keras import backend as K
import tensorflow as tf

with tf.device('/cpu:0'):
    def NEloss(y_true, y_pred):
        NP = y_true[:, 24, 0]
        NP = K.reshape(NP, [-1, 1, 1])
        NP = K.maximum(NP,1) / 128
        # 把坐标值弄出来
        y_true = y_true[:, 0:24, :]
        #print(y_true)

        # 得到真实坐标的可见性，然后将其转换为0和1的值
        vis = y_true[:, :, 2]
        y = K.ones(shape=(1))  # 下行equal中的y必须是tensor，就先声明一个
        vis = K.equal(vis, y)  # 结果是bool类型
        vis = K.cast(vis, dtype='float32')  # 转变数据类型，成为0和1
        #print(vis)

        # 得到不含v(可见性)的坐标值
        y_true_no_v = y_true[:, :, 0:2]

        # 坐标轴归一到[-1,1],预测时记得还原
        y_true_no_v = (y_true_no_v - 128) / 128
        y_pred = (y_pred - 128) / 128

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