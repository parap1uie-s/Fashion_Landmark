def NEloss(y_true, y_pred):
    # 获得NP的值，维度是(batchsize,24)
    NP = y_true[:, 24, 0]

    NP = K.reshape(NP, [-1, 1, 1])

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
    # y_true_no_v = (y_true_no_v - 256) / 256
    y_pred = (y_pred - 256) / 256

    # 每一个元素进行重复，0,1 -> 0,0,1,1
    vis = K.reshape(K.repeat_elements(vis, 2, axis=1), [-1, 24, 2])

    # 根据v得到可计算的点
    temp_t = tf.multiply(y_true_no_v, vis)
    temp_p = tf.multiply(y_pred, vis)

    neloss = K.mean(K.square(y_pred - y_true) / NP, axis=-1)

    # 出现nan是梯度爆炸
    return neloss

# 上衣、外套、连衣裙为两个腋窝点欧式距离，裤子和半身裙为两个裤头点的欧式距离

def computeNP(a):
    NP = np.sqrt(np.sum(np.square(a[0, :2] - a[1, :2]), axis=-1, keepdims=True))
    return NP