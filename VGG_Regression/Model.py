from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten, Reshape

def change_vgg16(input_shape):
    # 改变vgg最后一层，直接回归一个24 x 2的预测值，不考虑可见性
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
    x_landmark = Dense(48, name='prediction_landmark', activation='linear')(x_final_dense)
    x_landmark = Reshape((24, 2))(x_landmark)

    # visability block
    # x_vis = Dense(24, name='prediction_vis', activation='sigmoid')(x_final_dense)

    # model = Model(img_input, outputs=[x_landmark, x_vis], name='change_vgg16')
    model = Model(img_input, outputs=x_landmark, name='change_vgg16')

    return model