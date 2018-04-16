import keras.layers as KL

class ResNet(object):
    """docstring for ResNet101"""
    def __init__(self, input_tensor, architecture='resnet50'):
        self.keras_model = ""
        self.input_tensor = input_tensor
        self.output_layers = ""
        assert architecture in ['resnet50', 'resnet101'], 'architecture must be resnet50 or resnet101!'
        self.architecture = architecture
        self.construct_graph(input_tensor)
        
    def construct_graph(self, input_tensor, stage5=True):
        assert self.input_tensor is not None, "input_tensor can not be none!"
        # Stage 1
        x = KL.ZeroPadding2D((3, 3))(input_tensor)
        x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = BatchNorm(axis=3, name='bn_conv1')(x)
        x = KL.Activation('relu')(x)
        C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        C2 = x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        # Stage 3
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        C3 = x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        # Stage 4
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        block_count = {"resnet50": 5, "resnet101": 22}[self.architecture]
        for i in range(block_count):
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
        C4 = x
        # Stage 5
        if stage5:
            x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            C5 = x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        else:
            C5 = None
        self.output_layers = [C1, C2, C3, C4, C5]

    def conv_block(self, input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)(input_tensor)
        x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)