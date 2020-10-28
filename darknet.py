import numpy as np

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Conv2D, BatchNormalization, Activation, ReLU, Flatten, LeakyReLU
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Input, GlobalAveragePooling2D, Lambda, concatenate, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from ResNetModel import BinaryConv2D
    

#***************************************************************************************************
#Definition of Darknet2019. Follows https://pjreddie.com/darknet/imagenet/#darknet53
# Inspired by https://github.com/jmpap/YOLOV2-Tensorflow-2.0/blob/master/Yolo_V2_tf_2.ipynb
#***************************************************************************************************
def darknet19(input_shape, num_classes=10):

    inputs = Input(shape=input_shape)     
    
    # Layer 0
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(inputs)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 1
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 3
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 4
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 6
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 7
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 9
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 11
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 12
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 14
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    #Layer 17
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(num_classes, (1,1), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 24 
    x = GlobalAveragePooling2D()(x)
    
    OutputPath = Activation('softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model

def sRelu(x):
    x=Lambda(lambda z: z + 1)(x)
    x = ReLU()(x)
    x=Lambda(lambda z: z - 1)(x)
    return x
#
#  Binary Darknet 2019
#
def darknet19_binary(input_shape, num_classes=10):

    inputs = Input(shape=input_shape)     
    
    # Layer 0
    x = BinaryConv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(inputs)
    x = sRelu(x)
    
    # Layer 1
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = BinaryConv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = sRelu(x)
    
    # Layer 3
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 4
    x = BinaryConv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = sRelu(x)

    # Layer 5
    x = BinaryConv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = sRelu(x)

    # Layer 6
    x = BinaryConv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = sRelu(x)
    
    # Layer 7
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 8
    x = BinaryConv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = sRelu(x)

    # Layer 9
    x = BinaryConv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = sRelu(x)

    # Layer 10
    x = BinaryConv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = sRelu(x)
    
    # Layer 11
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 12
    x = BinaryConv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = sRelu(x)

    # Layer 13
    x = BinaryConv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = sRelu(x)

    # Layer 14
    x = BinaryConv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = sRelu(x)

    # Layer 15
    x = BinaryConv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = sRelu(x)

    # Layer 16
    x = BinaryConv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = sRelu(x)

    #Layer 17
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 18
    x = BinaryConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = sRelu(x)

    # Layer 19
    x = BinaryConv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = sRelu(x)

    # Layer 20
    x = BinaryConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = sRelu(x)

    # Layer 21
    x = BinaryConv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = sRelu(x)

    # Layer 22
    x = BinaryConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = sRelu(x)

    # Layer 23
    x = Conv2D(num_classes, (1,1), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)

    # Layer 24 
    x = GlobalAveragePooling2D()(x)
    
    OutputPath = Activation('softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model