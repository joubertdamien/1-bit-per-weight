#!/usr/bin/env python
# coding: utf-8

# # Tensorflow.keras implementation of one-bit-per-weight shifted ReLU CNN for CIFAR 100 
# ##  https://arxiv.org/abs/1907.06916
# ## Mark D. McDonnell, Hesham Mostafa, Runchun Wang, Andre van Schaik,
# ## Single-bit-per-weight deep convolutional neural networks without batch-normalization layers for embedded systems



# select a GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
from scipy.io import savemat,loadmat
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow
print('Tensorflow version = ',tensorflow.__version__)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, History

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image

from PIL import Image
from sklearn.utils import shuffle

#from tensorflow.keras import backend as K

from ResNetModel import resnet_srelu,resnet
from Utils import cutout,LR_WarmRestart,GetDataGen,plot_history
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
from datetime import datetime

#save model history
NAME = "OneBitPerWeight_Keras_Tiny-Imagenet{}".format(datetime.now().strftime("%Y%m%d"))
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(NAME),
                            histogram_freq=1,
                            write_graph=True,
                            write_images=False,
                            update_freq="epoch",
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None)


#params
#WhichDataSet = 'CIFAR10'
#WhichDataSet = 'CIFAR100'
WhichDataSet = 'TinyImagenet'
init_lr = 0.006
epochs = 300# for teststing 300
batch_size = 125#300 
My_wd=5e-4/2
resnet_width = 10#10
resnet_depth = 18
UseBinary=True
UseBinaryWeights = True #resnet
UseCutout=False #True
Loss = 'categorical_crossentropy'
Optimizer = SGD(lr=init_lr,decay=0.0005, momentum=0.99, nesterov=False)
Metrics = ['accuracy']
ModelsPath = './TrainedModels/Tensorflow.keras/'





#load and prepare data
img_width, img_height = 64,64
train_data_dir = 'Datasets/Tiny-Imagenet-200/train/images'
validation_data_dir = 'Datasets/Tiny-Imagenet-200/train/images'




datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.1,
                            rotation_range=20,
                            zoom_range=0.15,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            fill_mode="nearest")
val_datagen = ImageDataGenerator(rescale=1./255)
test_dataset = ImageDataGenerator() 



train_generator = datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),color_mode='rgb',
                                    batch_size= batch_size, subset ="training", class_mode='categorical', shuffle=True, seed=42)
val_generator = datagen.flow_from_directory(validation_data_dir,target_size=(64,64),color_mode='rgb',
                                    batch_size= batch_size, subset ="validation", class_mode='categorical', shuffle=False, seed=42)

test_generator = test_dataset.flow_from_directory('Datasets/Tiny-Imagenet-200/test/', target_size=(64,64),color_mode='rgb',
                                    batch_size= 1, class_mode=None,shuffle=False,seed=42)


x_train, y_train = next(train_generator)
x_test, y_test = next(val_generator)


num_classes = np.unique(y_train).shape[0]
y_true = val_generator.classes
input_shape = x_train.shape[1:]



def catcross_entropy_logits_loss():
    def loss(y_true, y_pred):
        return tensorflow.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    return loss


#define a datagen or generating training samples with flip and pad/crop augmentation, and if set to True, with cutout augmentation
dataGenerator = GetDataGen(UseCutout)

#define and compile the model
'''Temperature=25.0
model = resnet_srelu(Temperature,UseBinaryWeights=True,input_shape=input_shape, num_classes=200,
                     wd=My_wd,width=resnet_width)'''

model = resnet(UseBinaryWeights,input_shape = input_shape, num_classes=200, depth = resnet_depth,width=resnet_width,wd=My_wd)
model.compile(loss=catcross_entropy_logits_loss() ,optimizer = Optimizer, metrics = Metrics)

#print  the model
model.summary()

#define the learnng rate schedule
steps_per_epoch = int(np.floor(train_generator.n // batch_size ))
lr_scheduler = LR_WarmRestart(nbatch=steps_per_epoch,
                              initial_lr=init_lr, min_lr=init_lr*1e-4,
                              epochs_restart = [1.0,3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0]) 

#define callbacks
history = History()
callbacks = [lr_scheduler,history,tensorboard]
#callbacks = [lr_scheduler,history]


history = model.fit(train_generator,
                              validation_data=val_generator,
                              epochs=epochs,
                              verbose=2,
                              use_multiprocessing=False,
                              workers=1,
                              callbacks=callbacks,
                              steps_per_epoch =steps_per_epoch)





#get final performance
y_pred = model.predict(x_test)
#y_pred = model.predict_generator(test_generator,steps =128)
print(y_pred)
print('Test accuracy (%):', 100*sum(np.argmax(y_pred,-1)==np.argmax(y_test,-1))/y_test.shape[0])

#save the weigts used for updating
model.save_weights(ModelsPath+'Final_weights_'+WhichDataSet+'_OneBitPerWeight_model_sReLU.h5')

#plot loss and accuracy
#plot_history(model.history.history)
plot_history(history,epochs)

#plot learning rate schedule
plt.figure(figsize=(16,4))
plt.plot(np.arange(0,len(lr_scheduler.lr_used))/steps_per_epoch,lr_scheduler.lr_used)
plt.xlabel('epoch number')
plt.ylabel('learning rate')
plt.show()










