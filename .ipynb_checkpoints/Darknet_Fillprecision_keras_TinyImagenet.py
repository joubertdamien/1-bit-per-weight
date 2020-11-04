#!/usr/bin/env python
# coding: utf-8

# # Tensorflow.keras implementation of full precision CNN for CIFAR 100 
# ##  https://arxiv.org/abs/1802.08530
# ## M. D. McDonnell, 
# ## Training wide residual networks for deployment using a single bit for each weight
# ## ICLR, 2018

#In1
# select a GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
from scipy.io import savemat,loadmat
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow
print('Tensorflow version = ',tensorflow.__version__)
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, History
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image

from PIL import Image
from sklearn.utils import shuffle
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from darknet import darknet19
from Utils import cutout,LR_WarmRestart,GetDataGen,plot_history
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime

#Save model logs

NAME = "Darknet_FullprecisionWeight_TinyImagenet{}".format(datetime.now().strftime("%Y%m%d"))
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(NAME),
                            histogram_freq=1,
                            write_graph=True,
                            write_images=False,
                            update_freq="epoch",
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None)


#In2
#params

WhichDataSet = 'TINY-IMAGENET'
batch_size=300


#In3
#load and prepare data
img_width, img_height = 64,64
train_data_dir = 'Datasets/Tiny-Imagenet-200/train/images'
validation_data_dir = 'Datasets/Tiny-Imagenet-200/train/images'



#datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2)
datagen = ImageDataGenerator(rescale=1./255, 
                            rotation_range=20,
                            zoom_range=0.15,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            fill_mode="nearest",
                            validation_split = 0.1)
val_datagen = ImageDataGenerator(rescale=1./255)
test_dataset = ImageDataGenerator() #resacle?


train_generator = datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),color_mode='rgb',
                                    batch_size= batch_size, subset= "training",class_mode='categorical', shuffle=True, seed=42)
val_generator = datagen.flow_from_directory(validation_data_dir,target_size=(64,64),color_mode='rgb',
                                    batch_size= batch_size, subset = "validation",class_mode='categorical', shuffle=False, seed=42)

test_generator = test_dataset.flow_from_directory('Datasets/Tiny-Imagenet-200/test/', target_size=(64,64),color_mode='rgb',
                                    batch_size= 1, class_mode=None, shuffle=False, seed=42)



x_train, y_train = next(train_generator)
x_test, y_test = next(val_generator)
num_classes = np.unique(y_train).shape[0]
y_true = val_generator.classes
input_shape = x_train.shape[1:]

# Learning parameters 
init_lr = 0.1
epochs = 160 #300
My_wd=5e-4
UseCutout=True
Loss = 'categorical_crossentropy'
steps_per_epoch = int(np.floor(train_generator.n  // batch_size))
lr_fn = PolynomialDecay(initial_learning_rate=0.1, 
                        end_learning_rate=0.0000001,
                        decay_steps=int(steps_per_epoch*epochs), 
                        power=4.0)
Optimizer = SGD(learning_rate=lr_fn,
                momentum=0.9, 
                nesterov=False)
Metrics = ['accuracy']
ModelsPath = './TrainedModels/Tensorflow.keras/'


def catcross_entropy_logits_loss():
    def loss(y_true, y_pred):
        return tensorflow.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    return loss


#fdefine a datagen or generating training samples with flip and pad/crop augmentation, and if set to True, with cutout augmentation
dataGenerator = GetDataGen(UseCutout)

#define and compile the model
model = darknet19(input_shape=input_shape,num_classes=200)
model.compile(loss=Loss ,optimizer = Optimizer, metrics = Metrics)

#print  the model
model.summary()



#define callbacks
history = History()
callbacks = [history,tensorboard]
#callbacks = [lr_scheduler,history]


#In5
history = model.fit_generator(train_generator,
                              validation_data=val_generator,
                              epochs=epochs,
                              verbose=2,
                              callbacks=callbacks,
                              steps_per_epoch =400)

                             


#In6
#get final performance
y_pred = model.predict(x_test)
#print(y_pred)
print('Test accuracy (%):', 100*sum(np.argmax(y_pred,-1)==np.argmax(y_test,-1))/y_test.shape[0])


#plot loss and accuracy
#plot_history(model.history.history)
plot_history(history,epochs)

#plot learning rate schedule
plt.figure(figsize=(16,4))
plt.plot(np.arange(0,len(lr_scheduler.lr_used))/steps_per_epoch,lr_scheduler.lr_used)
plt.xlabel('epoch number')
plt.ylabel('learning rate')
plt.show()


#In8
#save the weigts used for updating
model.save_weights(ModelsPath+'Final_weights_'+WhichDataSet+'single-bit-per-weight_tinyImagNet_darknet.h5')




