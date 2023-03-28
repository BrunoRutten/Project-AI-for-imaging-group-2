import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model

IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')
     return train_gen, val_gen


class MobileNetBlock(Layer):
    def __init__(self, filters, strides):
        super(MobileNetBlock, self).__init__(name='MobileNetBlock')
        self.filters = filters
        self.strides = strides

    def call(self,inputs):
        x = DepthwiseConv2D(kernel_size = 3, strides = self.strides, padding = 'same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters = self.filters, kernel_size = 1, strides = 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

def get_model():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same',input_shape=(96,96,3)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MobileNetBlock(filters = 64, strides = 1),name = "1")
    model.add(MobileNetBlock(filters = 128, strides = 2), name = "2")
    model.add(MobileNetBlock(filters = 128, strides = 1), name = "3")
    model.add(MobileNetBlock(filters = 256, strides = 2), name = "4")
    model.add(MobileNetBlock(filters = 256, strides = 1), name = "5")
    model.add(MobileNetBlock(filters = 512, strides = 2), name = "6")

    model.add(MobileNetBlock(filters = 512, strides = 1), name = "7")
    model.add(MobileNetBlock(filters = 512, strides = 1), name = "8")
    model.add(MobileNetBlock(filters = 512, strides = 1), name = "9")
    model.add(MobileNetBlock(filters = 512, strides = 1), name = "10")
    model.add(MobileNetBlock(filters = 512, strides = 1), name = "11")

    model.add(MobileNetBlock(filters = 1024, strides = 2), name = "12")
    model.add(MobileNetBlock(filters = 1024, strides = 1), name = "13")

    # model.add(AvgPool2D(pool_size = 7, strides = 1))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
    return model




model = get_model()

# get the data generators
train_gen, val_gen = get_pcam_generators('C:/Users/20202181/OneDrive - TU Eindhoven/Desktop/Project imaging')

model_name = 'Nettest_2'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=100,
                    validation_data=val_gen,
                    validation_steps=100,
                    epochs=3,
                    callbacks=callbacks_list)