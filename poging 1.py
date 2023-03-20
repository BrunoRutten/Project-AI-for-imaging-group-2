# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:04:52 2023

@author: 20201796
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

IMAGE_SIZE = 224
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

def get_data_generators(base_dir, train_batch_size=TRAIN_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE):

    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')

    # instantiate data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=train_batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        valid_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=val_batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator

def get_mobilenetv2_model(activation='relu', optimizer=Adam(learning_rate=0.0001)):

    # load the MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

        # add new layers on top of the base model
        model = Sequential([
            base_model,
            Flatten(),
            Dense(128, activation=activation),
            Dense(1, activation='sigmoid')
        ])

    # compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

activation_functions = ['relu', 'elu', 'selu', 'tanh', 'sigmoid', 'linear']

for activation in activation_functions:

    # get the model with the current activation function
    model = get_mobilenetv2_model(activation=activation)

    # modify the model name to indicate the activation function being used
    model_name = f"my_first_cnn_model_{activation}"

    # get the data generators
    train_gen, val_gen = get_data_generators("C:/Users/20201796/Documents/TUe/Year 3 BMT/Q3/8P361 - Project AI for medical image analysis/8p361-project-imaging-master")

    # save the model and weights
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

    history = model.fit(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=3,
                        callbacks=callbacks_list)
