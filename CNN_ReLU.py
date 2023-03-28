<<<<<<< HEAD
"""
Project AI for Medical Image Analysis
Group 2
Note: this code is adapted from: 
    https://github.com/Haikoitoh/paper-implementation
Version 2, 28-3-2023
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add,Softmax, AveragePooling2D, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

def expansion_block(x, t: int, filters: int, block_id: int):
    """
    Expansion block in MobileNetV2 network.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    t : int
        Expansion factor.
    filters : int
        Number of output filters in the depthwise convolution layer.
    block_id : int
        Block identifier.

    Returns
    -------
    x : Tensor
        Output tensor after applying the expansion block.
    """
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = Conv2D(total_filters,1,padding='same',use_bias=False, name = prefix +'expand')(x)
    x = BatchNormalization(name=prefix +'expand_bn')(x)
    x = ReLU(6,name = prefix +'expand_relu')(x)
    return x

def depthwise_block(x, stride, block_id):
    """
    Apply depthwise convolution block to the input tensor x.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    stride : int
        Stride of the convolution block.
    block_id : int
        Index of the block.

    Returns
    -------
    Tensor
        Output tensor after applying the depthwise convolution block.
    """
    
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D(3, strides=(stride, stride), padding='same', use_bias=False, name=prefix+'depthwise_conv')(x)
    x = BatchNormalization(name=prefix+'dw_bn')(x)
    x = ReLU(6, name=prefix+'dw_relu')(x)
    return x


def projection_block(x, out_channels, block_id):
    """
    Applies a projection block to the input tensor x.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    out_channels : int
        Number of output channels in the projection block.
    block_id : int
        Block ID for naming layers.

    Returns
    -------
    x : Tensor
        Output tensor after applying the projection block.
    """
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters=out_channels, kernel_size=1, padding='same', use_bias=False, name=prefix + 'compress')(x)
    x = BatchNormalization(name=prefix + 'compress_bn')(x)
    return x


def Bottleneck(x,t,filters, out_channels,stride,block_id):
    """
    Creates a bottleneck block, which is composed of an expansion block, a 
    depthwise block, and a projection block, and a residual connection.
    Parameters
    ----------
    x : tensor
        Input tensor to the bottleneck block.
    t : int
        Integer expansion factor, used to scale the number of output channels 
        in the expansion block.
    filters : int
        Number of filters in the input tensor.
    out_channels : int
        Number of filters in the output tensor.
    stride : int
        Integer stride for the depthwise convolution operation in the 
        depthwise block.
    block_id : int
        Integer block ID used to name layers in the bottleneck block.
    
    Returns
    -------
    y : tensor
        Output tensor of the bottleneck block.
    
    """
    y = expansion_block(x,t,filters,block_id)
    y = depthwise_block(y,stride,block_id)
    y = projection_block(y, out_channels,block_id)
    if y.shape[-1]==x.shape[-1]:
        y = add([x,y])
    return y


def MobileNetV2(input_shape = (224,224,3), n_classes=2):
    """
    MobileNetV2 architecture model for image classification tasks

    Parameters
    ----------
    input_shape : tuple, optional
        Input tuple. The default is (224,224,3).
    n_classes : int, optional
        Defines the number of classes. The default is 2.

    Returns
    -------
    model : none
        Returns the MobileNetV2 model.

    """
    input = Input(input_shape)

    x = Conv2D(32,kernel_size=3,strides=(2,2),padding = 'same', use_bias=False)(input)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6, name = 'conv1_relu')(x)

    # 17 Bottlenecks

    x = depthwise_block(x,stride=1,block_id=1)
    x = projection_block(x, out_channels=16,block_id=1)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)


    #1*1 conv
    x = Conv2D(filters = 1280,kernel_size = 1,padding='same',use_bias=False, name = 'last_conv')(x)
    x = BatchNormalization(name='last_bn')(x)
    x = ReLU(6,name='last_relu')(x)

    #AvgPool 7*7
    x = GlobalAveragePooling2D(name='global_average_pool')(x)

    output = Dense(n_classes,activation='softmax')(x)

    model = Model(input, output)

    return model

def get_pcam_generators(
        base_dir: str, 
        train_batch_size: int = 32, 
        val_batch_size: int = 32
        ):
    """
    Generate image data batches for the PCam dataset using the Keras ImageDataGenerator.

    Parameters
    ----------
    base_dir : str
        The base directory path for the dataset.
    train_batch_size : int, optional
        The batch size for training data. The default is 32.
    val_batch_size : int, optional
        The batch size for validation data. The default is 32.

    Returns
    -------
    train_gen : DirectoryIterator
        A Keras DirectoryIterator for training data.
    val_gen : DirectoryIterator
        A Keras DirectoryIterator for validation data.

    """
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')


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

 
# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)
n_classes = 1

model = MobileNetV2(input_shape,n_classes)
model.summary()

# Set up data generators
train_gen, val_gen = get_pcam_generators("C:/Users/20201796/Documents/TUe/Year 3 BMT/Q3/8P361 - Project AI for medical image analysis/8p361-project-imaging-master")

model.compile(SGD(learning_rate=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

# save the model and weights
model_name = 'CNN_model_ReLU'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model, note that we define "mini-epochs"
train_steps = train_gen.n//train_gen.batch_size//20
val_steps = val_gen.n//val_gen.batch_size//20

# since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# not used during training
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=1,
                    callbacks=callbacks_list)
