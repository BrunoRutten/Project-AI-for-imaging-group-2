import tensorflow as tf

def ConvBlock(inputs, filters, kernel_size, strides, alpha=1.0):
    filters = int(filters * alpha)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(max_value=6)(x)

def DepthwiseConvBlock(inputs, pointwise_conv_filters, strides, alpha=1.0):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(max_value=6)(x)

def MobileNetV2(input_shape, num_classes, alpha=1.0):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # First layer: (224, 224, 3) -> (112, 112, 32)
    x = ConvBlock(inputs, 32, 3, strides=(2, 2), alpha=alpha)

    # Depthwise Separable Convolution blocks
    x = DepthwiseConvBlock(x, 64, strides=(1, 1), alpha=alpha)
    x = DepthwiseConvBlock(x, 128, strides=(2, 2), alpha=alpha)
    x = DepthwiseConvBlock(x, 128, strides=(1, 1), alpha=alpha)
    x = DepthwiseConvBlock(x, 256, strides=(2, 2), alpha=alpha)
    x = DepthwiseConvBlock(x, 256, strides=(1, 1), alpha=alpha)
    x = DepthwiseConvBlock(x, 512, strides=(2, 2), alpha=alpha)
    for i in range(5):
        x = DepthwiseConvBlock(x, 512, strides=(1, 1), alpha=alpha)

    x = DepthwiseConvBlock(x, 1024, strides=(2, 2), alpha=alpha)
    x = DepthwiseConvBlock(x, 1024, strides=(1, 1), alpha=alpha)

    # Final layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model
