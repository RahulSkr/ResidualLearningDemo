#Importing the necessary libraries
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Add, Concatenate, ELU, ReLU
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD

def residual_unit(layer, f_in, f_out, kernel_size, strides=(1,1), use_shortcut=False):
    '''
    This is a simple implementation of the residual unit as proposed by Kaiming He et al.,
    In the article titled: "Deep Residual Learning for Image Recognition" in 2015.

    The following implementation differs from the official ResNet implementation by TensorFlow
    This implementation of the residual unit is not being used to build the ResNet model.

    Here:
    _input: This represents the layer from which the feature maps are fed to the residual unit
    f_in  : This signifies the number of feature maps received from previous layer(s)
    f_out : This signifies the number of feature maps obtained as output from the residual unit
    kernel_size : The kernel size being used in the convolution layers
    strides     : The number of units skiped by the kernel in each direction
    use_shortcut: Flag variable which signifies whether or not a convolution layer is needed 
                    in the identity branch to match the number of input and output feature maps
    '''
    shortcut = layer

    if use_shortcut==True or strides!=(1,1):
        shortcut = Conv2D(f_out, kernel_size = kernel_size, strides=(1,1), padding = "same",
                            kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
                            bias_initializer = "zeros")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    layer = Conv2D(f_in, kernel_size = kernel_size, strides=(1,1), padding = "same",
                    kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
                    bias_initializer = "zeros")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(f_in, kernel_size = kernel_size, strides=(1,1), padding = "same",
                    kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
                    bias_initializer = "zeros")(layer)
    layer = BatchNormalization()(layer)

    return layer
    
def build_model(inputs, kernel_size, n_classes):
    '''
    This function is used to stack the residual units on top of one another to build the
    network

    We assume kernel size to be the same for all layers.

    inputs: represents the batch of input images
    kernel_size: represents the kernel dimensions which remain unchanged throughout the network
    n_classes: number of classes in the classification task
    '''
    x = Conv2D(4, kernel_size = kernel_size, strides=(1,1), padding = "same",
               kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
               bias_initializer = "zeros")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x) 

    x = residual_unit(x, 8, 8, kernel_size)
    x = MaxPooling2D()(x)

    x = residual_unit(x, 8, 8, kernel_size)
    x_shape = K.int_shape(x)
    x = AveragePooling2D(pool_size=(x_shape[1], x_shape[2]))(x)

    x = Flatten()(x)
    x = Dense(n_classes, activation = "softmax")(x)

    return x
	
in_ = Input((128,128,3))
out_layer = build_model(in_, (3,3), 10)
network = Model(inputs = in_, outputs = out_layer)
network.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
network.summary()