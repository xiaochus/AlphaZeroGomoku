# -*- coding: utf-8 -*-
"""Deep Reinforcement Learning Model.
A deep reinforcement learning model for Gomoku game based on
policy network and value network.
"""


from keras.models import Model
from keras.layers import Input, Conv2D, Dense, BatchNormalization
from keras.layers import Flatten, add
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import SGD


class PolicyValueNet(object):
    def __init__(self, shape, k, filters, kernels):
        """MobileNetv2
        This function defines a init parameters of architectures.

        # Arguments
            shape: An integer or tuple/list of 3 integers, shape
              of input tensor.
            k: Integer, number of residual block.
            filters: List, number of filters.
            kernels: tuple, size of kernels.
        """
        self.dims = shape
        self.k = k
        self.filters = filters
        self.kernels = kernels

    def _conv2d_unit(self, x, filters, kernels, strides=(1, 1)):
        """Convolution Unit
        This function defines a 2D convolution operation with BN and LeakyReLU.

        # Arguments
            x: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernels: An integer or tuple/list of 2 integers, specifying the
              width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
              specifying the strides of the convolution along the width and
              height. Can be a single integer to specify the same value for
              all spatial dimensions.

        # Returns
            Output tensor.
        """
        x = Conv2D(filters, kernels,
                   padding='same',
                   strides=strides,
                   activation='linear',
                   kernel_regularizer=l2(5e-4),
                   data_format="channels_first")(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def _residual_block(self, inputs, filters, kernels, strides=(1, 1)):
        """Residual Block
        This function defines a 2D convolution operation with BN and LeakyReLU.

        # Arguments
            x: Tensor, input tensor of residual block.
            filters: Integer, the dimensionality of the output space.
            kernels: An integer or tuple/list of 2 integers, specifying the
              width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
              specifying the strides of the convolution along the width and
              height. Can be a single integer to specify the same value for
              all spatial dimensions.

        # Returns
            Output tensor.
        """
        x = self._conv2d_unit(inputs, 2 * filters, kernels, strides)
        x = Conv2D(filters, kernels,
                   padding='same',
                   strides=strides,
                   activation='linear',
                   kernel_regularizer=l2(5e-4),
                   data_format="channels_first")(x)
        x = BatchNormalization(axis=1)(x)
        x = add([inputs, x])
        x = LeakyReLU()(x)

        return x

    def _value_output(self, x):
        """Value Network
        Value Network at the end of network.

        # Arguments
            x: Tensor, input tensor of value output layer.
        # Returns
            Output tensor.
        """
        x = self._conv2d_unit(x, 1, (1, 1), (1, 1))
        x = Flatten()(x)
        x = Dense(20, activation='linear', kernel_regularizer=l2(5e-4))(x)
        x = LeakyReLU()(x)
        x = Dense(1, activation='tanh', kernel_regularizer=l2(5e-4),
                  name='value_output')(x)

        return x

    def _policy_output(self, x):
        """Policy Network
        Policy Network at the end of network.

        # Arguments
            x: Tensor, input tensor of policy output layer.
        # Returns
            Output tensor.
        """
        out_dims = self.dims[1] * self.dims[2]
        x = self._conv2d_unit(x, 2, (1, 1), (1, 1))
        x = Flatten()(x)
        x = Dense(out_dims, activation='softmax', kernel_regularizer=l2(5e-4),
                  name='policy_output')(x)

        return x

    def get_model(self):
        """Get PolicyValueNet
        This function defines a PolicyValueNet architectures.

        # Returns
            PolicyValueNet model.
        """
        inputs = Input(shape=self.dims, name='inputs')
        x = self._conv2d_unit(inputs, self.filters, self.kernels)

        for i in range(self.k):
            x = self._residual_block(x, self.filters, self.kernels)

        value_output = self._value_output(x)
        policy_output = self._policy_output(x)

        model = Model(inputs=[inputs], outputs=[value_output, policy_output])
        model.compile(loss={'value_output': 'mse',
                            'policy_output': 'categorical_crossentropy'},
                      optimizer=SGD(),
                      loss_weights={'value_output': 0.5, 'policy_output': 0.5})

        return model
