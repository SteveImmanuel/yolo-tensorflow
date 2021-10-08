import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, BatchNormalization, Conv2D, Dropout, ReLU, Reshape
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input
from yolo.v1.config import *


class YoloV1Model(Model):
    def __init__(self, layer_config=BASE_ARCHITECTURE, S=7, C=20, B=2) -> None:
        super(YoloV1Model, self).__init__()
        self.model_layers = []
        self.S = S
        self.C = C
        self.B = B

        for layer in layer_config['CONV_LAYERS']:
            if type(layer) == str:
                self.model_layers.append(MaxPool2D(2, 2))
            elif type(layer) == tuple:
                self._add_conv_layer(layer)
            elif type(layer) == list:
                for _ in range(layer[-1]):
                    for i in range(len(layer) - 1):
                        self._add_conv_layer(layer[i])
        self.model_layers.append(Flatten())
        self.model_layers.append(Dense(layer_config['FC_LAYER'], kernel_regularizer='l1'))
        self.model_layers.append(ReLU(negative_slope=0.1))
        self.model_layers.append(Dropout(0.5))
        self.model_layers.append(Dense(S * S * (C + 5 * B), kernel_regularizer='l1'))
        self.model_layers.append(ReLU())
        self.model_layers.append(Reshape((S, S, (C + 5 * B))))

    def _add_conv_layer(self, layer_config):
        self.model_layers.append(
            Conv2D(layer_config[0], layer_config[1], layer_config[2], padding='same', kernel_regularizer=L2(l2=0.005), use_bias=False)
        )
        self.model_layers.append(BatchNormalization())
        self.model_layers.append(ReLU(negative_slope=0.1))

    def call(self, inputs):
        for layer in self.model_layers:
            inputs = layer(inputs)
        return inputs


class FastYoloV1Model(YoloV1Model):
    def __init__(self, S=7, C=20, B=2) -> None:
        super(FastYoloV1Model, self).__init__(layer_config=FAST_ARCHITECTURE, S=S, C=C, B=B)


if __name__ == '__main__':
    model = FastYoloV1Model()
    model.build(input_shape=(None, 448, 448, 3))
    model.call(Input(shape=(448, 448, 3)))
    model.summary()