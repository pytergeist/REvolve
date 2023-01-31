from .genes import FCGene, ParameterGene
from typing import List, Union, Optional
import tensorflow as tf
import numpy.typing as npt
import numpy as np
import tensorflow_addons as tfa


class MLPChromosome:
    """
    Class to store MLP layers and param genes as chromosome list
    Methods:
        decode:
            decodes chromosome into keras model
            return - tf.keras.Model
    """

    def __init__(self,
                 genes: List[Union[FCGene, ParameterGene]],
                 loss: Optional[float] = None,
                 metric: Optional[float] = None,
                 ):
        self.genes = genes
        self.loss = loss
        self.metric = metric

    def get_unique_key(self):
        key = []
        for gene in self.genes:
            key += gene.get_attributes()
        return "".join(map(str, key))

    def get_parameter(self, param: str, default_param):
        param_list = list(map(lambda x: x.parameter if x.gene_type == param
                              else None, self.genes
                              )
                          )
        param_idx = np.where(np.array(param_list) != None)[0]

        if len(param_idx) != 0:
            return param_list[param_idx[0]]
        else:
            return default_param

    def decode(self,
               input_shape: int, regression_target: int,
               regression_activation: str) -> tf.keras.Sequential:
        """
        decode encoded neural network architecture and return model
        :return: sequential keras model
        """
        _inputs = tf.keras.Input(shape=(input_shape,))

        x = tf.keras.layers.Dense(
            self.genes[0].neurons,
            activation=self.genes[0].activation,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.genes[0].l1, l2=self.genes[0].l2
                        )
                )(_inputs)

        x = tf.keras.layers.Dropout(self.genes[0].dropout)(x)

        for idx, gene in enumerate(self.genes[1:]):
            if gene.gene_type == 'fc' and gene.neurons != 0:
                x = tf.keras.layers.Dense(
                      gene.neurons,
                      activation=gene.activation,
                      kernel_regularizer=tf.keras.regularizers.L1L2(
                          l1=gene.l1, l2=gene.l2
                      )
                  )(x)
                x = tf.keras.layers.Dropout(gene.dropout)(x)

        output = tf.keras.layers.Dense(regression_target, activation=regression_activation)(x)
        return tf.keras.Model(inputs=_inputs, outputs=output)


class Conv2DChromosome:
    """
    Class to store MLP layers and param genes as chromosome list
    Methods:
        decode:
            decodes chromosome into keras model
            return - tf.keras.Model
    """

    def __init__(self, genes: List[Union[FCGene, ParameterGene]]):
        self.genes = genes

    def get_unique_key(self):
        key = []
        for gene in self.genes:
            key += gene.get_attributes()
        return "".join(map(str, key))

    def get_parameter(self, param: str, default_param):
        param_list = list(map(lambda x: x.parameter if x.gene_type == param
                              else None, self.genes
                              )
                          )
        param_idx = np.where(np.array(param_list) != None)[0]

        if len(param_idx) != 0:
            return param_list[param_idx[0]]
        else:
            return default_param

    def decode(self,
               input_shape: int, regression_target: int,
               regression_activation: str) -> tf.keras.Sequential:
        """
        decode encoded neural network architecture and return model
        :return: sequential keras model
        """
        _inputs = tf.keras.Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(
            filters=self.genes[0].filters,
            kernel_size=self.genes[0].kernel_size,
            strides=self.genes[0].stride,
            activation=self.genes[0].activation,
            padding="same",
                        )(_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.genes[1].gene_type == 'conv2d':
            x = tf.keras.layers.MaxPool2D(strides=self.genes[0].stride, padding="same")(x)
        else:
            x = tf.keras.layers.Flatten()(x)

        for idx, gene in enumerate(self.genes[1:]):

            if gene.gene_type == 'conv2d':
                if self.genes[idx + 2].gene_type != 'fc':
                    x = tf.keras.layers.Conv2D(
                        filters=gene.filters,
                        kernel_size=gene.kernel_size,
                        strides=gene.stride,
                        activation=gene.activation,
                        padding="same",
                    )(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.MaxPool2D(strides=gene.stride, padding="same")(x)
                else:
                    x = tf.keras.layers.Conv2D(
                        filters=gene.filters,
                        kernel_size=gene.kernel_size,
                        strides=gene.stride,
                        activation=gene.activation,
                        padding="same",
                    )(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Flatten()(x)

            if gene.gene_type == 'fc' and gene.neurons != 0:
                x = tf.keras.layers.Dense(
                      gene.neurons,
                      activation=gene.activation,
                      kernel_regularizer=tf.keras.regularizers.L1L2(
                          l1=gene.l1, l2=gene.l2
                      )
                  )(x)
                x = tf.keras.layers.Dropout(gene.dropout)(x)

        output = tf.keras.layers.Dense(regression_target, activation=regression_activation)(x)
        return tf.keras.Model(inputs=_inputs, outputs=output)
