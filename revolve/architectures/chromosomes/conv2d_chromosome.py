"""
File containing Conv2DChromosome class:
    Conv2DChromosome represents the architecture of a network, including fully connected and
    2d convolution layers, and parameter genes, and the loss and metric values of the chromosome.
"""

from typing import Optional, Dict, Union
import tensorflow as tf
from revolve.architectures.base import BaseChromosome


class Conv2DChromosome(BaseChromosome):
    """
    Subclass of BaseChromosome for storing and assesing 2D-convolution networks

    Attributes:
    genes (BaseGene): a list of gene objects containing paramaters for conv2d/fc/parameter-genes
    loss: chosen loss from chromosome
    metric: chosen metric for chromosome

    methods:
        decode(learnable_parameters: dict) - method to decode 2D convolution architecture and
        return keras model

    """

    def __init__(
        self,
        genes: list,
        loss: Optional[float] = None,
        metric: Optional[float] = None,
    ):
        """
        Initialize a Conv2DChromosome object.

        Attributes:
        - genes: list of gene objects
        - loss: a float representing the loss (default None)
        - metric: a float representing the metric (default None)

        Returns:
        None
        """
        self.genes = genes
        self.loss = loss
        self.metric = metric

    def decode(
        self, learnable_parameters: Dict[str, Union[str, float, int]]
    ) -> tf.keras.Model:
        """
        Decode the genes into a Keras model.

        Arguments:
        - learnable_parameters: dictionary containing parameters for model creation

        Returns:
        - Keras model
        """

        _inputs = tf.keras.Input(shape=learnable_parameters.get("input_shape"))

        x_conv = tf.keras.layers.Conv2D(
            filters=self.genes[0].filters,
            kernel_size=self.genes[0].kernel_size,
            strides=self.genes[0].stride,
            activation=self.genes[0].activation,
            padding="same",
        )(_inputs)
        x_conv = tf.keras.layers.BatchNormalization()(x_conv)
        if self.genes[1].gene_type == "conv2d" and self.genes[1].filters != 0:
            x_conv = tf.keras.layers.MaxPool2D(
                strides=self.genes[0].stride, padding="same"
            )(x_conv)
        else:
            x_conv = tf.keras.layers.Flatten()(x_conv)

        for idx, gene in enumerate(self.genes[1:]):
            if gene.gene_type == "conv2d" and gene.filters != 0:
                if self.genes[idx + 2].gene_type != "fc":
                    x_conv = tf.keras.layers.Conv2D(
                        filters=gene.filters,
                        kernel_size=gene.kernel_size,
                        strides=gene.stride,
                        activation=gene.activation,
                        padding="same",
                    )(x_conv)
                    x_conv = tf.keras.layers.BatchNormalization()(x_conv)
                    x_conv = tf.keras.layers.MaxPool2D(
                        strides=gene.stride, padding="same"
                    )(x_conv)
                else:
                    x_conv = tf.keras.layers.Conv2D(
                        filters=gene.filters,
                        kernel_size=gene.kernel_size,
                        strides=gene.stride,
                        activation=gene.activation,
                        padding="same",
                    )(x_conv)
                    x_conv = tf.keras.layers.BatchNormalization()(x_conv)
                    x_conv = tf.keras.layers.Flatten()(x_conv)

            if gene.gene_type == "fc" and gene.hidden_neurons != 0:
                x_conv = tf.keras.layers.Dense(
                    gene.hidden_neurons,
                    activation=gene.activation,
                    kernel_regularizer=tf.keras.regularizers.L1L2(
                        l1=gene.l1, l2=gene.l2
                    ),
                )(x_conv)
                x_conv = tf.keras.layers.Dropout(gene.dropout)(x_conv)

        output = tf.keras.layers.Dense(
            learnable_parameters.get("regression_target"),
            activation=learnable_parameters.get("regression_activation"),
        )(x_conv)
        return tf.keras.Model(inputs=_inputs, outputs=output)
