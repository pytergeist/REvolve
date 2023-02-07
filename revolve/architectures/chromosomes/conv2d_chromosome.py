from typing import Optional
import tensorflow as tf
from revolve.architectures.base import BaseChromosome


class Conv2DChromosome(BaseChromosome):
    """
    Class to store MLP layers and param genes as chromosome list
    Methods:
        decode:
            decodes chromosome into keras model
            return - tf.keras.Model
    """

    def __init__(
        self,
        genes: list,
        loss: Optional[float] = None,
        metric: Optional[float] = None,
    ):
        self.genes = genes
        self.loss = loss
        self.metric = metric

    def decode(self, learnable_parameters: dict) -> tf.keras.Sequential:
        """
        decode encoded neural network architecture and return model
        :return: sequential keras model
        """
        _inputs = tf.keras.Input(shape=learnable_parameters.get("input_shape"))

        x = tf.keras.layers.Conv2D(
            filters=self.genes[0].filters,
            kernel_size=self.genes[0].kernel_size,
            strides=self.genes[0].stride,
            activation=self.genes[0].activation,
            padding="same",
        )(_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.genes[1].gene_type == "conv2d" and self.genes[1].filters != 0:
            x = tf.keras.layers.MaxPool2D(strides=self.genes[0].stride, padding="same")(
                x
            )
        else:
            x = tf.keras.layers.Flatten()(x)

        for idx, gene in enumerate(self.genes[1:]):
            if gene.gene_type == "conv2d" and gene.filters != 0:
                if self.genes[idx + 2].gene_type != "fc":
                    x = tf.keras.layers.Conv2D(
                        filters=gene.filters,
                        kernel_size=gene.kernel_size,
                        strides=gene.stride,
                        activation=gene.activation,
                        padding="same",
                    )(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.MaxPool2D(strides=gene.stride, padding="same")(
                        x
                    )
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

            if gene.gene_type == "fc" and gene.hidden_neurons != 0:
                x = tf.keras.layers.Dense(
                    gene.hidden_neurons,
                    activation=gene.activation,
                    kernel_regularizer=tf.keras.regularizers.L1L2(
                        l1=gene.l1, l2=gene.l2
                    ),
                )(x)
                x = tf.keras.layers.Dropout(gene.dropout)(x)

        output = tf.keras.layers.Dense(
            learnable_parameters.get("regression_target"),
            activation=learnable_parameters.get("regression_activation"),
        )(x)
        return tf.keras.Model(inputs=_inputs, outputs=output)
