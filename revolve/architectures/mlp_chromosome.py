from typing import Optional
import tensorflow as tf
from revolve.architectures.base import Chromosome


class MLPChromosome(Chromosome):
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

    def decode(self, learnable_params) -> tf.keras.Model:
        """
        decode encoded neural network architecture and return model
        :return: sequential keras model
        """

        _inputs = tf.keras.Input(shape=learnable_params.get("input_shape"))

        x = tf.keras.layers.Dense(
            self.genes[0].parameters["hidden_neurons"],
            activation=self.genes[0].parameters["activation"],
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.genes[0].parameters["l1"], l2=self.genes[0].l2
            ),
        )(_inputs)

        x = tf.keras.layers.Dropout(self.genes[0].dropout)(x)

        for idx, gene in enumerate(self.genes[1:]):
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
            learnable_params.get("regression_target"),
            activation=learnable_params.get("regression_activation"),
        )(x)

        return tf.keras.Model(inputs=_inputs, outputs=output)
