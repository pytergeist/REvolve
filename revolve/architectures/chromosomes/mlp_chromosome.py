"""
File containing MLPDChromosome class:
    MLPChromosome represents the architecture of a network, including fully connected and
    parameter genes, and the loss and metric values of the chromosome.
"""

from typing import Optional, Dict, Union
import tensorflow as tf
from revolve.architectures.base import BaseChromosome


class MLPChromosome(BaseChromosome):
    """
    Subclass of BaseChromosome for storing and assesing multilayer perceptrons

    Attributes:
    genes (BaseGene): a list of gene objects containing paramaters for conv2d/fc/parameter-genes
    loss: chosen loss from chromosome
    metric: chosen metric for chromosome

    methods:
        decode(learnable_parameters: dict) - method to decode MLP architecture and
        return keras model

    """

    def __init__(
        self,
        genes: list,
        loss: Optional[float] = None,
        metric: Optional[float] = None,
    ):
        """
        Initialize a MLPChromosome with genes, loss and metric values.


        Attributes:
        genes (list): List of genes.
        loss (float, optional): Loss value. Defaults to None.
        metric (float, optional): Metric value. Defaults to None.

        """
        self.genes = genes
        self.loss = loss
        self.metric = metric

    def decode(
        self, learnable_parameters: Dict[str, Union[str, float, int]]
    ) -> tf.keras.Model:
        """
        Decode the chromosome into a TensorFlow model.

        Args:
        learnable_parameters (dict): Learnable parameters for the model.

        Returns:
        tf.keras.Model: The decoded TensorFlow model.

        """

        _inputs = tf.keras.Input(shape=learnable_parameters.get("input_shape"))

        x_mlp = tf.keras.layers.Dense(
            self.genes[0].hidden_neurons,
            activation=self.genes[0].activation,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.genes[0].l1, l2=self.genes[0].l2
            ),
        )(_inputs)

        x_mlp = tf.keras.layers.Dropout(self.genes[0].dropout)(x_mlp)

        for gene in self.genes[1:]:
            if gene.gene_type == "fc" and gene.hidden_neurons != 0:
                x_mlp = tf.keras.layers.Dense(
                    gene.hidden_neurons,
                    activation=gene.activation,
                    kernel_regularizer=tf.keras.regularizers.L1L2(
                        l1=gene.l1, l2=gene.l2
                    ),
                )(x_mlp)
                x_mlp = tf.keras.layers.Dropout(gene.dropout)(x_mlp)

        output = tf.keras.layers.Dense(
            learnable_parameters.get("regression_target"),
            activation=learnable_parameters.get("regression_activation"),
        )(x_mlp)

        return tf.keras.Model(inputs=_inputs, outputs=output)
