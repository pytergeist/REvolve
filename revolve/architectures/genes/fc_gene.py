"""
File containing FCGene class which inherits from the BaseGene class:
    FCGene represent a fully connected layer in the network architecture
"""

import tensorflow as tf
from revolve.architectures.base import BaseGene


class FCGene(BaseGene):
    """
    FCGene is a sub-class of BaseGene which represents an FC layer in an MLP/CONV model.

    Args:
        hidden_neurons: (int) The number of hidden_neurons.
        activation: (str) The activation function to use.
        dropout: (float) the dropout probability.
        l1: (float) l1 regularization coefficient
        l2: (float) l2 regularization coefficient
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hidden_neurons: int,
        activation: str,
        dropout: float,
        l1: float,
        l2: float,
    ):
        super().__init__(
            gene_type="fc",
            hidden_neurons=hidden_neurons,
            activation=activation,
            dropout=dropout,
            l1=l1,
            l2=l2,
        )

        self.__dict__.update(**self.parameters)

        self._validate_params()

    def _validate_params(self):
        """
        Internal method to validate the parameters passed to the constructor.
        """
        assert isinstance(
            self.parameters["hidden_neurons"], int
        ), f"invalid number of logits: {self.parameters['hidden_neurons']}"
        assert hasattr(
            tf.keras.activations, self.parameters["activation"]
        ), f"unknown activation function: {self.parameters['activation']}"
        assert isinstance(
            self.parameters["dropout"], float
        ), f"Invalid value for dropout: {self.parameters['dropout']}"
        assert (
            0 < self.parameters["dropout"] < 1.0
        ), f"Invalid value for dropout: {self.parameters['dropout']}"
        assert (
            isinstance(self.parameters["l1"], float) and self.parameters["l1"] >= 0
        ), f"Invalid L1: {self.parameters['l1']}"
        assert (
            isinstance(self.parameters["l2"], float) and self.parameters["l2"] >= 0
        ), f"Invalid L2: {self.parameters['l2']}"
