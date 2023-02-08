"""
File containing Conv2DGene class which inherits from the BaseGene class:
    Conv2D represent a conv2d layer in the network architecture
"""

import tensorflow as tf
from revolve.architectures.base import BaseGene


class Conv2DGene(BaseGene):
    """
    Conv2DGene is a subclass of BaseGene which represents a convolutional layer in a CNN model.

    Args
        filters: (int) The number of filters to use.
        kernel_size: (int) The size of the kernel to use.
        stride: (int) The size of the stride to use.
        activation: (str) The activation function to use.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int,
        activation: str,
    ):
        super().__init__(
            gene_type="conv2d",
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
        )

        self.__dict__.update(**self.parameters)

        self._validate_params()

    def _validate_params(self):
        """
        Validate the parameters of the Conv2DGene.

        :raises: (assertion error) If the parameters are invalid.
        """
        assert self.gene_type == "conv2d"
        assert isinstance(
            self.parameters["filters"], int
        ), f"invalid num filter: {self.parameters['filters']}, enter as integer"
        assert isinstance(
            self.parameters["kernel_size"], int
        ), f"invalid kernel size: {self.parameters['kernel_size']}, enter as integer"
        assert isinstance(
            self.parameters["stride"], int
        ), f"invalid stride: {self.parameters['stride']}, enter as integer"
        assert hasattr(
            tf.keras.activations, self.parameters["activation"]
        ), f"unknown activation function: {self.parameters['activation']}"
