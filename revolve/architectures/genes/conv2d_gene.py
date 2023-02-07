import tensorflow as tf
from revolve.architectures.base import BaseGene


class Conv2DGene(BaseGene):
    """
    Conv2DGene is a sub-class of BaseGene which represents a convolutional layer in a CNN model.

    :param filters: (int) The number of filters to use.
    :param kernel_size: (int) The size of the kernel to use.
    :param stride: (int) The size of the stride to use.
    :param activation: (str) The activation function to use.
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
            self.filters, int
        ), f"invalid num filter: {self.filters}, enter as integer"
        assert isinstance(
            self.kernel_size, int
        ), f"invalid kernel size: {self.kernel_size}, enter as integer"
        assert isinstance(
            self.stride, int
        ), f"invalid stride: {self.stride}, enter as integer"
        assert hasattr(
            tf.keras.activations, self.activation
        ), f"unknown activation function: {self.activation}"
