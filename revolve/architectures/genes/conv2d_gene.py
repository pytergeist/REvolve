import tensorflow as tf
from revolve.architectures.base import BaseGene


class Conv2DGene(BaseGene):
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
