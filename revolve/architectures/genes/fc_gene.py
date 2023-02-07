import tensorflow as tf

from revolve.architectures.base import BaseGene


class FCGene(BaseGene):
    def __init__(
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
        assert isinstance(
            self.parameters["hidden_neurons"], int
        ), f"invalid number of logits: {self.hidden_neurons}"
        assert hasattr(
            tf.keras.activations, self.activation
        ), f"unknown activation function: {self.activation}"
        assert isinstance(
            self.parameters["dropout"], float
        ), f"Invalid value for dropout: {self.dropout}"
        assert (
            0 < self.parameters["dropout"] < 1.0
        ), f"Invalid value for dropout: {self.dropout}"
        assert isinstance(self.l1, float) and self.l1 >= 0, f"Invalid L1: {self.l1}"
        assert isinstance(self.l2, float) and self.l2 >= 0, f"Invalid L2: {self.l2}"
