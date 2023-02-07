from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_addons as tfa


class BaseChromosome(ABC):
    """
    Abstract class for chromosomes
        implements abstract methods for:
            - decode
        implements methods for:
            - get_unique_key -> str
            -  get_parameter -> Any
    """

    @staticmethod
    def get_unique_key(genes: list) -> str:
        key = []
        for gene in genes:
            key += gene.get_attributes()
        return "".join(map(str, key))

    @staticmethod
    def get_parameter(param: str, default_param: Any, genes: list) -> Any:
        if any([len(gene.parameters) == 1 for gene in genes]):
            parameter = [
                getattr(gene, param) for gene in genes if hasattr(gene, param)
            ][0]
        else:
            parameter = None

        if parameter:
            return parameter
        else:
            return default_param

    @abstractmethod
    def decode(self, learnable_parameters: dict) -> tf.keras.Model:  # pragma: no cover
        pass

    def build_and_compile_model(self, learnable_parameters, genes: list):
        optimizer = self.get_parameter(
            "optimizer", learnable_parameters.get("optimizer"), genes
        )
        learning_rate = self.get_parameter(
            "learning_rate", learnable_parameters.get("learning_rate"), genes
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.learning_rate = learning_rate

        model = self.decode(learnable_parameters)

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[tfa.metrics.RSquare()],
        )

        return model

    @staticmethod
    def fit_model(model, train_data, valid_data, epochs, batch_size, callback):
        model.fit(
            train_data.batch(batch_size),
            epochs=epochs,
            validation_data=valid_data.batch(batch_size),
            callbacks=[callback],
            verbose=0,
        )

    @staticmethod
    def evaluate_model(model, test_data, batch_size):
        return model.evaluate(test_data.batch(batch_size), verbose=0)

    def get_fitness(
        self,
        learnable_parameters: dict,
        genes: list,
        data: Tuple[tf.data.Dataset],
        epochs,
        callback: object,
    ):
        train_data, valid_data, test_data = data
        model = self.build_and_compile_model(learnable_parameters, genes)

        batch_size = self.get_parameter(
            "batch_size", learnable_parameters.get("batch_size"), genes
        )

        self.fit_model(model, train_data, valid_data, epochs, batch_size, callback)

        loss, metric = self.evaluate_model(model, test_data, batch_size)

        return model, loss, metric
