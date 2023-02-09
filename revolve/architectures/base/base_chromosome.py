"""
File containing BaseChromosome class with abstract and defined method:
    each chromosome inherits BaseChromosome and every child chromosome
    represents the architecture of a network, including layers and parameter
    genes
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from revolve.architectures.genes import FCGene, Conv2DGene, ParameterGene

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()


class BaseChromosome(ABC):
    """
    The base class for defining a chromosome in an evolutionary algorithm.

    Arguments:
    - None
    """

    @staticmethod
    def get_gene_attributes(genes: List[Union[FCGene, Conv2DGene, ParameterGene]]):
        param_dict = {}
        for idx, gene in enumerate(genes):
            param_dict[f"{gene.gene_type}_{idx}"] = gene.get_attribute_dict
        return param_dict

    @staticmethod
    def get_unique_key(genes: list) -> str:
        """
        Get a unique key for the given genes.

        Arguments:
        - genes: list of genes to generate the unique key from

        Returns:
        - unique_key: unique key for the given genes
        """
        key = []
        for gene in genes:
            key += gene.get_attributes()
        return "".join(map(str, key))

    @staticmethod
    def get_parameter(param: str, default_param: Any, genes: list) -> Any:
        """
        Get a parameter from the given genes or return the default value if not found.

        Arguments:
        - param: parameter to extract from the genes
        - default_param: default value for the parameter if not found in the genes
        - genes: list of genes to extract the parameter from

        Returns:
        - parameter: extracted parameter or default value
        """
        if any(len(gene.parameters) == 1 for gene in genes):
            parameter = [
                getattr(gene, param) for gene in genes if hasattr(gene, param)
            ][0]

        else:
            parameter = None

        if parameter:
            return parameter

        return default_param

    @abstractmethod
    def decode(self, learnable_parameters: dict) -> tf.keras.Model:  # pragma: no cover
        """
        Decode the genes into a model.

        Arguments:
        - learnable_parameters: dictionary of learnable parameters

        Returns:
        - model: decoded model
        """

    def build_and_compile_model(self, learnable_parameters, loss, metric, genes: list):
        """
        Build and compile a model from the given learnable parameters and genes.

        Arguments:
        - learnable_parameters: dictionary of learnable parameters
        - genes: list of genes used to build the model

        Returns:
        - model: built and compiled model
        """
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
            loss=loss,
            metrics=metric,
        )

        return model

    @staticmethod
    def fit_model(
        model, train_data, valid_data, epochs, batch_size, callback
    ):  # pylint: disable=too-many-arguments
        """
        Fit the model to the training data.

        Arguments:
        - model: model to be fit
        - train_data: training data to fit the model to
        - valid_data: validation data to monitor the training
        - epochs: number of training epochs
        - batch_size: batch size for the training data
        - callback: callbacks to use during training

        Returns:
        - None
        """
        model.fit(
            train_data.batch(batch_size),
            epochs=epochs,
            validation_data=valid_data.batch(batch_size),
            callbacks=[callback],
            verbose=0,
        )

    @staticmethod
    def evaluate_model(model, test_data, batch_size):
        """
        Evaluate the model on the test data.

        Arguments:
        - model: model to be evaluated
        - test_data: test data to evaluate the model on
        - batch_size: batch size for the test data

        Returns:
        - loss: loss value of the model on the test data
        - metric: metric value of the model on the test data
        """
        return model.evaluate(test_data.batch(batch_size), verbose=0)

    def get_fitness(
        self,
        learnable_parameters: dict,
        genes: list,
        data: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
        loss: tf.keras.losses.Loss,
        metric: tf.keras.metrics.Metric,
        epochs,
        callback: object,
    ):  # pylint: disable=too-many-arguments
        """
        Calculate the fitness of a chromosome represented by `genes` given the
        `learnable_parameters`, training and test `data`, `epochs` to train, and `callback`
        for early stopping.

        Parameters
        ----------
        learnable_parameters : dict
            The learnable hyperparameters for the model.
        genes : list
            A list of genes representing a chromosome.
        data : Tuple[tf.data.Dataset]
            A tuple of datasets (train, validation, test)
        loss: tf.keras.losses.Loss
            A tf.keras loss object
        metric: tf.keras.metrics.Metric
            A tf.keras metric object
        epochs : int
            The number of epochs to train the model.
        callback : object
            The early stopping callback object.

        Returns
        -------
        model : tf.keras.Model
            The compiled and trained model.
        loss : float
            The mean squared error loss after evaluating the model on the test data.
        metric : float
            The R-squared metric after evaluating the model on the test data.
        """
        train_data, valid_data, test_data = data
        model = self.build_and_compile_model(learnable_parameters, loss, metric, genes)

        batch_size = self.get_parameter(
            "batch_size", learnable_parameters.get("batch_size"), genes
        )

        self.fit_model(model, train_data, valid_data, epochs, batch_size, callback)

        loss, metric = self.evaluate_model(model, test_data, batch_size)

        return model, loss, metric
