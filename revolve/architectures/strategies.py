import random
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from typing import (
    List, Union,
    Any
)
import tensorflow_addons as tfa
from .genes import FCGene, Conv2DGene, ParameterGene
from .chromosomes import MLPChromosome, Conv2DChromosome
from.bases import BaseStrategy


class MLPStrategy(BaseStrategy):

    def __init__(
            self,
            input_shape: int,
            hidden_neurons: List[int],
            activation: Union[List[str], str],
            l1: Union[List[float], float],
            l2: Union[List[float], float],
            max_fc: int,
            dropout: Union[List[float], float],
            batch_size: Union[List[int], int],
            optimizer: Union[List[str], str],
            learning_rate: Union[List[float], float],
            callback: object,
            regression_target: int = 1,
            regression_activation: str = 'relu',

    ):
        self.input_shape = input_shape
        self.neurons = hidden_neurons
        self.activation = activation
        self.l1 = l1
        self.l2 = l2
        self.max_fc = max_fc
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regression_target = regression_target
        self.regression_activation = regression_activation
        self.callback = callback
        self.learnable_parameters = self.get_learnable_parameters()

    @staticmethod
    def create_new_chromosome(genes: List[object]):
        return MLPChromosome(genes=genes)

    def get_learnable_parameters(self):

        handler_params = list(self.__dict__.keys())
        learnable_parameters = dict()

        for param in handler_params:
            if isinstance(getattr(self, param), list):
                learnable_parameters[param] = getattr(self, param)

        return learnable_parameters

    def parameter_choice(self, parameter: Any):

        params = self.learnable_parameters.get(parameter)

        if params is not None:
            return random.choice(params)
        else:
            return getattr(self, parameter)

    def generate_population(self, population_size: int) -> List:

        assert isinstance(population_size, int)

        population: List[MLPChromosome] = []

        training_params = [
            'batch_size',
            'optimizer',
            'learning_rate',
        ]

        key_store: List[str] = []

        while len(population) < population_size:

            genes = [FCGene(
                gene_type='fc',
                neurons=self.parameter_choice('neurons'),
                activation=self.parameter_choice('activation'),
                dropout=self.parameter_choice('dropout'),
                l1=self.parameter_choice('l1'),
                l2=self.parameter_choice('l2'),
            ) for _ in range(self.max_fc)
                    ] + \
                    [
                        ParameterGene(
                            parameter_name=param,
                            parameter=self.parameter_choice(param)
                        )
                        for param in training_params
                    ]
            chromosome = MLPChromosome(
                genes,

            )

            key = chromosome.get_unique_key()

            if key not in key_store:
                population.append(chromosome)
                key_store.append(key)
            else:
                pass

        return population

    def asses(self,
              x_train: npt.NDArray[np.float32],
              y_train: npt.NDArray[np.float32],
              x_test: npt.NDArray[np.float32],
              y_test: npt.NDArray[np.float32],
              chromosome: MLPChromosome,
              epochs: int
              ):

        model = chromosome.decode(self.input_shape, self.regression_target, self.regression_activation)

        optimizer = chromosome.get_parameter('optimizer', self.optimizer)

        learning_rate = chromosome.get_parameter('learning_rate', self.learning_rate)

        batch_size = chromosome.get_parameter('batch_size', self.batch_size)

        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.learning_rate = learning_rate

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[tfa.metrics.RSquare()]
        )

        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_split=0.1,
            batch_size=batch_size,
            callbacks=[self.callback],
            verbose=0
        )

        mse, r_square = model.evaluate(x_test, y_test, verbose=0)

        return model, mse, r_square


class Conv2DStrategy(BaseStrategy):

    def __init__(
            self,
            input_shape: int,
            hidden_neurons: List[int],
            activation: Union[List[str], str],
            l1: Union[List[float], float],
            l2: Union[List[float], float],
            max_fc: int,
            filters: Union[List[float], float],
            kernel_size: Union[List[float], float],
            stride: Union[List[float], float],
            max_conv: int,
            dropout: Union[List[float], float],
            batch_size: Union[List[int], int],
            optimizer: Union[List[str], str],
            learning_rate: Union[List[float], float],
            callback: object,
            regression_target: int = 1,
            regression_activation: int = 'relu',

    ):
        self.input_shape = input_shape
        self.neurons = hidden_neurons
        self.activation = activation
        self.l1 = l1
        self.l2 = l2
        self.max_fc = max_fc
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_conv = max_conv
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regression_target = regression_target
        self.regression_activation = regression_activation
        self.callback = callback
        self.learnable_parameters = self.get_learnable_parameters()

    @staticmethod
    def create_new_chromosome(genes: List[object]):
        return Conv2DChromosome(genes=genes)

    def get_learnable_parameters(self):

        handler_params = list(self.__dict__.keys())
        learnable_parameters = dict()

        for param in handler_params:
            if isinstance(getattr(self, param), list):
                learnable_parameters[param] = getattr(self, param)

        return learnable_parameters

    def parameter_choice(self, parameter):

        params = self.learnable_parameters.get(parameter)

        if params is not None:
            return random.choice(params)
        else:
            return getattr(self, parameter)

    def generate_population(self, population_size: int) -> List:

        assert isinstance(population_size, int)

        population: List[Conv2DChromosome] = []

        training_params = [
            'batch_size',
            'optimizer',
            'learning_rate',
        ]

        key_store: List[str] = []

        while len(population) < population_size:

            genes = [
                Conv2DGene(
                    gene_type='conv2d',
                    filters=self.parameter_choice('filters'),
                    kernel_size=self.parameter_choice('kernel_size'),
                    stride=self.parameter_choice('stride'),
                    activation=self.parameter_choice('activation'),
                )
                for _ in range(self.max_conv)
            ] + \
                [FCGene(
                gene_type='fc',
                neurons=self.parameter_choice('neurons'),
                activation=self.parameter_choice('activation'),
                dropout=self.parameter_choice('dropout'),
                l1=self.parameter_choice('l1'),
                l2=self.parameter_choice('l2'),
            ) for _ in range(self.max_fc)
                    ] + \
                    [
                        ParameterGene(
                            parameter_name=param,
                            parameter=self.parameter_choice(param)
                        )
                        for param in training_params
                    ]
            chromosome = Conv2DChromosome(
                genes,

            )

            key = chromosome.get_unique_key()

            if key not in key_store:
                population.append(chromosome)
                key_store.append(key)
            else:
                pass

        return population

    def asses(self,
              x_train: npt.NDArray[np.float32],
              y_train: npt.NDArray[np.float32],
              x_test: npt.NDArray[np.float32],
              y_test: npt.NDArray[np.float32],
              chromosome: Conv2DChromosome,
              epochs: int
              ):

        model = chromosome.decode(self.input_shape, self.regression_target, self.regression_activation)

        optimizer = chromosome.get_parameter('optimizer', self.optimizer)

        learning_rate = chromosome.get_parameter('learning_rate', self.learning_rate)

        batch_size = chromosome.get_parameter('batch_size', self.batch_size)

        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.learning_rate = learning_rate

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[tfa.metrics.RSquare()]
        )

        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_split=0.1,
            batch_size=batch_size,
            callbacks=[self.callback],
            verbose=0
        )

        mse, r_square = model.evaluate(x_test, y_test, verbose=0)

        return model, mse, r_square
