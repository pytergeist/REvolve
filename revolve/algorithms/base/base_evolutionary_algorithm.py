"""
File containing base class for evolutionary algorithms, implementing abstract
and defined methods
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Union
import tensorflow as tf
import numpy as np

from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome
from revolve.architectures.strategies import MLPStrategy, Conv2DStrategy


class BaseEvolutionaryAlgorithm(ABC):
    """
    An abstract base class for implementing Evolutionary Algorithm.

    Methods
    -------
    get_min_fitness(population)
        Get the chromosome with the lowest loss from a population.
    evolve_population(data, generation)
        Evolve the population to the next generation.
    _population_asses(data)
        Assess the population's fitness.
    get_model_fitness(chromosome, data)
        Get the fitness of a chromosome.
    fit(data, generations)
        Fit the evolutionary algorithm to data for a given number of generations.
    """

    @staticmethod
    def get_min_fitness(population):
        """
        Get the chromosome with the lowest loss from a population.

        Parameters
        ----------
        population : list[chromosome]
            The population of chromosomes.

        Returns
        -------
        chromosome : BaseChromosome
            The chromosome with the lowest loss.
        """
        min_idx = np.argsort([chromosome.loss for chromosome in population])[0]
        return population[min_idx]

    @abstractmethod
    def evolve_population(
        self,
        data: Tuple[tf.data.Dataset],
        generation: int,
    ) -> list:
        """
        abstract method to evolve the population to the next generation.

        Parameters
        ----------
        data : Tuple[tf.data.Dataset]
            The data to fit the algorithm to.
        generation : int
            The current generation of the population.

        Returns
        -------
        population : list[Chromosome]
            The evolved population.
        """

    @abstractmethod
    def _population_asses(self, data: Tuple[tf.data.Dataset]):
        """
        Assess the population's fitness.

        Parameters
        ----------
        data : Tuple[tf.data.Dataset]
            The data to fit the algorithm to.
        """

    def get_model_fitness(
        self,
        chromosome: Union[MLPChromosome, Conv2DChromosome],
        data: Tuple[tf.data.Dataset],
        strategy: Union[MLPStrategy, Conv2DStrategy],
    ) -> tf.keras.Model:
        """
        Get the fitness of a chromosome.

        Parameters
        ----------
        chromosome : BaseChromosome
            The chromosome to get the fitness for.
        data : Tuple[tf.data.Dataset]
            The data to fit the chromosome with.
        strategy: Union[MLPStrategy, Conv2DStrategy]
            architecture strategy passed up from child class

        Returns
        -------
        model : tf.keras.Model
            The fitted model.
        """
        model, loss, metric = chromosome.get_fitness(
            strategy.parameters,
            chromosome.genes,
            data,
            strategy.loss,
            strategy.metric,
            strategy.epochs,
            strategy.callback,
        )
        chromosome.loss = loss
        chromosome.metric = metric

        return model

    @abstractmethod
    def fit(
        self,
        data: Tuple[tf.data.Dataset],
        generations: int,
    ):
        """
        Fit the evolutionary algorithm to data for a given number of generations.

        Parameters
        ----------
        data : Tuple[tf.data.Dataset]
            The data to fit the chromosome with.
        generations : int
            The number of generations to run the EA for
        """
