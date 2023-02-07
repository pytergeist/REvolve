from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from revolve.architectures.base import BaseChromosome
import tensorflow as tf
import numpy as np


class BaseEvolutionaryAlgorithm(ABC):
    @staticmethod
    def get_min_fitness(population):
        min_idx = np.argsort([chromosome.loss for chromosome in population])[0]
        return population[min_idx]

    @abstractmethod
    def evolve_population(
        self,
        data: Tuple[tf.data.Dataset],
        generation: int,
    ) -> list:
        pass

    @abstractmethod
    def _population_asses(self, data: Tuple[tf.data.Dataset]):
        pass

    def get_model_fitness(
        self,
        chromosome: BaseChromosome,
        data: Tuple[tf.data.Dataset],
    ) -> tf.keras.Model:
        model, loss, metric = chromosome.get_fitness(
            self.strategy.parameters,
            chromosome.genes,
            data,
            self.strategy.epochs,
            self.strategy.callback,
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
        pass
