from typing import Union, Tuple
import tensorflow as tf
import numpy as np
from typing import List
from revolve.architectures.base import Strategy, Chromosome


class EvolutionaryAlgorithmElitism:
    """
    Class that implements a genetic algorithm with elitism.

    Parameters:
    strategy (object): The strategy class object that will be used to assess the population.
    parameters (Dict[float, int]): A dictionary of parameters for the algorithm.

    Attributes:
    data (list): The data collected for each generation.
    architectures (list): The architectures generated for each generation.
    population (list): The population of chromosomes for the current generation.
    """

    def __init__(
        self,
        strategy: Strategy,
        pop_size: int,
        elitism_size: int,
        operations: object,
    ):
        self.strategy = strategy
        self.pop_size = pop_size
        self.elitism_size = elitism_size
        self.data: List[list] = []
        self.elite_models = [None] * elitism_size
        self.population: List[Strategy] = []
        self.operations = operations

    @staticmethod
    def elitism(population: List, elitism_size: int, models: list):
        """
        Selects the elite individuals from the current generation.

        Parameters:
        generation_data (list): The data for the current generation.
        """
        elite_idx = np.argsort([chromosome.loss for chromosome in population])[
            :elitism_size
        ]
        elite_models = [model for model in list(map(models.__getitem__, elite_idx))]
        population = [
            chromosome for chromosome in list(map(population.__getitem__, elite_idx))
        ]

        return population, elite_models

    @staticmethod
    def get_min_fitness(population):
        min_idx = np.argsort([chromosome.loss for chromosome in population])[0]
        return population[min_idx]

    def evolve_population(
        self,
        data: Tuple[tf.data.Dataset],
        generation: int,
    ) -> List:
        """
        Evolves the population of chromosomes for a given number of generations.

        Parameters:
        x_train (NDArray): The training data.
        y_train (NDArray): The target values for the training data.
        x_test (NDArray): The test data.
        y_test (NDArray): The target values for the test data.
        epochs (int): The number of training epochs for each model.
        generation (int): The current generation number.

        Returns:
        List: A list containing the best chromosome for the generation.
        """

        models = self._population_asses(data)

        self.data += [
            [chromosome, chromosome.loss, chromosome.metric, generation]
            for chromosome in self.population
        ]

        prev_population = self.population

        self.population, self.elite_models = self.elitism(
            population=self.population,
            elitism_size=self.elitism_size,
            models=models,
        )

        while len(self.population) < self.pop_size:
            parent1, parent2 = self.operations.selection(prev_population)

            offspring = self.operations.crossover(
                parents=(parent1, parent2), strategy=self.strategy
            )

            mutated_offspring = self.operations.mutation(
                offspring=offspring, parameters=self.strategy.parameters
            )

            self.population.append(mutated_offspring)

        return self.get_min_fitness(prev_population)

    def _population_asses(self, data):
        """
        Assess the population of chromosomes using the given strategy.

        Parameters:
        x_train (NDArray): The training data.
        y_train (NDArray): The target values for the training data.
        x_test (NDArray): The test data.
        y_test (NDArray): The target values for the test data.
        epochs (int): The number of training epochs for each model.
        generation (int): The current generation number.

        Returns:
        List: A list containing the data for each chromosome in the population.
        """
        models = []

        if all(self.elite_models):
            for idx, chromosome in enumerate(self.population):
                if idx > self.elitism_size - 1:
                    model = self.get_model_fitness(
                        chromosome,
                        data,
                    )

                    models.append(model)

                else:
                    self.get_elite_model_fitness(idx, chromosome, data)
                    models.append(self.elite_models[idx])

        else:
            for chromosome in self.population:
                model = self.get_model_fitness(chromosome, data)
                models.append(model)

        return models

    def get_model_fitness(
        self,
        chromosome: Chromosome,
        data: Tuple[tf.data.Dataset],
    ):
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

    def get_elite_model_fitness(
        self,
        idx: int,
        chromosome: Chromosome,
        data: Tuple[tf.data.Dataset],
    ):
        batch_size = chromosome.get_parameter(
            "batch_size", self.strategy.parameters.get("batch_size"), chromosome.genes
        )

        _, _, test_data = data
        loss, metric = self.elite_models[idx].evaluate(
            test_data.batch(batch_size), verbose=0
        )

        chromosome.loss = loss
        chromosome.metric = metric

    def fit(
        self,
        data: Tuple[tf.data.Dataset],
        generations: int,
    ):
        self.population = self.strategy.generate_population(self.pop_size)

        for generation in range(generations):
            best_chromosome = self.evolve_population(data, generation)
            print(
                f"Generation {generation}, Best error: {best_chromosome.loss}, Best R2 {best_chromosome.metric}"
            )
