from typing import Tuple
import tensorflow as tf
import numpy as np
from typing import List
from revolve.architectures.base import BaseStrategy, BaseChromosome
from .base import BaseEvolutionaryAlgorithm


class EvolutionaryAlgorithmElitism(BaseEvolutionaryAlgorithm):
    """
    A subclass of `BaseEvolutionaryAlgorithm` implementing an elitism approach to evolve population.

    Attributes:
    strategy (BaseStrategy): An object of `BaseStrategy` class representing the strategy for the model.
    pop_size (int): An integer representing the size of population.
    elitism_size (int): An integer representing the number of models to be carried over from previous generation.
    data (List[list]): A list of lists representing the data of each chromosome.
    elite_models (List[None or Model]): A list of model objects representing the top models from previous generations.
    population (List[BaseStrategy]): A list of objects of `BaseStrategy` class representing the population.
    operations (object): An object containing operations such as selection, crossover and mutation to be performed
    on population.

    """

    def __init__(
        self,
        strategy: BaseStrategy,
        pop_size: int,
        elitism_size: int,
        operations: object,
    ):
        """
        Initializes the object of class `EvolutionaryAlgorithmElitism`

        Arguments:
        strategy (BaseStrategy): An object of `BaseStrategy` class representing the strategy for the model.
        pop_size (int): An integer representing the size of population.
        elitism_size (int): An integer representing the number of models to be carried over from previous generation.
        operations (object): An object containing operations such as selection, crossover and mutation to be performed on population.

        """
        self.strategy = strategy
        self.pop_size = pop_size
        self.elitism_size = elitism_size
        self.data: List[list] = []
        self.elite_models = [None] * elitism_size
        self.population: List[BaseChromosome] = []
        self.operations = operations

    @staticmethod
    def elitism(
        population: List, elitism_size: int, models: list
    ) -> Tuple[list[BaseChromosome], list[tf.keras.Model]]:
        elite_idx = np.argsort([chromosome.loss for chromosome in population])[
            :elitism_size
        ]
        elite_models = [model for model in list(map(models.__getitem__, elite_idx))]
        population = [
            chromosome for chromosome in list(map(population.__getitem__, elite_idx))
        ]

        return population, elite_models

    def evolve_population(
        self,
        data: Tuple[tf.data.Dataset],
        generation: int,
    ) -> BaseChromosome:
        """
        Function to evolve the population by applying selection, crossover and mutation operations.

        Arguments:
        data (Tuple[tf.data.Dataset]): A tuple containing the training, validation and testing data.
        generation (int): An integer representing the current generation number.

        Returns:
        Chromosome: The best chromosome from the population.

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

            offspring = self.operations.crossover(parents=(parent1, parent2))

            mutated_offspring = self.operations.mutation(
                offspring=offspring, parameters=self.strategy.parameters
            )

            self.population.append(mutated_offspring)

        return self.get_min_fitness(prev_population)

    def _population_asses(self, data: Tuple[tf.data.Dataset]) -> List[tf.keras.Model]:
        """
        A helper function to evaluate the fitness of all models in population.

        Arguments:
        data (Tuple[tf.data.Dataset]): A tuple containing the training, validation and testing data.

        Returns:
        List: A list of all models from the current generation.

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

        tf.keras.backend.clear_session()

        return models

    def get_elite_model_fitness(
        self,
        idx: int,
        chromosome: BaseChromosome,
        data: Tuple[tf.data.Dataset],
    ):
        """
        Evaluate the fitness of elite model from population and updates chromosome

        Args:
        idx (int): Index of the elite chromosome
        chromosome (BaseChromosome): Elite chromosome to be updated
        data (Tuple[tf.data.Dataset]): Tuple of data sets containing training, validation and test datasets.

        """
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
        """
        Train the evolutionary algorithm for a specified number of generations.

        Args:
        - data (Tuple[tf.data.Dataset]): The dataset to use for training and testing the model.
        - generations (int): The number of generations to evolve the population for.

        Returns:
        None
        """
        self.population = self.strategy.generate_population(self.pop_size)

        for generation in range(generations):
            best_chromosome = self.evolve_population(data, generation)
            print(
                f"Generation {generation}, Best error: {best_chromosome.loss}, Best R2 {best_chromosome.metric}"
            )
