from numpy.typing import NDArray
from typing import List, Union
import numpy as np
import numpy.typing as npt
from typing import List
from revolve.architectures.strategies import MLPStrategy, Conv2DStrategy
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome


class EvolutionaryAlgorithmElitism:
    """
    A class that implements a genetic algorithm with elitism.

    Parameters:
    strategy (object): The strategy class object that will be used to assess the population.
    parameters (Dict[float, int]): A dictionary of parameters for the algorithm.

    Attributes:
    data (list): The data collected for each generation.
    architectures (list): The architectures generated for each generation.
    population (list): The population of chromosomes for the current generation.
    """

    def __init__(self,
                 strategy: Union[MLPStrategy, Conv2DStrategy],
                 pop_size: int,
                 elitism_size: int,
                 operations: object,
                 ):
        self.strategy = strategy
        self.pop_size = pop_size
        self.elitism_size = elitism_size
        self.data: List[list] = []
        self.elite_models = [None] * elitism_size
        self.population: List[Union[MLPChromosome, Conv2DChromosome]] = []
        self.operations = operations

    def elitism(self, generation_data: List, elitism_size: int, models: list):
        """
        Selects the elite individuals from the current generation.

        Parameters:
        generation_data (list): The data for the current generation.
        """
        elite_idx = np.array(generation_data, dtype=object)[:, 1].argsort()[:elitism_size]
        elite_models = [
            model for model in
            list(map(models.__getitem__, elite_idx))
        ]
        population = [
            chromosome[0] for chromosome in
            list(map(generation_data.__getitem__, elite_idx))
        ]

        return population, elite_models

    def evolve_population(self,
                          x_train: NDArray, y_train: NDArray,
                          x_test: NDArray, y_test: NDArray,
                          epochs: int, generation: int
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

        models = self._population_asses(x_train, y_train, x_test, y_test, epochs)

        generation_data = [
            [chromosome, chromosome.loss, chromosome.metric, generation]
            for chromosome in self.population
        ]

        self.data += generation_data

        self.population, self.elite_models = self.elitism(
            generation_data=generation_data,
            elitism_size=self.elitism_size,
            models=models,
        )

        while len(self.population) < self.pop_size:

            for operation in self.operations.get_operations():
                if operation == 'selection':
                    self.operations
                    parent1, parent2 = getattr(self.operations, operation)(
                        generation_data=generation_data,
                    )

                if operation == 'crossover':
                    offspring = getattr(self.operations, operation)(
                        parent1=parent1,
                        parent2=parent2,
                        strategy=self.strategy,
                    )

                if operation == 'mutation':
                    mutated_offspring = getattr(self.operations, operation)(
                        offspring=offspring,
                        learnable_parameters=self.strategy.learnable_parameters
                    )

            self.population.append(mutated_offspring)

        return min(generation_data, key=lambda x: x[1])

    def _population_asses(self, x_train, y_train, x_test, y_test, epochs):
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

                    model = self.model_asses(
                        chromosome, x_train, y_train,
                        x_test, y_test, epochs
                    )

                    models.append(model)

                else:

                    self.model_asses_elite(idx, chromosome, x_test, y_test)
                    models.append(self.elite_models[idx])

        else:

            for chromosome in self.population:
                model = self.model_asses(
                    chromosome, x_train, y_train,
                    x_test, y_test, epochs
                )
                models.append(model)

        return models

    def model_asses(self,
                    chromosome: Union[MLPChromosome, Conv2DChromosome],
                    x_train: npt.NDArray, y_train: npt.NDArray,
                    x_test: npt.NDArray, y_test: npt.NDArray,
                    epochs: int
                    ):

        model, loss, metric = self.strategy.asses(
            x_train, y_train,
            x_test, y_test,
            chromosome, epochs,
        )
        chromosome.loss = loss
        chromosome.metric = metric

        return model

    def model_asses_elite(self,
                          idx: int, chromosome: Union[MLPChromosome, Conv2DChromosome],
                          x_test: npt.NDArray, y_test: npt.NDArray):

        mse, r_square = self.elite_models[idx].evaluate(x_test, y_test, verbose=0)
        chromosome.loss = mse
        chromosome.metric = r_square

    def fit(self,
            x_train: NDArray, y_train: NDArray,
            x_test: NDArray, y_test: NDArray, epochs: int,
            generations: int
            ):

        self.population = self.strategy.generate_population(self.pop_size)

        for generation in range(generations):
            best_chromosome = self.evolve_population(x_train, y_train, x_test, y_test, epochs, generation)
            # if generation % 10 == 0 or generation == generations - 1:
            print(
                f'Generation {generation}, Best error: {best_chromosome[1]}, Best R2 {best_chromosome[2]}'
            )
