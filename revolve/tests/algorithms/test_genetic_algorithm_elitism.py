# import pytest
#
#
# class TestGeneticAlgorithmElitism:
#
#     def test_evolve_population(self, ga_elitism_mlp, data):
#         x_train, y_train, x_test, y_test = data
#         epochs, generation = 1, 1
#         assert hasattr(ga_elitism_mlp.strategy, 'asses')
#         assert isinstance(ga_elitism_mlp.strategy.asses, object)
#         ga_elitism_mlp.evolve_population(x_train, y_train, x_test, y_test, epochs, generation)
#         # Assert that the population size remains the same
#         assert len(ga_elitism_mlp.population) == 10
#
#         # Assert that the elite models are correctly selected
#         assert ga_elitism_mlp.elite_models == [1, 2]
#
#         # Assert that the crossover and mutation methods are called
#         ga_elitism_mlp.operations.tournament_selection.assert_called()
#         ga_elitism_mlp.operations.uniform_crossover.assert_called()
#         ga_elitism_mlp.operations.mutation.assert_called()
