import pytest
import numpy as np
import tensorflow as tf
from revolve.architectures import Conv2DStrategy, MLPStrategy
from revolve.architectures import MLPChromosome, Conv2DChromosome


class TestMLPStrategy:

    def test_get_learnable_parameters(self, mlp_strategy):
        learnable_parameters = mlp_strategy.get_learnable_parameters()
        assert 'neurons' in learnable_parameters
        assert 'dropout' in learnable_parameters
        assert 'batch_size' in learnable_parameters
        assert 'optimizer' in learnable_parameters
        assert 'learning_rate' in learnable_parameters
        assert 'l1' in learnable_parameters
        assert 'l2' in learnable_parameters

    def test_create_new_chromosome(self, mlp_strategy, gene_list_mlp):
        new_chromosome = mlp_strategy.create_new_chromosome(gene_list_mlp)
        assert isinstance(new_chromosome, MLPChromosome)

    def test_parameter_choice(self, mlp_strategy):
        neurons_param = mlp_strategy.parameter_choice('neurons')
        assert neurons_param in mlp_strategy.neurons
        activation_param = mlp_strategy.parameter_choice('activation')
        assert activation_param == mlp_strategy.activation

    def test_generate_population(self, mlp_strategy):
        population_size = 5
        population = mlp_strategy.generate_population(population_size)
        assert len(population) == population_size
        for chromosome in population:
            assert isinstance(chromosome, MLPChromosome)

    def test_asses(self, mlp_strategy):
        x_train = np.random.rand(100, 10)
        y_train = np.random.rand(100, 1)
        x_test = np.random.rand(20, 10)
        y_test = np.random.rand(20, 1)
        population = mlp_strategy.generate_population(1)
        chromosome = population[0]
        epochs = 10
        model, mse, r_square = mlp_strategy.asses(x_train, y_train, x_test, y_test, chromosome, epochs)
        assert isinstance(model, tf.keras.Model)
        assert isinstance(mse, float)
        assert isinstance(r_square, float)


class TestConv2DStrategy:

    def test_create_new_chromosome(self, conv2d_strategy, gene_list_conv2d):
        new_chromosome = conv2d_strategy.create_new_chromosome(gene_list_conv2d)
        assert isinstance(new_chromosome, Conv2DChromosome)

    def test_get_learnable_parameters(self, conv2d_strategy):
        learnable_parameters = conv2d_strategy.get_learnable_parameters()
        assert 'neurons' in learnable_parameters
        assert 'dropout' in learnable_parameters
        assert 'batch_size' in learnable_parameters
        assert 'optimizer' in learnable_parameters
        assert 'learning_rate' in learnable_parameters
        assert 'l1' in learnable_parameters
        assert 'l2' in learnable_parameters
        assert 'filters' in learnable_parameters
        assert 'kernel_size' in learnable_parameters
        assert 'stride' in learnable_parameters

    def test_parameter_choice(self, conv2d_strategy):
        neurons_param = conv2d_strategy.parameter_choice('neurons')
        assert neurons_param in conv2d_strategy.neurons
        kernel_size_param = conv2d_strategy.parameter_choice('kernel_size')
        assert kernel_size_param in conv2d_strategy.kernel_size
        activation_param = conv2d_strategy.parameter_choice('activation')
        assert activation_param == conv2d_strategy.activation

    def test_generate_population(self, conv2d_strategy):
        population_size = 5
        population = conv2d_strategy.generate_population(population_size)
        assert len(population) == population_size
        for chromosome in population:
            assert isinstance(chromosome, Conv2DChromosome)

    def test_asses(self, conv2d_strategy):
        x_train = np.random.rand(100, 10, 10, 1)
        y_train = np.random.rand(100, 1)
        x_test = np.random.rand(20, 10, 10, 1)
        y_test = np.random.rand(20, 1)
        population = conv2d_strategy.generate_population(1)
        chromosome = population[0]
        epochs = 10
        model, mse, r_square = conv2d_strategy.asses(x_train, y_train, x_test, y_test, chromosome, epochs)
        assert isinstance(model, tf.keras.Model)
        assert isinstance(mse, float)
        assert isinstance(r_square, float)
