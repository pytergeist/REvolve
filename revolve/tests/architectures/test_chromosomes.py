import pytest
import tensorflow as tf
from revolve.architectures import MLPChromosome, Conv2DChromosome


class TestMLPChromosome:

    def test_get_unique_key(self, gene_list_mlp):
        # Define test inputs
        genes = gene_list_mlp

        # Initialize chromosome
        chromosome = MLPChromosome(genes)

        # Test get_unique_key method
        key = chromosome.get_unique_key()

        # Assert that the key is a string
        assert isinstance(key, str)

    def test_get_parameter(self, gene_list_mlp):
        # Create genes for test
        genes = gene_list_mlp
        # Create instance of MLPChromosome
        chromosome = MLPChromosome(genes)
        # Test get_parameter method
        param = chromosome.get_parameter(param='batch_size', default_param=64)
        assert param == 64

        param = chromosome.get_parameter(param='learning_rate', default_param=1e-5)
        assert param == 1e-3

    def test_decode(self, gene_list_mlp):
        # Define test inputs
        input_shape = 2
        regression_target = 1
        regression_activation = 'relu'
        genes = gene_list_mlp
        # Initialize chromosome
        chromosome = MLPChromosome(genes)

        # Test decode method
        model = chromosome.decode(input_shape, regression_target, regression_activation)
        # Assert that the model is a keras model
        assert isinstance(model, tf.keras.Model)


class TestConv2DChromosome:

    def test_get_unique_key(self, gene_list_conv2d):
        # Define test inputs
        genes = gene_list_conv2d

        # Initialize chromosome
        chromosome = Conv2DChromosome(genes)

        # Test get_unique_key method
        key = chromosome.get_unique_key()

        # Assert that the key is a string
        assert isinstance(key, str)

    def test_get_parameter(self, gene_list_conv2d):
        # Create genes for test
        genes = gene_list_conv2d
        # Create instance of MLPChromosome
        chromosome = Conv2DChromosome(genes)
        # Test get_parameter method
        param = chromosome.get_parameter(param='batch_size', default_param=64)
        assert param == 64

        param = chromosome.get_parameter(param='learning_rate', default_param=1e-5)
        assert param == 1e-3

    def test_decode(self, gene_list_conv2d):
        # Define test inputs
        input_shape = (2, 2, 1)
        regression_target = 1
        regression_activation = 'relu'
        genes = gene_list_conv2d
        # Initialize chromosome
        chromosome = Conv2DChromosome(genes)

        # Test decode method
        model = chromosome.decode(input_shape, regression_target, regression_activation)
        assert any([layer.name.startswith('flatten') for layer in model.layers])

        # Assert that the model is a keras model
        assert isinstance(model, tf.keras.Model)


if __name__ == '__main__':
    pytest.main()
