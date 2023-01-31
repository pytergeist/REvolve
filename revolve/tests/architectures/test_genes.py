import pytest
from typing import Any
from revolve.architectures import FCGene, ParameterGene, Conv2DGene


class TestFCGene:

    def test_init(self, fc_gene):
        assert fc_gene.name == 'fc_gene'
        assert fc_gene.gene_type == 'fc'
        assert fc_gene.neurons == 128
        assert fc_gene.activation == 'relu'
        assert fc_gene.dropout == 0.5
        assert fc_gene.l1 == 0.01
        assert fc_gene.l2 == 0.01

    def test_init_errors(self):
        with pytest.raises(AssertionError) as e:
            FCGene(gene_type='fc', neurons='1.1', activation='relu', dropout=0.5, l1=0.01, l2=0.01)
        assert str(e.value) == 'invalid number of logits: 1.1'

        with pytest.raises(AssertionError) as e:
            FCGene(gene_type='fc', neurons=128, activation='test', dropout=0.5, l1=0.01, l2=0.01)
        assert str(e.value) == 'unknown activation function: test'

        with pytest.raises(AssertionError) as e:
            FCGene(gene_type='fc', neurons=128, activation='relu', dropout=-0.1, l1=0.01, l2=0.01)
        assert str(e.value) == 'Invalid value for dropout: -0.1'

        with pytest.raises(AssertionError) as e:
            FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.1, l1=-7, l2=0.01)
        assert str(e.value) == 'Invalid L1: -7'

        with pytest.raises(AssertionError) as e:
            FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.1, l1=0.03, l2=-10)
        assert str(e.value) == 'Invalid L2: -10'

    def test_mutate_learnable(self, mlp_learnable_parameters):
        fc_gene = FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.5, l1=0.01, l2=0.01)
        fc_gene.mutate(mlp_learnable_parameters)
        assert fc_gene.neurons in mlp_learnable_parameters['neurons']
        assert fc_gene.activation in mlp_learnable_parameters['activation']
        assert fc_gene.dropout in mlp_learnable_parameters['dropout']
        assert fc_gene.l1 in mlp_learnable_parameters['l1']
        assert fc_gene.l2 in mlp_learnable_parameters['l2']

    def test_mutate_not_learnable(self, fc_gene, mlp_learnable_parameters={}):
        fc_gene.mutate(mlp_learnable_parameters)
        assert fc_gene.neurons == 128
        assert fc_gene.activation == 'relu'
        assert fc_gene.dropout == 0.5
        assert fc_gene.l1 == 0.01
        assert fc_gene.l2 == 0.01

    def test_get_attributes(self):
        fc_gene = FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.5, l1=0.01, l2=0.01)
        assert fc_gene.get_attributes() == [128, 'relu', 0.5, 0.01, 0.01]

    def test_get_attribute_names(self):
        assert FCGene.get_attribute_names() == ['neurons', 'activation', 'dropout', 'l1', 'l2']


class TestCon2DGene:

    def test_init(self, conv2d_gene):
        assert conv2d_gene.name == 'conv2d_gene'
        assert conv2d_gene.gene_type == 'conv2d'
        assert conv2d_gene.filters == 16
        assert conv2d_gene.kernel_size ==2
        assert conv2d_gene.stride == 1
        assert conv2d_gene.activation == 'relu'

    def test_init_errors(self):
        with pytest.raises(AssertionError) as e:
            Conv2DGene(gene_type='conv2d', filters=1.1, kernel_size=2, stride=1, activation='relu')
        assert str(e.value) == 'invalid num filter: 1.1, enter as integer'

        with pytest.raises(AssertionError) as e:
            Conv2DGene(gene_type='conv2d', filters=1, kernel_size=(2, 3, 4), stride=1, activation='relu')
        assert str(e.value) == 'invalid kernel size: (2, 3, 4), enter as integer'

        with pytest.raises(AssertionError) as e:
            Conv2DGene(gene_type='conv2d', filters=1, kernel_size=2, stride=(1, 1), activation='relu')
        assert str(e.value) == 'invalid stride: (1, 1), enter as integer'

        with pytest.raises(AssertionError) as e:
            Conv2DGene(gene_type='conv2d', filters=1, kernel_size=2, stride=1, activation='test')
        assert str(e.value) == 'unknown activation function: test'

    def test_mutate_learnable(self, conv2d_learnable_parameters):
        conv2d_gene = Conv2DGene(gene_type='conv2d', filters=1, kernel_size=2, stride=1, activation='relu')
        conv2d_gene.mutate(conv2d_learnable_parameters)
        assert conv2d_gene.filters in conv2d_learnable_parameters['filters']
        assert conv2d_gene.kernel_size in conv2d_learnable_parameters['kernel_size']
        assert conv2d_gene.stride in conv2d_learnable_parameters['stride']
        assert conv2d_gene.activation in conv2d_learnable_parameters['activation']

    #
    def test_mutate_not_learnable(self, conv2d_gene, mlp_learnable_parameters={}):
        conv2d_gene.mutate(mlp_learnable_parameters)
        assert conv2d_gene.filters == 16
        assert conv2d_gene.kernel_size == 2
        assert conv2d_gene.stride == 1
        assert conv2d_gene.activation == 'relu'

    def test_get_attributes(self):
        conv2d_gene = Conv2DGene(gene_type='conv2d', filters=1, kernel_size=2, stride=1, activation='relu')
        assert conv2d_gene.get_attributes() == [1, 2, 1, 'relu']

    def test_get_attribute_names(self):
        assert Conv2DGene.get_attribute_names() == ['filters', 'kernel_size', 'stride', 'activation']


class TestParameterGene:
    def test_init(self):
        param_gene = ParameterGene(parameter_name='learning_rate', parameter=0.001)
        assert param_gene.name == 'param_gene'
        assert param_gene.gene_type == 'learning_rate'
        assert param_gene.parameter == 0.001

    def test_mutate_learnable(self, mlp_learnable_parameters):
        param_gene = ParameterGene(parameter_name='learning_rate', parameter=0.001)
        param_gene.mutate(mlp_learnable_parameters)
        assert param_gene.parameter in mlp_learnable_parameters['learning_rate']

    def test_mutate_not_learnable(self, mlp_learnable_parameters={}):
        param_gene = ParameterGene(parameter_name='learning_rate', parameter=0.001)
        param_gene.mutate(mlp_learnable_parameters)
        assert param_gene.parameter == 0.001

    def test_get_attributes(self):
        param_gene = ParameterGene(parameter_name='learning_rate', parameter=0.001)
        assert param_gene.get_attributes() == ['learning_rate', 0.001]


if __name__ == '__main__':
    pytest.main()
