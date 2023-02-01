import random
import tensorflow as tf
from typing import Any
from .bases import BaseGene


class FCGene(BaseGene):
    def __init__(self,
                 gene_type: str, neurons: int,
                 activation: str, dropout: float,
                 l1: float, l2: float,
                 ):
        self.name = 'fc_gene'
        self.gene_type = gene_type
        self.neurons = neurons
        self.activation = activation
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2

        assert self.gene_type == 'fc'
        assert isinstance(self.neurons, int), f'invalid number of logits: {self.neurons}'
        assert hasattr(tf.keras.activations, self.activation), f'unknown activation function: {self.activation}'
        assert isinstance(self.dropout, float), f'Invalid value for dropout: {self.dropout}'
        assert 0 < self.dropout < 1.0, f'Invalid value for dropout: {self.dropout}'
        assert isinstance(self.l1, float) and self.l1 >= 0, f'Invalid L1: {self.l1}'
        assert isinstance(self.l2, float) and self.l2 >= 0, f'Invalid L2: {self.l2}'

    def mutate(self, learnable_parameters: dict):

        parameters = self.get_attribute_names()

        for param in parameters:
            if learnable_parameters.get(param):
                setattr(self, param, random.choice(learnable_parameters[param]))
            else:
                pass

    def get_attributes(self):
        return list(self.__dict__.values())[2:]

    @classmethod
    def get_attribute_names(cls):
        config = {
            'gene_type': 'fc', 'neurons': 1,
            'activation': 'relu', 'dropout': 0.1,
            'l1': 0.1, 'l2': 0.1,
        }
        tmp_cls = cls(**config)
        attribute_names = list(tmp_cls.__dict__.keys())
        return attribute_names[2:]


class Conv2DGene(BaseGene):
    def __init__(self, gene_type: str, filters: int, kernel_size: int, stride: int, activation: str):
        self.name = 'conv2d_gene'
        self.gene_type = gene_type
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

        assert self.gene_type == 'conv2d'
        assert isinstance(filters, int), f'invalid num filter: {self.filters}, enter as integer'
        assert isinstance(kernel_size, int), f'invalid kernel size: {self.kernel_size}, enter as integer'
        assert isinstance(stride, int), f'invalid stride: {self.stride}, enter as integer'
        assert hasattr(tf.keras.activations, self.activation), f'unknown activation function: {self.activation}'

    def mutate(self, learnable_parameters: dict):

        parameters = self.get_attribute_names()

        for param in parameters:
            if learnable_parameters.get(param):
                setattr(self, param, random.choice(learnable_parameters[param]))
            else:
                pass

    def get_attributes(self):
        return list(self.__dict__.values())[2:]

    @classmethod
    def get_attribute_names(cls):
        config = {
            'gene_type': 'conv2d', 'filters': 1,
            'kernel_size': 2, 'stride': 1,
            'activation': 'relu',
        }
        tmp_cls = cls(**config)
        attribute_names = list(tmp_cls.__dict__.keys())
        return attribute_names[2:]


class ParameterGene(BaseGene):
    def __init__(self, parameter_name: str, parameter: Any):
        self.name = 'param_gene'
        self.gene_type = parameter_name
        self.parameter = parameter

    def mutate(self, learnable_parameters):

        if learnable_parameters.get(self.gene_type):
            self.parameter == random.choice(learnable_parameters[self.gene_type])

    def get_attributes(self):
        return list(self.__dict__.values())[1:]

    def get_attribute_names(self):
        return self.gene_type

