import numpy as np
import pytest
from revolve.architectures.genes import FCGene, Conv2DGene, ParameterGene
from revolve.architectures.base import BaseGene


@pytest.mark.parametrize(
    "gene, gene_params, gene_type",
    [
        (FCGene, "fc_gene_params", "fc"),
        (Conv2DGene, "conv_gene_params", "conv2d"),
    ],
)
def test_gene_init(gene, gene_params, gene_type, request):
    gene_params = request.getfixturevalue(gene_params)
    new_gene = gene(**gene_params)
    assert isinstance(new_gene, gene)
    assert isinstance(new_gene, BaseGene)
    assert new_gene.gene_type == gene_type
    assert new_gene.parameters == gene_params


@pytest.mark.parametrize(
    "gene, gene_params, learnable_params",
    [
        (FCGene, "fc_gene_params", "fc_learnable_params"),
        (Conv2DGene, "conv_gene_params", "conv_learnable_params"),
    ],
)
def test_fc_gene_mutate(gene, gene_params, learnable_params, request):
    gene_params = request.getfixturevalue(gene_params)
    learnable_params = request.getfixturevalue(learnable_params)

    gene = gene(**gene_params)
    original_params = gene.parameters.copy()
    gene.mutate(learnable_params)
    for param in gene.parameters.keys():
        assert getattr(gene, param) != original_params[param]
        assert getattr(gene, param) in learnable_params[param]

    np.testing.assert_array_equal(
        list(gene.parameters.keys()), list(gene_params.keys())
    )


@pytest.mark.parametrize(
    "gene, gene_params",
    [(FCGene, "fc_gene_params"), (Conv2DGene, "conv_gene_params")],
)
def test_fc_gene_validate_params(gene, gene_params, request):
    gene_params = request.getfixturevalue(gene_params)
    fc_gene = gene(**gene_params)
    assert fc_gene._validate_params() is None


@pytest.mark.parametrize(
    "gene, gene_params",
    [(FCGene, "fc_gene_params"), (Conv2DGene, "conv_gene_params")],
)
def test_fc_gene_get_attributes(gene, gene_params, request):
    gene_params = request.getfixturevalue(gene_params)
    gene = gene(**gene_params)
    assert gene.get_attributes() == list(gene_params.values())


@pytest.mark.parametrize(
    "gene, gene_params",
    [(FCGene, "fc_gene_params"), (Conv2DGene, "conv_gene_params")],
)
def test_fc_gene_get_attribute_names(gene, gene_params, request):
    gene_params = request.getfixturevalue(gene_params)
    gene = gene(**gene_params)
    assert gene.get_attribute_names == list(gene_params.keys())


def test_parameter_gene_init(parameter_gene_params):
    for key, value in parameter_gene_params.items():
        param_gene = ParameterGene(parameter_name=key, parameter=value)
        assert isinstance(param_gene, ParameterGene)
        assert isinstance(param_gene, BaseGene)
        assert param_gene.gene_type == key
        assert hasattr(param_gene, key)
        assert getattr(param_gene, key) == value


def test_parameter_gene_mutate(parameter_gene_params, parameter_learnable_params):
    for key, value in parameter_gene_params.items():
        param_gene = ParameterGene(parameter_name=key, parameter=value)
        original_params = param_gene.parameters.copy()
        param_gene.mutate(parameter_learnable_params)
        assert getattr(param_gene, key) != original_params[key]
        for param in param_gene.parameters.keys():
            assert getattr(param_gene, param) in parameter_learnable_params[param]


def test_parameter_gene_validate_params(parameter_gene_params):
    for key, value in parameter_gene_params.items():
        param_gene = ParameterGene(parameter_name=key, parameter=value)
        assert param_gene._validate_params() is None


def test_parameter_gene_get_attributes(parameter_gene_params):
    for key, value in parameter_gene_params.items():
        param_gene = ParameterGene(parameter_name=key, parameter=value)
        assert param_gene.get_attributes() == [value]


def test_parameter_gene_get_attribute_names(parameter_gene_params):
    for key, value in parameter_gene_params.items():
        param_gene = ParameterGene(parameter_name=key, parameter=value)
        assert param_gene.get_attribute_names == [key]


if __name__ == "__main__":
    pytest.main()
