import pytest
from copy import deepcopy
from revolve.operators import mutation
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome


@pytest.mark.parametrize(
    "chromosome, parameters, expected_chromosome",
    [
        ("mlp_chromosome", "mlp_params", MLPChromosome),
        ("conv2d_chromosome", "conv_network_params", Conv2DChromosome),
    ],
)
def test_mutation(chromosome, parameters, expected_chromosome, request):
    chromosome = request.getfixturevalue(chromosome)
    parameters = request.getfixturevalue(parameters)
    original_chromosome = deepcopy(chromosome)
    offspring = mutation(chromosome, 1.0, parameters)

    for gene, original_gene in zip(offspring.genes, original_chromosome.genes):
        for param, value in gene.parameters.items():
            if isinstance(parameters.get(param), list):
                assert getattr(gene, param) != getattr(original_gene, param)
