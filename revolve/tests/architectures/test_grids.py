import pytest
from revolve.architectures.grids import MLPParameterGrid, ConvParameterGrid


@pytest.mark.parametrize(
    "grid, parameters",
    [(MLPParameterGrid, "mlp_params"), (ConvParameterGrid, "conv_network_params")],
)
def test_grid_init(grid, parameters, request):
    parameters = request.getfixturevalue(parameters)
    grid = grid(**parameters)
    assert all([key in parameters.keys() for key in grid.__dict__.keys()])


@pytest.mark.parametrize(
    "grid, parameters, choice, expected",
    [
        (MLPParameterGrid, "mlp_params", "hidden_neurons", [200, 300]),
        (MLPParameterGrid, "mlp_params", "batch_size", [16, 64]),
        (ConvParameterGrid, "conv_network_params", "filters", [32, 64]),
        (ConvParameterGrid, "conv_network_params", "activation", ["swish", "tanh"]),
    ],
)
def test_grid_get_parameter(grid, parameters, choice, expected, request):
    parameters = request.getfixturevalue(parameters)
    grid = grid(**parameters)
    param = grid.get(choice)
    assert param == expected


@pytest.mark.parametrize(
    "grid, parameters",
    [
        (MLPParameterGrid, "mlp_params"),
        (ConvParameterGrid, "conv_network_params"),
    ],
)
def test_grid_learnable_parameters(grid, parameters, request):
    parameters = request.getfixturevalue(parameters)
    grid = grid(**parameters)
    learnable_parameters = {key: value for key, value in grid.learnable_parameters}
    assert all(
        [
            key in parameters.keys()
            for key in learnable_parameters.keys()
            if isinstance(parameters[key], list)
        ]
    )


@pytest.mark.parametrize(
    "grid, parameters",
    [
        (MLPParameterGrid, "mlp_params"),
        (ConvParameterGrid, "conv_network_params"),
    ],
)
def test_grid_learnable_parameters(grid, parameters, request):
    parameters = request.getfixturevalue(parameters)
    grid = grid(**parameters)
    learnable_parameters = {key: value for key, value in grid.static_parameters}
    assert all(
        [
            key in parameters.keys()
            for key in learnable_parameters.keys()
            if not isinstance(parameters[key], list)
        ]
    )
