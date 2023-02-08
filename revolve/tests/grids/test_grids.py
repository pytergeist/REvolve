import pytest
from revolve.grids import MLPParameterGrid, ConvParameterGrid


@pytest.mark.parametrize(
    "grid, parameters",
    [("mlp_grid", "mlp_params"), ("conv_grid", "conv_network_params")],
)
def test_grid_init(grid, parameters, request):
    parameters = request.getfixturevalue(parameters)
    grid = request.getfixturevalue(grid)
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
    "grid",
    [
        "mlp_grid",
        "conv_grid",
    ],
)
def test_grid_learnable_parameters(grid, request):
    grid = request.getfixturevalue(grid)
    learnable_parameters = grid.learnable_parameters
    assert all(list(isinstance(value, list) for value in learnable_parameters.values()))


@pytest.mark.parametrize(
    "grid",
    [
        "mlp_grid",
        "conv_grid",
    ],
)
def test_grid_learnable_parameters(grid, request):
    grid = request.getfixturevalue(grid)
    static_parameters = grid.static_parameters
    assert all(
        list(not isinstance(value, list) for value in static_parameters.values())
    )
