import pytest
import unittest.mock as mock
import tensorflow as tf


@pytest.mark.parametrize(
    "chromosome",
    [
        "mlp_chromosome",
        "conv2d_chromosome",
    ],
)
def test_chromosome_get_unique_key(chromosome, request):
    chromosome = request.getfixturevalue(chromosome)
    key = chromosome.get_unique_key(chromosome.genes)
    assert isinstance(key, str)


@pytest.mark.parametrize(
    "chromosome, params",
    [
        ("mlp_chromosome", "mlp_params"),
        ("conv2d_chromosome", "conv_network_params"),
    ],
)
def test_mlp_chromosome_decode(chromosome, params, request):
    chromosome = request.getfixturevalue(chromosome)
    params = request.getfixturevalue(params)
    model = chromosome.decode(params)
    assert isinstance(model, tf.keras.Model)


@pytest.mark.parametrize(
    "chromosome, genes, gene_params",
    [
        ("mlp_chromosome", "mlp_chromosome_genes", "parameter_gene_params"),
        ("mlp_chromosome", "mlp_chromosome_fc_only_genes", "parameter_gene_params"),
        ("conv2d_chromosome", "conv_chromosome_genes", "parameter_gene_params"),
        (
            "mlp_chromosome",
            "conv_chromosome_fc_conv_only_genes",
            "parameter_gene_params",
        ),
    ],
)
def test_chromosome_get_parameter(chromosome, genes, gene_params, request):
    default_value = 10
    chromosome = request.getfixturevalue(chromosome)
    genes = request.getfixturevalue(genes)
    gene_params = request.getfixturevalue(gene_params)

    for key, value in gene_params.items():
        param = chromosome.get_parameter(key, default_value, genes)
        if any([len(gene.parameters) == 1 for gene in genes]):
            assert param == value
        else:
            assert param == default_value


@pytest.mark.parametrize(
    "chromosome, params",
    [
        ("mlp_chromosome", "mlp_params"),
        ("conv2d_chromosome", "conv_network_params"),
    ],
)
def test_chromosome_decode(chromosome, params, request):
    chromosome = request.getfixturevalue(chromosome)
    params = request.getfixturevalue(params)
    model = chromosome.decode(params)
    assert isinstance(model, tf.keras.Model)


@pytest.mark.parametrize(
    "chromosome, params, gene_params",
    [
        ("mlp_chromosome", "mlp_params", "parameter_gene_params"),
        ("conv2d_chromosome", "conv_network_params", "parameter_gene_params"),
    ],
)
def test_build_and_compile_model(chromosome, params, gene_params, request):
    chromosome = request.getfixturevalue(chromosome)
    params = request.getfixturevalue(params)
    gene_params = request.getfixturevalue(gene_params)
    loss = "mean_squared_error"
    metric = "mean_absolute_error"
    model = chromosome.build_and_compile_model(params, loss, metric, chromosome.genes)
    assert isinstance(model, tf.keras.Model)
    assert (
        str(model.optimizer.name)
        == tf.keras.optimizers.get(gene_params["optimizer"]).name
    )
    assert model.optimizer.learning_rate == gene_params["learning_rate"]


@pytest.mark.parametrize(
    "chromosome",
    [
        "mlp_chromosome",
        "conv2d_chromosome",
    ],
)
def test_fit_model(chromosome, request):
    chromosome = request.getfixturevalue(chromosome)
    model_mock = mock.MagicMock()
    train_data_mock = mock.MagicMock()
    valid_data_mock = mock.MagicMock()
    callback_mock = mock.MagicMock()

    train_data_mock.batch.return_value = "train_batch"
    valid_data_mock.batch.return_value = "valid_batch"

    epochs = 10
    batch_size = 128

    chromosome.fit_model(
        model_mock, train_data_mock, valid_data_mock, epochs, batch_size, callback_mock
    )

    model_mock.fit.assert_called_with(
        "train_batch",
        epochs=epochs,
        validation_data="valid_batch",
        callbacks=[callback_mock],
        verbose=0,
    )


@pytest.mark.parametrize(
    "chromosome",
    [
        "mlp_chromosome",
        "conv2d_chromosome",
    ],
)
def test_evaluate_model(chromosome, request):
    chromosome = request.getfixturevalue(chromosome)
    model_mock = mock.MagicMock()
    test_data_mock = mock.MagicMock()

    test_data_mock.batch.return_value = "test_batch"
    batch_size = 128

    result = chromosome.evaluate_model(model_mock, test_data_mock, batch_size)

    model_mock.evaluate.assert_called_with(
        "test_batch",
        verbose=0,
    )
    assert result == model_mock.evaluate.return_value


@pytest.mark.parametrize(
    "chromosome, data, params",
    [
        ("mlp_chromosome", "mock_data", "mlp_params"),
        ("conv2d_chromosome", "mock_data", "conv_network_params"),
    ],
)
def test_get_fitness(chromosome, data, params, request):
    chromosome = request.getfixturevalue(chromosome)
    mock_data = request.getfixturevalue(data)
    params = request.getfixturevalue(params)

    train_data, valid_data, test_data = mock_data
    callback = mock.MagicMock()

    model_mock = mock.MagicMock()
    batch_size = params["batch_size"]
    epochs = 10
    loss = mock.MagicMock()
    metric = mock.MagicMock()
    loss_value = 0.5
    metric_value = 0.6

    chromosome.build_and_compile_model = mock.MagicMock(return_value=model_mock)
    chromosome.get_parameter = mock.MagicMock(return_value=batch_size)
    chromosome.fit_model = mock.MagicMock(return_value=model_mock)
    chromosome.evaluate_model = mock.MagicMock(return_value=(loss_value, metric_value))

    result = chromosome.get_fitness(
        params,
        chromosome.genes,
        mock_data,
        loss,
        metric,
        epochs,
        callback,
    )

    chromosome.build_and_compile_model.assert_called_with(
        params, loss, metric, chromosome.genes
    )
    chromosome.get_parameter.assert_called_with(
        "batch_size", params.get("batch_size"), chromosome.genes
    )
    chromosome.fit_model.assert_called_with(
        model_mock, train_data, valid_data, epochs, batch_size, callback
    )

    chromosome.evaluate_model.assert_called_with(model_mock, test_data, batch_size)

    assert result == (model_mock, loss_value, metric_value)
