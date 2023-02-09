## REvolve - A python library for evolutionary neural architecture search for regression tasks ##

REvolve is a python library for performing simple evolutionary neural architecture search for regression problems.  REvolve is split into four components: algorithms, architectures, grids, and operators.  The algorithms' directory contains the evolutionary algorithms, architectures contains the 
network strategy modules along with the chromosome and gene encodings of the architectures, 
grids provides classes to define the search grid for each strategy, and operators holds
evolutionary operations (selection, mutation, and crossover) along with 
an Operator class which allows you to register different evolutionary
operations based on your specific task. 

## Grids

Two search spaces are currently supported, one for multilayer perceptrons
and one for 2D-Convolution network; both of which can be imported from the grids module.

```python 
from revolve.grids import MLPParameterGrid
from revolve.grids import ConvParameterGrid
```

The parameters of these grids define the hyperparameter search space, parameters specified as non-iterables
(e.g. strings, floats and integers) are treated as static, whilst those defined as lists will be automatically detected
as learnable hyperparameter and used in the neural evolution algorithm.

```python 
mlp_params = MLPParameterGrid(
    # always static parameters
    input_shape=(x_train.shape[1],), 
    regression_target=1,
    regression_activation= "relu",
    
    #####################################################
    # Any of the below parameters can be specified as a #
    #  list to make them learnable or a single value to #
    #                   make them static                #
    #####################################################

    
    # specifying 0 in hidden_neurons makes the number
    # of layers variable
    hidden_neurons=[0, 100, 200, 300] 
    activation='relu',
    dropout=[0.1, 0.2, 0.5],
    batch_size=64,
    optimizer='adam',
    learning_rate=[1e-3, 1e-4, 1e-5],
    l1=[1e-3, 1e-4, 1e-5],
    l2=1e-5,
)
```

## Strategies

Two strategies are currently implemented: the MLPStrategy which takes the MLPParameterGrid
as input, and the Conv2DStrategy which takes ConvParameterGrid as input.
Each strategy can be imported from the architecture module of REvolve. 


```python 
from revolve.architectures import MLPStrategy
from revolve.architectures import Conv2DStrategy
```

The strategy objects take the parameter grid previously specified as input along with the maximum number of layers, 
the number of epochs used in training, callbacks, loss, metrics, and a squeeze/expand parameter for constraining the 
architecture. The loss and metrics can either be input by their string names or as defined tensorflow objects.

```python 
loss = 'mean_squared_error'
loss = tf.keras.losses.MeanSquaredError()
```

The squeeze_fc parameter ensures that h_0>h_1>...>h_n, where h_n refers to the number of logits on hidden 
layer n. The expand_conv parameter (seen in examples/conv2d_regression.ipynb -- coming very soon!) ensures that c_0<c_1<c_2...c_n, where c_n
refers to the filter size of convolution layers n. These two parameter can be True/False to implement or not implament 
the constraint. These constraints only apply to the initial population of architectures.

```python 
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2)

mlp_strategy = MLPStrategy(
    max_fc = 3,
    squeeze_fc=False, # do not constrain
    epochs=50,
    callback=callback
    loss='mean_squared_error',
    metric = tfa.metrics.RSquare(name='r_square'),
    parameters = mlp_params
)
```

## Operators

The operators' module currently has contains two selection operations (tournament_selection and 
roulette_wheel_selection), one mutation operation (random_mutation), and one cross over operation (uniform_crossover).

```python 
from revolve.operators import (
    mutation,
    tournament_selection,
    uniform_crossover,
    roulette_wheel_selection,
)
```

The operators' module provides as Operations class that is used to register
the evolutionary operations to be used, either from REvolve or user defined. The Operations class allows for argument 
definitions. Currently, the EAs framework take operations in the order selection, crossover, mutation so ensure you 
register the evolutionary operations in that order.


```python 
from revolve.operators import Operations

# example with roulette_wheel_selection
operations = Operations()
operations.register(roulette_wheel_selection)
operations.register(uniform_crossover, probability=0.9)
operations.register(mutation, probability=0.2)

# example with tournament selection
operations = Operations()
operations.register(tournament_selection, size=5)
operations.register(uniform_crossover, probability=0.9)
operations.register(mutation, probability=0.2)
```

## Algorithms 

The algorithm currently implemented is EA with elitism. This can both be imported from the 
algorithms module and take the strategy, population_size, and elitism_size, and operations as arguments..


```python 
from from revolve.algorithms import EvolutionaryAlgorithmElitism

ea = EvolutionaryAlgorithmElitism(
    strategy=mlp_strategy,
    pop_size=10,
    elitism_size=2,
    operations=operations,

)
```

Once initialised the EA can be run with the .fit() function, which takes data and the number 
of generations as input. The data must be provided as a tuple of tensorflow datasets; including 
training_data, validation_data, and test_data.


```python 
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

data = (train_data, valid_data, test_data)

best_chromosome = ea.fit(
    data=data,
    generations=5,
)
```

Once fit the EA will return the best chromosome, which can be decoded into a model
with the .decode function which take the grid parameters as an argument. The results 
for every generation can be returned as a dataframe with the ea.results_df() function. 
This returns a dataframe with the learnt_parameters, static_parameters, loss and metric values and
the generation it was found in for every chromosome. The row of parameters for the best chromosome can 
be returned by parsing the dataframe for the lowest loss value.

```python 
model = best_chromosome.decode(params)
df = ga.results_df()
best_chromosome_row = df[df.loss == df.loss.min() ]
```

The elite models can be accessed with the ea.elite_models attribute, returning a sorted list of the 
trained elite models. 

```python 
elite_models = ga.elite_models

[<keras.engine.functional.Functional at 0x2a194c8b0>,
 <keras.engine.functional.Functional at 0x2b34b4dc0>]
```











