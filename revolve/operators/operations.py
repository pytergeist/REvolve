from functools import partial


class Operations:
    """
    Class that stores evolutionary operations such as selection,
    crossover and mutation.

    methods
    ------
        register: the register method sets a evolutionary operation as an
        attribute to the operations class. The operations can be selection,
        crossover or mutation.
        get_operations: returns list of registered operations
    """

    def register(self, function, *args, **kwargs):
        """

        :param function: the desired function - can be from operators module or user defined
        :param args: additional arguments to be set by functools.partial
        :param kwargs: additional keyword arguments to be set by functools.partial
        :return: None

        func.name assigns name to the attribute, follow naming convention for evolutionary
        operations (i.e. *_selection, *_crossover, *_mutation) as this is needed by evolutionary algorithm
        class.
        """
        func = partial(function, *args, **kwargs)
        func.name = function.__name__.split("_")[-1]

        setattr(self, func.name, func)

    def get_operations(self):
        return list(self.__dict__.keys())
