from dataclasses import dataclass


@dataclass
class ParameterGrid:
    """
    Base class that implements a grid  of learnable parameters of the model.

    """

    def get(self, parameter_name):
        """
        Return the value of the specified learnable parameter.

        :param parameter_name: str, the name of the parameter to get
        :return: value of the specified parameter
        """
        return getattr(self, parameter_name)

    @property
    def learnable_parameters(self):
        """
        Return a list of learnable parameters, where each element is a tuple containing the name of the parameter
        and its value(s).

        :return: list of learnable parameters
        """
        return [
            (key, self.__dict__[key])
            for key in self.__dict__
            if isinstance(self.__dict__[key], list)
        ]

    @property
    def static_parameters(self):
        """
        Return a list of static parameters, where each element is a tuple containing the name of the parameter
        and its value.

        :return: list of static parameters
        """
        return [
            (key, self.__dict__[key])
            for key in self.__dict__
            if not isinstance(self.__dict__[key], list)
        ]
