from __future__ import annotations
import random
from abc import ABC, abstractmethod


class BaseGene(ABC):
    """
    Base class for Gene.

    A Gene is a representation of a portion of the machine learning model architecture or training configuration.

    """

    def __init__(self, gene_type: str, **kwargs):
        """
        Initialize a new instance of the class.

        Parameters:
        gene_type (str): The type of the gene.
        kwargs: Other keyword arguments to store as the parameters of the gene.
        """
        self.name = "gene"
        self.gene_type = gene_type
        self.parameters = kwargs

    @abstractmethod
    def _validate_params(self):  # pragma: no cover
        """
        Placeholder method to validate parameters, can be overridden by subclasses

        This method should validate that the parameters of the gene are within acceptable ranges.
        """
        pass

    def mutate(self, learnable_parameters: dict):
        """
        Mutate the gene.

        Parameters:
        learnable_parameters (dict): A dictionary of learnable parameters.
        """
        for param, value in self.parameters.items():
            if isinstance(learnable_parameters.get(param), list):
                setattr(self, param, random.choice(learnable_parameters.get(param)))

    def get_attributes(self):
        """
        Get the attributes of the gene.

        Returns:
        list: A list of values representing the parameters of the gene.
        """
        return list(self.parameters.values())

    @property
    def get_attribute_names(self):
        """
        Get the names of the attributes of the gene.

        Returns:
        list: A list of names of the parameters of the gene.
        """
        return list(self.parameters.keys())
