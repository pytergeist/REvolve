"""
File containing BaseGene class with abstract and defined method:
    each gene inherits BaseGene and every child gene either
    represents a layer or parameter of the network
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseGene(ABC):
    """
    Base class for Gene.

    A Gene is a representation of a portion of the machine learning model architecture
    or training configuration.

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

    def mutate(self, learnable_parameters: Dict[str, Optional[Any]]):
        """
        Mutate the gene.

        Parameters:
        learnable_parameters (dict): A dictionary of learnable parameters.
        """
        for param in self.parameters:
            value = learnable_parameters.get(param)
            if value is not None and isinstance(value, list):
                setattr(self, param, random.choice(value))

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

    @property
    def get_attribute_dict(self):
        return self.parameters
