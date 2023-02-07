from __future__ import annotations
import random
from abc import ABC, abstractmethod


class BaseGene(ABC):
    def __init__(self, gene_type: str, **kwargs):
        self.name = "gene"
        self.gene_type = gene_type
        self.parameters = kwargs

    @abstractmethod
    def _validate_params(self):  # pragma: no cover
        """
        Placeholder method to validate parameters, can be overridden by subclasses
        """
        pass

    def mutate(self, learnable_parameters: dict):
        """CHECK THIS MUTATE METHOD IS WORKING"""

        for param, value in self.parameters.items():
            if isinstance(learnable_parameters.get(param), list):
                setattr(self, param, random.choice(learnable_parameters.get(param)))

    def get_attributes(self):
        return list(self.parameters.values())

    @property
    def get_attribute_names(self):
        return list(self.parameters.keys())
