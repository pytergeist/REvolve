from functools import partial


class Operations:
    def register(self, function, *args, **kwargs):
        func = partial(function, *args, **kwargs)
        func.name = function.__name__.split("_")[-1]

        setattr(self, func.name, func)

    def get_operations(self):
        return list(self.__dict__.keys())
