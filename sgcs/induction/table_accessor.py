import numpy as np
from functools import reduce


class TableAccessor(object):
    def __init__(self, *dimensions, data=None, autocreate=True):
        self.autocreate = autocreate
        self.total_size = 0
        self._dimensions = []
        self.dimensions = dimensions

        self.table = None
        self.set_raw_table(data)

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._dimensions = value
        self.total_size = reduce(
            lambda acc, dim: acc * dim, self._dimensions,
            1 if len(self._dimensions) > 0 else 0)

    def _calculate_index(self, *params):
        # noinspection PyTypeChecker
        if self.table is None or self.total_size != len(self.table):
            # noinspection PyTypeChecker
            raise IndexError('Malformed table (expected size {0}, got {1}'
                             .format(self.total_size, len(self.table)))

        if len(params) != len(self._dimensions):
            raise IndexError('Dimension {0} expected, {1} provided!'
                             .format(len(self._dimensions), len(params)))

        index = 0

        for i in range(len(params)):
            param = params[i]
            dim = self._dimensions[i]

            if param >= dim:
                raise IndexError('Param {0} out of bounds (required={1}, dim_size={2}'
                                 .format(i, param, dim))

            index += param
            index *= (self._dimensions[i + 1] if i + 1 < len(self._dimensions) else 1)

        return index

    @staticmethod
    def _create_empty_int32_table(size):
        return np.zeros((1, size)).astype(np.int32)[0]

    def get(self, *params):
        index = self._calculate_index(*params)
        return self.table[index]

    def set(self, *params):
        index = self._calculate_index(*params[:-1])
        self.table[index] = params[-1]

    def get_raw_table(self):
        return self.table

    def set_raw_table(self, new_table=None):
        if new_table is not None:
            if len(new_table) != self.total_size:
                raise IndexError('Table length = {0} and provided initializer length = {1} does not match!'
                                 .format(self.total_size, len(new_table)))
            self.table = new_table
        elif self.total_size > 0 and self.autocreate:
            self.table = self._create_empty_int32_table(self.total_size)

    def get_shaped_raw_table(self):
        return self.get_raw_table().reshape(tuple(self._dimensions))

    def set_shaped_raw_table(self, new_table):
        self.set_raw_table(new_table.reshape((self.total_size,)))