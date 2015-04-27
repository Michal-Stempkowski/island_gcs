import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import pycuda.autoinit
_ = pycuda.autoinit

from sgcs.induction.source_generation.nodes import kernel


class Data(object):
    def __init__(self, name, data_provider, wrapper):
        self.name = name
        self.data_provider = data_provider
        self.wrapper = wrapper

    def __call__(self):
        return self.wrapper(self.data_provider())


class CykDataCollector(object):
    def __init__(self, *tuples):
        self.cuda_type = 'int*'
        self.data = {name: Data(name, data_provider, par_type) for (par_type, name, data_provider) in tuples}

    def headers(self):
        return sorted(self.data.keys())

    def get_data_packages(self):
        return [(self.data[name])() for name in self.headers()]


class CykRunner:
    def __init__(self, world_settings_schema, island_settings_schema, source_code_schema):
        self.preferences_headers,  self.preferences_table = self.generate_preferences_table(
            world_settings_schema, island_settings_schema)
        self.error_table = self.generate_post_mortem_error_table(world_settings_schema)
        self.source_code_schema = source_code_schema
        self.module = None
        self.func = lambda *args, block, grid: None
        self.cyk_block = None
        self.cyk_header_block = None

        self.data_collector =\
            CykDataCollector(
                (cuda.In, 'prefs', lambda: self.preferences_table),
                (cuda.InOut, 'error_table', lambda: self.error_table),
                (cuda.InOut, 'table', lambda: self.cyk_block),
                (cuda.InOut, 'table_header', lambda: self.cyk_header_block)
                # (cuda.InOut, 'cyk_terminal')
            )

    def compile_kernel_if_necessary(self):
        if self.source_code_schema.requires_update:
            additional_data = dict(
                preferences_headers=self.preferences_headers,
                kernel_param_names=self.data_collector.headers())
            self.module = SourceModule(self.source_code_schema.generate_schema(additional_data), no_extern_c=1)
            self.func = self.module.get_function(kernel.tag())
            self.source_code_schema.requires_update = False

    @staticmethod
    def _dict_union(d1, d2):
        result = dict()
        result.update(d1)
        result.update(d2)

        return result

    @classmethod
    def generate_preferences_table(cls, world_settings, island_settings):
        joined_settings = [cls._dict_union(world_settings.field_list(), settings.field_list())
                           for settings in island_settings]
        headers = sorted(joined_settings[0].keys())

        prefs = np.array([
            [
                settings[x] for x in headers
            ] for settings in joined_settings
        ])
        return headers, prefs.reshape(1, len(prefs) * len(prefs[0])).astype(np.int32)[0]

    @staticmethod
    def generate_post_mortem_error_table(world_settings):
        return np.zeros(world_settings.number_of_blocks * world_settings.number_of_threads, dtype=np.int32)

    @property
    def number_of_blocks(self):
        return int(self.preferences_table[self.preferences_headers.index('number_of_blocks')])

    @property
    def number_of_threads(self):
        return int(self.preferences_table[self.preferences_headers.index('number_of_threads')])

    @property
    def max_symbols_in_cell(self):
        return int(self.preferences_table[self.preferences_headers.index('max_symbols_in_cell')])

    @staticmethod
    def create_empty_int32_table(size):
        return np.zeros((1, size)).astype(np.int32)[0]

    def generate_cyk_header_block(self, sentence):
        return self.create_empty_int32_table(self.number_of_blocks * len(sentence) * len(sentence))

    def generate_cyk_block(self, header_size):
        return self.create_empty_int32_table(header_size * self.max_symbols_in_cell)

    def run_cyk(self, sentence):
        self.compile_kernel_if_necessary()

        self.cyk_header_block = self.generate_cyk_header_block(sentence)
        self.cyk_block = self.generate_cyk_block(len(self.cyk_header_block))

        self.func(
            cuda.In(np.array(sentence).astype(np.int32)),
            *self.data_collector.get_data_packages(),
            block=(self.number_of_threads, 1, 1),
            grid=(self.number_of_blocks, 1, 1))

        if np.any(self.error_table != 0):
            print(self.error_table)
            raise RuntimeError()