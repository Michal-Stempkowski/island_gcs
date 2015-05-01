import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import pycuda.autoinit
from sgcs.induction.table_accessor import TableAccessor

_ = pycuda.autoinit

from sgcs.induction.source_generation.nodes import kernel


class Data(object):
    def __init__(self, name, wrapper, table_accessor):
        self.name = name
        self.wrapper = wrapper
        self.table_accessor = table_accessor

    def __call__(self):
        return self.wrapper(self.table_accessor.get_raw_table())

    def get(self):
        return self.table_accessor


class CykDataCollector(object):
    def __init__(self, *tuples):
        self.cuda_type = 'int*'
        self.data = {name: Data(name, par_type, table_accessor)
                     for (par_type, name, table_accessor) in tuples}

    def headers(self):
        return sorted(self.data.keys())

    def get_data_packages(self):
        return [(self.data[name])() for name in self.headers()]


class CykCudaBuilder(object):
    def __init__(self):
        pass

    @staticmethod
    def init_table(cuda_runner, table_name, dimensions):
        table_accessor = cuda_runner.get_table_accessor(table_name)
        table_accessor.dimensions = dimensions
        table_accessor.set_raw_table()
        return table_accessor

    @staticmethod
    def create_table_accessors(cuda_runner, world_settings_schema):
        return CykDataCollector(
            (cuda.In, 'prefs',
             TableAccessor(
                 len(cuda_runner.preferences_headers),
                 world_settings_schema.number_of_blocks,
                 data=cuda_runner.preferences_table
             )),
            (cuda.InOut, 'error_table',
             TableAccessor(
                 world_settings_schema.number_of_blocks,
                 world_settings_schema.number_of_threads
             )),
            (cuda.InOut, 'table', TableAccessor()),
            (cuda.InOut, 'table_header', TableAccessor()),
            (cuda.InOut, 'rules_by_right',
             TableAccessor(
                 cuda_runner.number_of_blocks,
                 cuda_runner.max_alphabet_size,
                 cuda_runner.max_alphabet_size
             )),
            (cuda.InOut, 'rules_by_right_header',
             TableAccessor(
                 cuda_runner.number_of_blocks,
                 cuda_runner.max_alphabet_size,
                 cuda_runner.max_alphabet_size,
                 cuda_runner.max_symbols_in_cell
             ))
        )

    @staticmethod
    def get_additional_data(cuda_runner):
        additional_preferences = [
            ('alphabet_size', 'preferences[get_index(max_number_of_terminal_symbols)] + ' +
             'preferences[get_index(max_number_of_non_terminal_symbols)]'),
            (0,)
        ]
        additional_data = dict(
            preferences_headers=cuda_runner.preferences_headers,
            additional_preferences=additional_preferences,
            additional_preferences_headers=[pref[0] for pref in filter(lambda p: len(p) > 1,
                                                                       additional_preferences)],
            kernel_param_names=cuda_runner.data_collector.headers())
        return additional_data

    @staticmethod
    def print_all_cuda_tables(cuda_runner, error_table):
        print(error_table)
        for name, data in cuda_runner.data_collector.data.items():
            print(name)
            print(data.get().get_raw_table())
            print(len(data.get().get_raw_table()))

    @classmethod
    def perform_cuda_cleanup_operations(cls, cuda_runner):
        error_table = cuda_runner.get_table_accessor('error_table').get_raw_table()
        if np.any(error_table != 0):
            cls.print_all_cuda_tables(cuda_runner, error_table)
            raise RuntimeError()

    @classmethod
    def perform_startup_cuda_operations(cls, cuda_runner, sentence):
        table_header = cls.init_table(cuda_runner, 'table_header',
                                      [cuda_runner.number_of_blocks, len(sentence), len(sentence)])
        cls.init_table(cuda_runner, 'table', table_header.dimensions[:] + [cuda_runner.max_symbols_in_cell])

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
                settings[x] for x in headers # if not x.startswith('_')
            ] for settings in joined_settings
        ])
        return headers, prefs.reshape(1, len(prefs) * len(prefs[0])).astype(np.int32)[0]


class CykCudaRunner:
    def __init__(self, world_settings_schema, island_settings_schema, source_code_schema, cuda_builder=CykCudaBuilder()):
        self.cuda_builder = cuda_builder
        self.preferences_headers,  self.preferences_table = self.cuda_builder.generate_preferences_table(
            world_settings_schema, island_settings_schema)
        self.source_code_schema = source_code_schema
        self.module = None
        self.func = lambda *args, block, grid: None

        self.data_collector = self.cuda_builder.create_table_accessors(self, world_settings_schema)

    def get_table_accessor(self, name):
        return self.data_collector.data[name].get()

    def compile_kernel_if_necessary(self):
        if self.source_code_schema.requires_update:
            self.module = SourceModule(self.source_code_schema.generate_schema(
                self.cuda_builder.get_additional_data(self)), no_extern_c=1)
            self.func = self.module.get_function(kernel.tag())
            self.source_code_schema.requires_update = False

    @property
    def number_of_blocks(self):
        return int(self.preferences_table[self.preferences_headers.index('number_of_blocks')])

    @property
    def number_of_threads(self):
        return int(self.preferences_table[self.preferences_headers.index('number_of_threads')])

    @property
    def max_symbols_in_cell(self):
        return int(self.preferences_table[self.preferences_headers.index('max_symbols_in_cell')])

    @property
    def max_number_of_terminal_symbols(self):
        return int(self.preferences_table[self.preferences_headers.index('max_number_of_terminal_symbols')])

    @property
    def max_number_of_non_terminal_symbols(self):
        return int(self.preferences_table[self.preferences_headers.index('max_number_of_non_terminal_symbols')])

    @property
    def max_alphabet_size(self):
        return self.max_number_of_terminal_symbols + self.max_number_of_non_terminal_symbols

    def get_cuda_block_shape(self):
        return self.number_of_threads, 1, 1

    def get_cuda_grid_shape(self):
        return self.number_of_blocks, 1, 1

    def run_cyk(self, sentence):
        self.compile_kernel_if_necessary()

        self.cuda_builder.perform_startup_cuda_operations(self, sentence)

        self.func(
            cuda.In(np.array(sentence).astype(np.int32)),
            *self.data_collector.get_data_packages(),
            block=self.get_cuda_block_shape(),
            grid=self.get_cuda_grid_shape())

        self.cuda_builder.perform_cuda_cleanup_operations(self)