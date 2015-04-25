import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import pycuda.autoinit
_ = pycuda.autoinit

from sgcs.induction.source_generation.nodes import kernel


class CykRunner:
    def __init__(self, world_settings_schema, island_settings_schema, source_code_schema):
        self.preferences_table = self.generate_preferences_table(
            world_settings_schema, island_settings_schema)
        self.source_code_schema = source_code_schema
        self.module = None
        self.func = lambda _1, _2, _3, _4, block, grid: None
        self.cyk_block = None
        self.cyk_header_block = None

    def compile_kernel_if_necessary(self):
        if self.source_code_schema.requires_update:
            self.module = SourceModule(self.source_code_schema.generate_schema(), no_extern_c=1)
            self.func = self.module.get_function(kernel.tag())
            self.source_code_schema.requires_update = False

    @staticmethod
    def generate_preferences_table(world_settings, island_settings):
        prefs = np.array([
            [
                settings.sentence_length,
                settings.max_alphabet_size,
                settings.max_symbols_in_cell,
                world_settings.num_of_blocks,
                world_settings.num_of_threads
            ] for settings in island_settings
        ])
        return prefs.reshape(1, len(prefs) * len(prefs[0])).astype(np.int32)[0]

    @property
    def num_of_blocks(self):
        return int(self.preferences_table[3])

    @property
    def num_of_threads(self):
        return int(self.preferences_table[4])

    @property
    def max_symbols_in_cell(self):
        return int(self.preferences_table[2])

    @staticmethod
    def create_empty_int32_table(size):
        return np.zeros((1, size)).astype(np.int32)[0]

    def generate_cyk_header_block(self, sentence):
        return self.create_empty_int32_table(self.num_of_blocks * len(sentence) * len(sentence))

    def generate_cyk_block(self, header_size):
        return self.create_empty_int32_table(header_size * self.max_symbols_in_cell)

    def run_cyk(self, sentence):
        self.compile_kernel_if_necessary()

        self.cyk_header_block = self.generate_cyk_header_block(sentence)
        self.cyk_block = self.generate_cyk_block(len(self.cyk_header_block))

        self.func(
            cuda.In(self.preferences_table),
            cuda.In(np.array(sentence).astype(np.int32)),
            cuda.InOut(self.cyk_block),
            cuda.InOut(self.cyk_header_block),
            block=(self.num_of_threads, 1, 1),
            grid=(self.num_of_blocks, 1, 1))