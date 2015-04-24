import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


class CykRunner:
    def __init__(self, world_settings_schema, island_settings_schema, source_code):
        self.preferences_table = self.generate_preferences_table(
            world_settings_schema, island_settings_schema)
        self.module = SourceModule(source_code)
        self.func = self.module.get_function('cyk_kernel')

    def run_test(self):
        a = np.random.randn(4, 4)
        a = a.astype(np.float32)
        a_gpu = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_gpu, a)

        mod = SourceModule("""
          __global__ void doublify(float *a)
          {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] *= 2;
          }
          """)

        func = mod.get_function("doublify")
        func(a_gpu, block=(4, 4, 1))

        a_doubled = np.empty_like(a)
        cuda.memcpy_dtoh(a_doubled, a_gpu)
        print(a_doubled)
        print(a)

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
        cyk_header_block = self.generate_cyk_header_block(sentence)
        cyk_block = self.generate_cyk_block(len(cyk_header_block))

        self.func(
            cuda.In(self.preferences_table),
            cuda.In(np.array(sentence).astype(np.int32)[0]),
            cuda.InOut(cyk_block),
            cuda.InOut(cyk_header_block),
            block=(self.num_of_threads, 1, 1))

        print(self.preferences_table)
        print(cyk_header_block)
        print(len(cyk_header_block))
        print(cyk_block[:32])
        print(len(cyk_block))
        print(cuda.get_version())