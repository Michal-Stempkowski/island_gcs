from sgcs.induction.cyk_cuda_runner import CykCudaRunner
from sgcs.induction.source_generation.cyk_source_schema import CykSourceSchema


class DynamicSettingsClass(object):
    def field_list(self):
        return vars(self)

    @staticmethod
    def virtual_fields(self):
        return []


class WorldSettings(DynamicSettingsClass):
    def __init__(self, num_of_blocks, num_of_threads, max_number_of_terminal_symbols,
                 max_number_of_non_terminal_symbols):
        self.number_of_blocks = num_of_blocks
        self.number_of_threads = num_of_threads
        self.max_number_of_terminal_symbols = max_number_of_terminal_symbols
        self.max_number_of_non_terminal_symbols = max_number_of_non_terminal_symbols


class IslandSettings(DynamicSettingsClass):
    def __init__(self, sentence_length, max_symbols_in_cell):
        self.sentence_length = sentence_length
        self.max_symbols_in_cell = max_symbols_in_cell


def get_sut():
    # print('Hello world!')
    world_settings = WorldSettings(3, 32, 32, 32)
    island_settings = [IslandSettings(16, 16) for _ in range(world_settings.number_of_blocks)]
    source_code_schema = CykSourceSchema()

    test = CykCudaRunner(world_settings, island_settings, source_code_schema)

    return test
    # sentence = [1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5]
    # test.run_cyk(sentence)
    # test.run_test()