from sgcs.induction.cyk_runner import CykRunner
from sgcs.induction.source_generation.cyk_source_schema import CykSourceSchema


class WorldSettings(object):
    def __init__(self, num_of_blocks, num_of_threads):
        self.num_of_blocks = num_of_blocks
        self.num_of_threads = num_of_threads


class IslandSettings(object):
    def __init__(self, sentence_length, max_alphabet_size, max_symbols_in_cell):
        self.sentence_length = sentence_length
        self.max_alphabet_size = max_alphabet_size
        self.max_symbols_in_cell = max_symbols_in_cell


def get_sut():
    # print('Hello world!')
    world_settings = WorldSettings(3, 32)
    island_settings = [IslandSettings(16, 32, 16) for _ in range(world_settings.num_of_blocks)]
    source_code_schema = CykSourceSchema()

    test = CykRunner(world_settings, island_settings, source_code_schema)

    return test
    # sentence = [1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5]
    # test.run_cyk(sentence)
    # test.run_test()