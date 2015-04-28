from unittest import TestCase
from pycuda.compiler import SourceModule
from sgcs.induction.source_generation.nodes import preferences
from sgcs.induction.source_generation.nodes.cuda_helper import cuda_helper
from sgcs.tests import common
import numpy as np
import pycuda.driver as cuda


class TestCodeGeneration(TestCase):
    def setUp(self):
        self.sut = common.get_sut()
        self.sentence = [1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5]

    def test_is_cuda_working(self):
        res = preferences.preferences.link(
            {'cuda_helper': cuda_helper},
            preferences.tag(),
            {'preferences_headers': ["a", "b", "c"],
                'preferences_conditions': ['opt > enum_size'],
                'preferences_sample_logic': [('true', 1),
                                             ('true', 2),
                                             (3,)]})

        print(res)
        self.assertTrue(
            """enum option : int
    {
        a,
        b,
        c,
        enum_size,                          // DO NOT use it as enum, it represents size of this enum
        ////
        enum_size_with_additionals          // DO NOT use it as enum, it represents size of this enum
    };""".replace(' ', '') in res.replace(' ', ''))