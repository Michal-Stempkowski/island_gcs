from unittest import TestCase
from pycuda.compiler import SourceModule
from sgcs.tests import common
import numpy as np
import pycuda.driver as cuda


class TestCykRunner(TestCase):
    def setUp(self):
        self.sut = common.get_sut()
        self.sentence = [1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5]

    def test_is_cuda_working(self):
        a = np.array([x for x in range(16)]).reshape((4, 4))
        # a = np.array.randn(4, 4)
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
        self.assertTrue(np.array_equal(a * 2, a_doubled))

    def test_is_cyk_working(self):
        self.sut.run_cyk(self.sentence)

        table_header = self.sut.get_table_accessor('table_header').get_raw_table()
        table = self.sut.get_table_accessor('table').get_raw_table()

        self.assertEquals(768, len(table_header))
        self.assertEquals(12288, len(table))

        # row_0 = np.extract((table > 0), table).tolist()
        # print(row_0)
        # self.assertTrue(np.array_equal(row_0, np.concatenate((self.sentence, self.sentence, self.sentence))))