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
        self.assertEquals(768, len(self.sut.cyk_header_block))
        self.assertEquals(12288, len(self.sut.cyk_block))