from unittest import TestCase


class TestSanity(TestCase):
    def setUp(self):
        pass

    def test_am_i_sane(self):
        self.assertEquals(True, True)