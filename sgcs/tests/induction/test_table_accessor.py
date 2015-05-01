from unittest import TestCase
from hamcrest import *
from sgcs.induction.table_accessor import TableAccessor
import numpy as np


class TestTableAccessor(TestCase):
    def setUp(self):
        self.default_table = TableAccessor(2, 2, 2)

    def test_valid_total_size(self):
        assert_that(self.default_table.total_size, equal_to(8))

        empty_table = TableAccessor(0, 0, 0)
        assert_that(empty_table.total_size, equal_to(0))
        assert_that(empty_table.table, is_(None))

    def test_valid_fields_should_be_accessible(self):
        index_gen = ((x, y, z) for x in range(2) for y in range(2) for z in range(2))

        for coord in index_gen:
            assert_that(self.default_table.get(*coord), equal_to(0))

    def test_invalid_access_should_cause_an_exception(self):
        invalid_locations = ((2, 0, 0), (0, 2, 0), (0, 0, 2))

        for invalid in invalid_locations:
            assert_that(calling(lambda: self.default_table.get(*invalid)), raises(IndexError))

    def test_invalid_size_table_should_fail_on_set_raw(self):
        invalid_table = self.default_table._create_empty_int32_table(2)
        assert_that(calling(lambda: self.default_table.set_raw_table(invalid_table)), raises(IndexError))

    def test_set_should_work(self):
        for x, y, z in ((x, y, z) for x in range(2) for y in range(2) for z in range(2)):
            value = x * 4 + y * 2 + z
            self.default_table.set(x, y, z, value)
            assert_that(self.default_table.table[value], equal_to(value))

    def test_invalid_set_should_raise_exeption(self):
        assert_that(calling(lambda: self.default_table.set(0, 0, 15)), raises(IndexError))
        assert_that(calling(lambda: self.default_table.set(0, 0, 0, 0, 15)), raises(IndexError))
        assert_that(calling(lambda: self.default_table.set(0, 6, 0, 15)), raises(IndexError))

    def test_valid_table_should_success_on_set_raw(self):
        valid_table = self.default_table._create_empty_int32_table(8)
        valid_table[1] = 4

        self.default_table.set_raw_table(valid_table)
        assert_that(self.default_table.get_raw_table()[1], equal_to(4))

    def test_set_raw_with_no_param_should_reset_table(self):
        valid_table = self.default_table._create_empty_int32_table(8)
        valid_table[1] = 4

        self.default_table.set_raw_table(valid_table)
        self.default_table.set_raw_table()
        assert_that(self.default_table.get_raw_table()[1], equal_to(0))

    def test_no_autocreate_should_stop_table_from_being_created(self):
        table = TableAccessor(2, 2, 2, autocreate=False)
        assert_that(table.get_raw_table(), equal_to(None))

    def test_set_and_get_shaped_table_should_work_properly(self):
        init_table = np.array([
            [
                [0, 1],
                [2, 3]
            ],
            [
                [4, 5],
                [6, 7]
            ]
        ])

        self.default_table.set_shaped_raw_table(init_table)

        for i in range(8):
            assert_that(self.default_table.get(i // 4, (i // 2) % 2, i % 2), equal_to(i))

        self.assertTrue(np.array_equal(self.default_table.get_shaped_raw_table(), init_table))