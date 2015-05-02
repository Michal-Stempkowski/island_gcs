import numpy as np
from unittest import TestCase
from sgcs.induction.source_generation.source_node import SourceNode
from sgcs.tests import common

kernel_main = SourceNode('test_cyk_rules_kernel_main', """
////CPP test kernel
CCM void @@sn_absolute_identifier_tag@@(int* sentence,
    @@sg_repeat(vals(kernel_param_names), begin(int* ), separator(, int* ))@@, const int thread_id, const int block_id)
{
    const int number_of_blocks = preferences(block_id, AT).get(prefs, preferences::number_of_blocks);
    const int number_of_threads = preferences(block_id, AT).get(prefs,  preferences::number_of_threads);

    local_data thread_data(block_id, number_of_blocks, thread_id, number_of_threads, error_table);
    cyk_rules rules(rules_by_right, rules_by_right_header, &thread_data, prefs);

    @@sg_insert(content_id(test_code))@@
}
""",
                        dependencies=[
                            'cuda_helper',
                            'preferences',
                            'local_data',
                            'cyk_rules'])


class TestCykRules(TestCase):
    def setUp(self):
        self.sut = common.get_sut()
        self.sentence = [1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5]

        self.sut.source_code_schema.kernel_main = kernel_main
        self.rules_header = self.sut.get_table_accessor('rules_by_right_header')
        self.rules = self.sut.get_table_accessor('rules_by_right')
        self.test_get_rule_by_right_side_code =\
            '''////CPP
            if (thread_data.block_id == 0 && thread_data.thread_id == 0)
             {
                rules_by_right[0] = rules.get_rule_by_right_side(value, value, 0, AT);
             }'''

    def test_is_rules_get_rule_by_right_side_for_non_terminals_working(self):
        non_terminal_symbol = 3
        self.rules_header.set(0, non_terminal_symbol, non_terminal_symbol, 1)
        self.rules.set(0, non_terminal_symbol, non_terminal_symbol, 0, 4)

        self.sut.additional_data['test_code'] =\
            '''
                ////CPP
                int value = {symbol};
            '''.format(symbol=non_terminal_symbol) + self.test_get_rule_by_right_side_code
        self.assertEquals(0, self.rules.get(0, 0, 0, 0))

        self.sut.run_cyk(self.sentence)

        self.assertEquals(4, self.rules.get(0, 0, 0, 0))

    def test_is_rules_get_rule_by_right_side_for_terminals_working(self):
        terminal_symbol = 3
        masked_value = -2 ** 31 + terminal_symbol
        real_index = self.sut.max_number_of_non_terminal_symbols + terminal_symbol

        self.rules_header.set(0, real_index, real_index, 1)
        self.rules.set(0, real_index, real_index, 0, 4)

        self.sut.additional_data['test_code'] =\
            '''
                ////CPP
                int value = {symbol};
            '''.format(symbol=masked_value) + self.test_get_rule_by_right_side_code

        self.assertEquals(0, self.rules.get(0, 0, 0, 0))

        self.sut.run_cyk(self.sentence)

        self.assertEquals(4, self.rules.get(0, 0, 0, 0))

    def test_get_number_of_rules(self):
        non_terminal_symbol = 3
        self.rules_header.set(0, non_terminal_symbol, non_terminal_symbol, 1)
        self.rules.set(0, non_terminal_symbol, non_terminal_symbol, 0, 4)

        self.sut.additional_data['test_code'] =\
            '''
                ////CPP
                int value = {symbol};
            '''.format(symbol=non_terminal_symbol) +\
            '''////CPP
            if (thread_data.block_id == 0 && thread_data.thread_id == 0)
             {
                rules_by_right[0] = rules.get_number_of_rules(value, value);
                rules_by_right[1] = rules.get_number_of_rules(value + 1, value + 1);
             }'''
        self.assertEquals(0, self.rules.get(0, 0, 0, 0))
        self.assertEquals(0, self.rules.get(0, 0, 0, 1))

        self.sut.run_cyk(self.sentence)

        self.assertEquals(1, self.rules.get(0, 0, 0, 0))
        self.assertEquals(0, self.rules.get(0, 0, 0, 1))