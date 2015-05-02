from unittest import TestCase
from sgcs.induction.source_generation.source_node import SourceNode
from sgcs.tests import common


kernel_main = SourceNode('default_cyk_kernel_main', """
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

    def test_is_cuda_working(self):
        self.sut.source_code_schema.kernel_main = kernel_main
        self.sut.additional_data['test_code'] = 'rules_by_right[0] = 4;'

        self.sut.run_cyk(self.sentence)

        self.assertEquals(4, self.sut.get_table_accessor('rules_by_right').get(0, 0, 0, 0))