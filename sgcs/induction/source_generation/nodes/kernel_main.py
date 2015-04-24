from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'kernel_main'

kernel_main = SourceNode('default_cyk_kernel_main', """
CCM void __sn_absolute_identifier_tag__(int* prefs, int* sentence, int* table,
    int* table_header, const int thread_id, const int block_id)
{
    const int number_of_blocks = preferences(block_id, AT).get(prefs, preferences::number_of_blocks);
    const int number_of_threads = preferences(block_id, AT).get(prefs, preferences::number_of_threads);
}
""",
                        dependencies=['cuda_helper', 'preferences', 'cyk_table'])
