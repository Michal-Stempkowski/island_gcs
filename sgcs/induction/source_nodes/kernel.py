from sgcs.induction.source_nodes.source_node import SourceNode


def tag():
    return 'kernel'

cyk_kernel = SourceNode('default_cyk_kernel', """
// Compilation_time: __sn_timestamp_tag__

__global__ void __sn_absolute_identifier_tag__(int* prefs, int* sentence, int* table, int* table_header)
{
    const int thread_id = threadIdx.x;
    table_header[thread_id] = thread_id * 10000;
}
""")
