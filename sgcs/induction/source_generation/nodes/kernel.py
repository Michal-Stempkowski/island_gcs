from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'kernel'

# language=C++
cyk_kernel = SourceNode('default_cyk_kernel', """
extern "C" {
__global__ void __sn_absolute_identifier_tag__(int* prefs, int* sentence, int* table, int* table_header)
{
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    kernel_main(prefs, sentence, table, table_header, thread_id, block_id);
}
}

// Compilation_time: __sn_timestamp_tag__
""",
                        dependencies=['cuda_helper', 'kernel_main'])
