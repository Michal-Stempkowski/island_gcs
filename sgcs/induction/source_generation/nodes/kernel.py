from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'kernel'

# language=C++
cyk_kernel = SourceNode('default_cyk_kernel', """
////CPP
extern "C" {

__global__ void __sn_absolute_identifier_tag__(int* prefs, int* sentence, int* table, int* table_header, int* error_table)
{
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    init_post_mortem(thread_id);

    kernel_main(prefs, sentence, table, table_header, error_table, thread_id, block_id);
}
}

// Compilation_time: __sn_timestamp_tag__
""",
                        dependencies=['cuda_helper', 'kernel_main'])
