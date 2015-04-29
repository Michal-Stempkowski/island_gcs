from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'kernel'

# language=C++
cyk_kernel = SourceNode('default_cyk_kernel', """
////CPP
extern "C" {

__global__ void @@sn_absolute_identifier_tag@@(int* sentence,
    @@sg_repeat(vals(kernel_param_names), begin(int* ), separator(, int* ))@@)
{
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    init_post_mortem(thread_id);

    kernel_main(sentence, @@sg_repeat(vals(kernel_param_names), separator(, ))@@, thread_id, block_id);
}
}

// Compilation_time: @@sn_timestamp_tag@@
""",
                        dependencies=['cuda_helper', 'kernel_main'])
