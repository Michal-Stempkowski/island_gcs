from sgcs.induction.source_nodes.source_node import SourceNode


def tag():
    return 'kernel'

cyk_kernel = SourceNode('kernel', """
__global__ void cyk_kernel(int* prefs, int* sentence, int* table, int* table_header)
{
    const int thread_id = threadIdx.x;
    table_header[thread_id] = thread_id;
}
""")
