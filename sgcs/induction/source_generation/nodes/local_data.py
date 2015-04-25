from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'local_data'

local_data = SourceNode('default_local_data', """
////CPP
class local_data
{
public:
    CCM local_data(
        const int block_id_, const int number_of_blocks_,
        const int thread_id_, const int number_of_threads_,
        int* error_table_) :
            block_id(block_id_),
            number_of_blocks(number_of_blocks_),
            thread_id(thread_id_),
            number_of_threads(number_of_threads_),
            error_table(error_table_)
    {

    }

    const int block_id;
    const int number_of_blocks;
    const int thread_id;
    const int number_of_threads;
    int* error_table;
};
""",
                        dependencies=['cuda_helper'])
