from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cuda_post_mortem'

cuda_post_mortem = SourceNode('cuda_post_mortem', """
////CPP
#if !defined CUDA_POST_MORTEM_H
#define CUDA_POST_MORTEM_H

enum post_mortem_error : int
{
    internal_error = -1,
    cyk_row_fill_error = 2 << 0
};

__shared__ bool working_properly;

CCM void throw_post_mortem_error(local_data *thread_data,
    post_mortem_error error_code, const char* source_code_localization)
{
    int index = generate_absolute_index(
        thread_data->block_id, thread_data->number_of_blocks,
        thread_data->thread_id, thread_data->number_of_threads
    );

    auto result = table_get(thread_data->error_table, index);

    if (result > post_mortem_error::internal_error)
    {
        table_set(thread_data->error_table, index, result | error_code);
    }
    else
    {
        thread_data->error_table[0] = post_mortem_error::internal_error;
    }

    log_debug("post_mortem error (%d) occurred at %s!\\n", error_code, source_code_localization);

    working_properly = false;
}

CCM void init_post_mortem(const int thread_id)
{
    if (thread_id == 0)
    {
        working_properly = true;
    }
    __syncthreads();
}

#endif
""",
                              dependencies=['cuda_helper', 'local_data'])
