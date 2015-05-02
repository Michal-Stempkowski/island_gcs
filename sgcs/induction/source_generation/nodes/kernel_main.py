from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'kernel_main'

kernel_main = SourceNode('default_cyk_kernel_main', """
////CPP
CCM void @@sn_absolute_identifier_tag@@(int* sentence,
    @@sg_repeat(vals(kernel_param_names), begin(int* ), separator(, int* ))@@, const int thread_id, const int block_id)
{
    const int number_of_blocks = preferences(block_id, AT).get(prefs, preferences::number_of_blocks);
    const int number_of_threads = preferences(block_id, AT).get(prefs,  preferences::number_of_threads);

    local_data thread_data(block_id, number_of_blocks, thread_id, number_of_threads, error_table);
    cyk_rules rules(rules_by_right, rules_by_right_header, &thread_data, prefs);

    cyk_table cyk(block_id, AT, prefs, table, table_header);

    int row = 0;
    int col = cyk.get_starting_col_coord_for_thread(thread_id);

    for (int i = 0; working_properly && i < cyk.size(); ++i)
    {
        //// throw_post_mortem_error(&thread_data, post_mortem_error::test_error, AT);
        //log_debug("%d\\n", col);
        if (row < 0 || col < 0 ||
            row >= cyk.size() || col >= cyk.size())
        {

        }
        else if (row == 0)
        {
            cyk.fill_first_row(sentence, row, col, &thread_data);
        }
        else
        {

        }

        int old_row = row;
        int old_col = col;
        row = cyk.get_row_coord_for_thread(thread_id, old_row, old_col);
        col = cyk.get_col_coord_for_thread(thread_id, old_row, old_col);
    }
}
""",
                        dependencies=[
                            'cuda_helper',
                            'cuda_post_mortem',
                            'preferences',
                            'cyk_table',
                            'local_data',
                            'cyk_rules'])
