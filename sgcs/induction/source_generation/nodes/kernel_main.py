from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'kernel_main'

kernel_main = SourceNode('default_cyk_kernel_main', """
CCM void __sn_absolute_identifier_tag__(int* prefs, int* sentence, int* table,
    int* table_header, const int thread_id, const int block_id)
{
    const int number_of_blocks = preferences(block_id, AT).get(prefs, preferences::number_of_blocks);
    const int number_of_threads = preferences(block_id, AT).get(prefs,  preferences::number_of_threads);

    cyk_table cyk(block_id, AT, prefs, table, table_header);

    int row = 0;
    int col = cyk.get_starting_col_coord_for_thread(thread_id);

    for (int i = 0; i < cyk.size(); ++i)
    {
        if (row < 0 || col < 0)
        {

        }
        else if (row == 0)
        {
            auto symbol = table_get(sentence, generate_absolute_index(
                col, cyk.size()));

            auto result = cyk.add_symbol_to_cell(row, col, symbol);
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
                        dependencies=['cuda_helper', 'preferences', 'cyk_table'])
