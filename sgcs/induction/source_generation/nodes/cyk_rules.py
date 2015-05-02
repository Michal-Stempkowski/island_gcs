from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cyk_rules'

cyk_rules = SourceNode('default_cyk_rules', """
////CPP
class cyk_rules
{
public:
    static const int terminal_mask = 2 << 31;
    static const int terminal_mask_negated = ~(2 << 31);

    CCM cyk_rules(int* rules_by_right_, int* rules_by_right_header_, local_data *thread_data_, int* prefs_);
    CCM int get_rule_by_right_side(int left_symbol, int right_symbol, int pos,
        const char* source_code_localization);

    CCM bool is_terminal(int symbol);
    CCM int translate_terminal_to_table_coord(int symbol);
    CCM int translate_if_terminal(int symbol);

private:
    int* rules_by_right;
    int* rules_by_right_header;
    local_data* thread_data;
    int* prefs;
};

CCM cyk_rules::cyk_rules(int* rules_by_right_, int* rules_by_right_header_,
    local_data *thread_data_, int* prefs_) :

    rules_by_right(rules_by_right_),
    thread_data(thread_data_),
    prefs(prefs_),
    rules_by_right_header(rules_by_right_header_)
{

}

CCM int cyk_rules::get_rule_by_right_side(int left_symbol, int right_symbol, int pos,
    const char* source_code_localization)
{
    ////left_symbol = (left_symbol & terminal_mask) ? (left_symbol & terminal_mask_negated) : left_symbol;

    left_symbol = translate_if_terminal(left_symbol);
    right_symbol = translate_if_terminal(right_symbol);

    const int alphabet_size = preferences(thread_data->block_id, AT).get(prefs, preferences::alphabet_size);
    const int rules_in_cell = table_get(rules_by_right_header, generate_absolute_index(
        thread_data->block_id, thread_data->number_of_blocks,
        left_symbol, alphabet_size,
        right_symbol, alphabet_size));

    if (rules_in_cell < error::no_errors_occured)
    {
        throw_post_mortem_error(thread_data,
            post_mortem_error::cyk_fatal_index_out_of_bounds, source_code_localization);
    }

    return table_get(rules_by_right, generate_absolute_index(
        thread_data->block_id, thread_data->number_of_blocks,
        left_symbol, alphabet_size,
        right_symbol, alphabet_size,
        pos, rules_in_cell));
}

CCM bool cyk_rules::is_terminal(int symbol)
{
    return symbol & terminal_mask;
}

CCM int cyk_rules::translate_terminal_to_table_coord(int symbol)
{
    return (symbol & terminal_mask_negated) +
        preferences(thread_data->block_id, AT).get(prefs, preferences::max_number_of_non_terminal_symbols);
}

CCM int cyk_rules::translate_if_terminal(int symbol)
{
    return is_terminal(symbol) ? translate_terminal_to_table_coord(symbol) : symbol;
}
""",
                        dependencies=['cuda_helper', 'local_data', 'preferences', 'cuda_post_mortem'])