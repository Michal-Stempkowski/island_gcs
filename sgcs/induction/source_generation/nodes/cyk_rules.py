from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cyk_rules'

cyk_rules = SourceNode('default_cyk_rules', """
////CPP
class cyk_rules
{
public:
    CCM cyk_rules(int* rules_by_right_, local_data *thread_data_);
    CCM int get_rule_by_right_sight(int left_symbol, int right_symbol) const;

private:
    int* rules_by_right;
    local_data* thread_data;
};

CCM cyk_rules::cyk_rules(int* rules_by_right_, local_data *thread_data_) :
    rules_by_right(rules_by_right_),
    thread_data(thread_data_)
{

}

CCM int cyk_rules::get_rule_by_right_sight(int left_symbol, int right_symbol) const
{
    return table_get(rules_by_right, generate_absolute_index(
        thread_data->block_id, thread_data->number_of_blocks));
}
""",
                        dependencies=['cuda_helper', 'local_data'])