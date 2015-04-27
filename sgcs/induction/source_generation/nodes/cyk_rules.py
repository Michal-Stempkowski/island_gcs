from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cyk_rules'

cyk_rules = SourceNode('default_cyk_rules', """
////CPP
class cyk_rules
{
public:
    CCM cyk_rules(int* rules_by_right_);

private:
    int* rules_by_right;
};

CCM cyk_rules::cyk_rules(int* rules_by_right_) :
    rules_by_right(rules_by_right_)
{

}
""",
                        dependencies=['cuda_helper'])