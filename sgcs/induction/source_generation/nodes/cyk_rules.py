from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cyk_rules'

cyk_rules = SourceNode('default_cyk_rules', """
////CPP
class cyk_rules
{
public:
    CCM cyk_rules(int* terminal_rules_);

private:
    int* terminal_rules;
};

CCM cyk_rules::cyk_rules(int* terminal_rules_) :
    terminal_rules(terminal_rules_)
{

}
""",
                        dependencies=['cuda_helper'])