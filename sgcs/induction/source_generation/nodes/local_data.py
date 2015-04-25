from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'local_data'

cyk_kernel = SourceNode('default_local_data', """
////CPP
class
""",
                        dependencies=['cuda_helper'])
