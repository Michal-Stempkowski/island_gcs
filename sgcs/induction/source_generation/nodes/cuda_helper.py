from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cuda_helper'

cuda_debug = SourceNode('cuda_debug', """
#if !defined CUDA_DEBUG
#define CUDA_DEBUG 1
#endif

#if CUDA_DEBUG
#define log_debug(...) printf(__VA_ARGS__)
#else
#define log_debug(...)
#endif
""")

cuda_helper_common = SourceNode('cuda_helper_common', """
#if !defined CUDA_HELPER_COMMON_H
#define CUDA_HELPER_COMMON_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define CCM __host__ __device__

#endif
""")

cuda_helper = SourceNode(
    'default_cyk_helper',
    """
__sn_cuda_debug__
__sn_cuda_helper_common__
""",
    internal_dependencies={
        '__sn_cuda_debug__': cuda_debug,
        '__sn_cuda_helper_common__': cuda_helper_common
    })

