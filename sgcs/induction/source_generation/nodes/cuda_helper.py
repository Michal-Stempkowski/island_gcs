from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'cuda_helper'

cuda_debug = SourceNode('cuda_debug', """
////CPP
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
////CPP
#if !defined CUDA_HELPER_COMMON_H
#define CUDA_HELPER_COMMON_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define CCM __device__

#endif
""")

cuda_error = SourceNode('cuda_error', """
////CPP
#if !defined CUDA_ERROR_H
#define CUDA_ERROR_H

enum error : int
{
    no_errors_occured = 0,
    index_out_of_bounds = -1
};

#endif
""")

cuda_table_helper = SourceNode('cuda_table_helper', """
////CPP

#if !defined CUDA_TABLE_HELPER_H
#define CUDA_TABLE_HELPER_H

CCM int generate_absolute_index(int x, int x_max);
CCM int generate_absolute_index(int x, int x_max, int y, int y_max);
CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max);
CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max, int i, int i_max);

CCM int table_get(int* table, int absolute_index);
CCM int table_set(int* table, int absolute_index, int value);

static CCM int apply_param(int a, int a_max, int index)
{
    if (a < 0 || a >= a_max)
    {
        return error::index_out_of_bounds;
    }

    if (index != error::index_out_of_bounds)
    {
        index = index * a_max + a;
    }

    return index;
}

CCM int generate_absolute_index(int x, int x_max)
{
    return apply_param(x, x_max, 0);
}

CCM int generate_absolute_index(int x, int x_max, int y, int y_max)
{
    return apply_param(y, y_max, generate_absolute_index(x, x_max));
}

CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max)
{
    return apply_param(z, z_max, generate_absolute_index(x, x_max, y, y_max));
}

CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max, int i, int i_max)
{
    return apply_param(i, i_max, generate_absolute_index(x, x_max, y, y_max, z, z_max));
}

CCM int table_get(int* table, int absolute_index)
{
    return
        absolute_index >= error::no_errors_occured ?
        table[absolute_index] :
        absolute_index;
}

CCM int table_set(int* table, int absolute_index, int value)
{
    return absolute_index >= error::no_errors_occured ?
        table[absolute_index] = value, error::no_errors_occured :
        absolute_index;
}

#endif
""")

cuda_helper = SourceNode(
    'default_cyk_helper',
    """
    ////CPP
#include <iostream>
#include <sstream>

@@sn_cuda_debug@@
@@sn_cuda_helper_common@@
@@sn_cuda_error@@
@@sn_cuda_table_helper@@
""",
    internal_dependencies={
        '@@sn_cuda_debug@@': cuda_debug,
        '@@sn_cuda_helper_common@@': cuda_helper_common,
        '@@sn_cuda_error@@': cuda_error,
        '@@sn_cuda_table_helper@@': cuda_table_helper
    })

