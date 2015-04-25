from sgcs.induction.source_generation.source_node import SourceNode


def tag():
    return 'preferences'


preferences = SourceNode('preferences_access_class', """
////CPP
#if !defined(PREFERENCES_H)
#define PREFERENCES_H

class preferences
{
public:
    enum option : int
    {
        sentence_length = 0,
        max_alphabet_size,
        max_symbols_in_cell,
        number_of_blocks,
        number_of_threads,
        enum_size               // DO NOT use it as enum, it represents size of this enum
    };

    static const int invalid_value = -1;

    CCM preferences(const int block_id, const char* source_code_localization_);

    CCM int get(int *preferences, option opt) const;

    CCM ~preferences();

private:
    CCM int get_index(int field_id) const;

    const int block_id;
    const char* source_code_localization;
};

CCM preferences::preferences(const int block_id_, const char* source_code_localization_) :
source_code_localization(source_code_localization_), block_id(block_id_)
{

}

CCM int preferences::get(int *preferences, option opt) const
{
    if (opt < 0 || opt >= enum_size)
    {
        return invalid_value;
    }

    return preferences[get_index(opt)];
}

CCM preferences::~preferences()
{
}

CCM int preferences::get_index(int field_id) const
{
    return block_id * enum_size + field_id;
}

#endif""",
                         dependencies=['cuda_helper'])
