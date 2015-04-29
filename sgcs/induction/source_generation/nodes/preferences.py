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
        @@sg_repeat(vals(preferences_headers), begin(), separator(,\n        ))@@,
        enum_size,                          // DO NOT use it as enum, it represents size of this enum
        @@sg_repeat(
            vals(additional_preferences_headers),
            begin(),
            separator(,\n        ),
            end(,),
            optional_generation(////))@@
        enum_size_with_additionals          // DO NOT use it as enum, it represents size of this enum
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
    if (opt < 0 || opt == enum_size || opt >= enum_size_with_additionals)
    {
        return invalid_value;
    }
    else if (opt > enum_size)
    {
        return @@sg_switch(table(additional_preferences), var(opt))@@;
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
