from sgcs.induction.source_generation.nodes import cuda_helper
import sgcs.induction.source_generation.nodes.kernel as kernel
import sgcs.induction.source_generation.nodes.kernel_main as kernel_main
import sgcs.induction.source_generation.nodes.preferences as preferences
import sgcs.induction.source_generation.nodes.cyk_table as cyk_table
import sgcs.induction.source_generation.nodes.local_data as local_data
import sgcs.induction.source_generation.nodes.cuda_post_mortem as cuda_post_mortem


class CykSourceSchema(object):
    def __init__(self):
        self.files = dict()
        self.requires_update = True

    def generate_schema(self):
        _ = self.kernel
        _ = self.cuda_helper
        _ = self.kernel_main
        _ = self.preferences
        _ = self.cyk_table
        _ = self.local_data
        _ = self.cuda_post_mortem

        _ = self.kernel.link(self.files, kernel.tag())
        # print(_.split('\n')[-2])
        # print(_)
        return _

    @property
    def kernel(self):
        return self._source_getter(kernel.tag(), kernel.cyk_kernel)

    @kernel.setter
    def kernel(self, value):
        self._source_setter(kernel.tag(), value)

    @kernel.deleter
    def kernel(self):
        self._source_deleter(kernel.tag())

    @property
    def cuda_helper(self):
        return self._source_getter(cuda_helper.tag(), cuda_helper.cuda_helper)

    @cuda_helper.setter
    def cuda_helper(self, value):
        self._source_setter(cuda_helper.tag(), value)

    @cuda_helper.deleter
    def cuda_helper(self):
        self._source_deleter(cuda_helper.tag())

    @property
    def kernel_main(self):
        return self._source_getter(kernel_main.tag(), kernel_main.kernel_main)

    @kernel_main.setter
    def kernel_main(self, value):
        self._source_setter(kernel_main.tag(), value)

    @kernel_main.deleter
    def kernel_main(self):
        self._source_deleter(kernel_main.tag())

    @property
    def preferences(self):
        return self._source_getter(preferences.tag(), preferences.preferences)

    @preferences.setter
    def preferences(self, value):
        self._source_setter(preferences.tag(), value)

    @preferences.deleter
    def preferences(self):
        self._source_deleter(preferences.tag())

    @property
    def cyk_table(self):
        return self._source_getter(cyk_table.tag(), cyk_table.cyk_table)

    @cyk_table.setter
    def cyk_table(self, value):
        self._source_setter(cyk_table.tag(), value)

    @cyk_table.deleter
    def cyk_table(self):
        self._source_deleter(cyk_table.tag())

    @property
    def local_data(self):
        return self._source_getter(local_data.tag(), local_data.local_data)

    @local_data.setter
    def local_data(self, value):
        self._source_setter(local_data.tag(), value)

    @local_data.deleter
    def local_data(self):
        self._source_deleter(local_data.tag())

    @property
    def cuda_post_mortem(self):
        return self._source_getter(cuda_post_mortem.tag(), cuda_post_mortem.cuda_post_mortem)

    @cuda_post_mortem.setter
    def cuda_post_mortem(self, value):
        self._source_setter(cuda_post_mortem.tag(), value)

    @cuda_post_mortem.deleter
    def cuda_post_mortem(self):
        self._source_deleter(cuda_post_mortem.tag())

    def _source_getter(self, tag, default_source):
        if tag not in self.files:
            self.files[tag] = default_source
            default_source.register(self.files, tag)
            self.requires_update = True

        return self.files[tag]

    def _source_setter(self, tag, value):
        if tag in self.files:
            self.files[tag].unregister(self.files, tag)

        self.files[tag] = value
        value.register(self.files, tag)
        self.requires_update = True

    def _source_deleter(self, tag):
        self.files[tag].unregister(self.files, tag)
        self.requires_update = True
