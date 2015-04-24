from sgcs.induction.source_generation.nodes import cuda_helper
import sgcs.induction.source_generation.nodes.kernel as kernel


class CykSourceSchema(object):
    def __init__(self):
        self.files = dict()
        self.requires_update = True

    def generate_schema(self):
        _ = self.kernel
        _ = self.cuda_helper

        _ = self.kernel.link(self.files, kernel.tag())
        # print(_.split('\n')[-2])
        print(_)
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
