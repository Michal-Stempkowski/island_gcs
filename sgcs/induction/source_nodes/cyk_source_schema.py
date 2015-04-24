import sgcs.induction.source_nodes.kernel as kernel


class CykSourceSchema(object):
    def __init__(self):
        self.files = dict()
        self.requires_update = True

    def generate_schema(self):
        _ = self.kernel

        _ = self.kernel.link(self.files, kernel.tag())
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
