import sgcs.induction.source_nodes.kernel as kernel


class CykSourceSchema(object):
    def __init__(self):
        self.files = dict()

    @property
    def kernel(self):
        if kernel.tag() not in self.files:
            self.files[kernel.tag()] = kernel.cyk_kernel
            kernel.cyk_kernel.register(self.files)

        return self.files[kernel.tag()]

    @kernel.setter
    def kernel(self, value):
        if kernel.tag() in self.files:
            self.files[kernel.tag()].unregister(self.files)

        self.files[kernel.tag()] = value
        value.register(self.files)

    @kernel.deleter
    def kernel(self):
        self.files[kernel.tag()].unregister(self.files)