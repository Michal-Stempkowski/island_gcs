class SourceNode(object):
    absolute_identifier_tag = '__sn_absolute_identifier_tag__'

    def __init__(self, name, source, dependencies=None):
        self.name = name
        self.source = source
        self.dependencies = dependencies or []

    def link(self, files, absolute_identifier):
        return self.inner_link(files, set(), absolute_identifier)

    def inner_link(self, files, dependency_set, absolute_identifier):
        result = ''
        for filename in self.dependencies:
            if filename not in dependency_set:
                dependency_set.add(filename)
                result += files[filename].inner_link(dependency_set, filename)

        dependency_set.add(absolute_identifier)
        result += self.source.replace(self.absolute_identifier_tag, absolute_identifier)

        return result

    def register(self, files, absolute_identifier):
        files[absolute_identifier] = self

    @staticmethod
    def unregister(files, absolute_identifier):
        del files[absolute_identifier]