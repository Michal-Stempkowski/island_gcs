class SourceNode(object):
    def __init__(self, name, source, dependencies=None):
        self.name = name
        self.source = source
        self.dependencies = dependencies or []

    def link(self, files):
        return self.inner_link(files, set())

    def inner_link(self, files, dependency_set):
        result = ''
        for filename in self.dependencies:
            if filename not in dependency_set:
                dependency_set.add(filename)
                result += files[filename].inner_link(dependency_set)

        dependency_set.add(self.name)
        result += self.source

        return result

    def register(self, files):
        files[self.name] = self

    def unregister(self, files):
        del files[self.name]