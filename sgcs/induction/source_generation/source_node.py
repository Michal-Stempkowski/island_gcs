from datetime import datetime
import re


class SourceNode(object):
    absolute_identifier_tag = '__sn_absolute_identifier_tag__'

    def __init__(self, name, source, dependencies=None, internal_dependencies=None):
        self.name = name
        self.source = source
        self.dependencies = dependencies or []

        internal_dependencies = internal_dependencies or dict()
        self.token_resolver = TokenResolver(internal_dependencies)

    def link(self, files, absolute_identifier, additional_data=None):
        return self.inner_link(files, set(), absolute_identifier, additional_data if additional_data else dict())

    def inner_link(self, files, dependency_set, absolute_identifier, additional_data):
        result = ''
        for filename in self.dependencies:
            if filename not in dependency_set:
                dependency_set.add(filename)
                result += files[filename].inner_link(files, dependency_set, filename, additional_data)

        dependency_set.add(absolute_identifier)
        result += self.token_resolver.resolve_tokens(self, absolute_identifier, files, dependency_set, additional_data)

        return result

    def register(self, files, absolute_identifier):
        files[absolute_identifier] = self

    @staticmethod
    def unregister(files, absolute_identifier):
        del files[absolute_identifier]


class TokenResolver(object):
    absolute_identifier_tag = '__sn_absolute_identifier_tag__'
    timestamp_string_tag = '__sn_timestamp_tag__'
    identifier_param_regex = r'{name}\s*?\(\s*?(?P<{name}>[a-zA-Z_]\w*?)?\s*?\)'
    verbose_context_param_regex = r'{name}\s*?\((?P<{name}>.*?)\)'

    @staticmethod
    def param(param_type, param_name):
        return r'\s*? {type} \s*?'.format(type=param_type).format(name=param_name)

    def __init__(self, internal_dependencies):
        self.internal_dependencies = internal_dependencies
        self.code_generation_regex = re.compile(r'__sg_.*?__', re.DOTALL)
        self.generation_rules = [
            (re.compile(
                r'''

                    (?P<whole_pattern>__sg_repeat\(
                        {vals}

                        (
                            ,{begin}
                            (
                                , {separator}

                                (
                                    ,{end}
                                )?
                            )?
                        )?
                    \)__)

                '''
                .format(
                    vals=self.param(self.identifier_param_regex, 'vals'),
                    begin=self.param(self.verbose_context_param_regex, 'begin'),
                    separator=self.param(self.verbose_context_param_regex, 'separator'),
                    end=self.param(self.verbose_context_param_regex, 'end'))
                , re.DOTALL | re.VERBOSE),
                lambda source, matches, additional_data: #source)
                source.replace(matches['whole_pattern'],
                               (matches['begin'] or '') +
                               (matches['separator'] or '').join(additional_data[matches['vals']]) +
                               (matches['end'] or '')))
        ]

    @staticmethod
    def private_header(node_name, tag):
        return '__{0}_{1}__'.format(node_name, tag)

    def resolve_tokens(self, node, absolute_identifier, files, dependency_set, additional_data):
        source = node.source.replace(self.absolute_identifier_tag, absolute_identifier)
        source = source.replace(self.timestamp_string_tag, datetime.now().ctime())

        for token, source_node in self.internal_dependencies.items():
            source = source.replace(token, source_node.inner_link(
                files, dependency_set, self.private_header(node.name, token), additional_data))

        macros = self.code_generation_regex.findall(source)
        # print(macros)
        for macro in macros:
            for matcher, resolver in self.generation_rules:
                matches = [match.groupdict() for match in matcher.finditer(macro)]
                if matches:
                    source = resolver(source, matches[0], additional_data)
                    # print(matches)

        return source