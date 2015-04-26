from datetime import datetime
import re
import collections


class SourceNode(object):
    absolute_identifier_tag = '__sn_absolute_identifier_tag__'

    def __init__(self, name, source, dependencies=None, internal_dependencies=None):
        self.name = name
        self.source = source
        self.dependencies = dependencies or []

        internal_dependencies = internal_dependencies or dict()
        self.token_resolver = TokenResolver(self, internal_dependencies)

    def link(self, files, absolute_identifier, additional_data=None):
        return self.inner_link(files, set(), absolute_identifier, additional_data if additional_data else dict())

    def _link_dependency(self, filename, dependency_set, files, additional_data):
        if filename not in dependency_set:
            dependency_set.add(filename)
            return files[filename].inner_link(files, dependency_set, filename, additional_data)

        return ''

    def inner_link(self, files, dependency_set, absolute_identifier, additional_data):
        result = ''
        for filename in self.dependencies:
            result += self._link_dependency(filename, dependency_set, files, additional_data)

        dependency_set.add(absolute_identifier)
        result += self.token_resolver.resolve_tokens(self, absolute_identifier, files, dependency_set, additional_data)

        return result

    def register(self, files, absolute_identifier):
        files[absolute_identifier] = self

    @staticmethod
    def unregister(files, absolute_identifier):
        del files[absolute_identifier]


class Param(object):
    def __init__(self, type_regex, name, is_optional=False):
        self.type = type_regex
        self.name = name
        self.is_optional = is_optional

    @staticmethod
    def _on(condition, append):
        return append if condition else ''

    def expand(self, is_first=False):
        return r'{optional_start}{comma}\s*? {type} \s*?{optional_end}'\
            .format(
                type=self.type,
                comma=self._on(not is_first, r','),
                optional_start=self._on(self.is_optional, r'('),
                optional_end=self._on(self.is_optional, r')?'))\
            .format(name=self.name)


class Token(object):
    whole_pattern_tag = 'whole_pattern'

    def __init__(self, name, logic, *args):
        self.name = name
        self.logic = logic
        self.parameters = args

    def expand(self):
        return r'''(?P<{whole_pattern_tag}>__sg_{token_name}\({parameters}\)__)'''\
            .format(
                whole_pattern_tag=self.whole_pattern_tag,
                token_name=self.name,
                parameters=' '.join([param.expand(param is self.parameters[0]) for param in self.parameters]))

    def get_regex(self):
        return re.compile(self.expand(), re.DOTALL | re.VERBOSE)

    def resolve(self, source, matches, additional_data):
        parameters_names = [param.name for param in self.parameters]
        parameters_names.insert(0, self.whole_pattern_tag)
        params = collections.OrderedDict()
        for key in parameters_names:
            params[key] = matches[key] if key in matches else None
        # params = {key: (matches[key] if key in matches else None) for key in parameters_names}
        return self.logic(source, additional_data, **params)


class TokenResolver(object):
    absolute_identifier_tag = '__sn_absolute_identifier_tag__'
    timestamp_string_tag = '__sn_timestamp_tag__'
    identifier_param_regex = r'{name}\s*?\(\s*?(?P<{name}>[a-zA-Z_]\w*?)?\s*?\)'
    verbose_context_param_regex = r'{name}\s*?\((?P<{name}>.*?)\)'

    @staticmethod
    def param(param_type, param_name, first=False, optional=False):
        return Param(param_type, param_name, optional).expand(first)

    def __init__(self, source_node, internal_dependencies):
        self.source_node = source_node
        self.internal_dependencies = internal_dependencies
        self.code_generation_regex = re.compile(r'__sg_.*?__', re.DOTALL)

        self.generation_rules = [
            Token(
                'repeat',

                lambda source, additional_data, whole_pattern, vals, begin, separator, end:\
                source.replace(whole_pattern,
                               (begin or '') + (separator or '').join(additional_data[vals]) + (end or '')),

                Param(self.identifier_param_regex, 'vals'),
                Param(self.verbose_context_param_regex, 'begin', is_optional=True),
                Param(self.verbose_context_param_regex, 'separator', is_optional=True),
                Param(self.verbose_context_param_regex, 'end', is_optional=True))
        ]

    def private_header(self, tag):
        return '__{0}_{1}__'.format(self.source_node.name, tag)

    def _apply_absolute_identifier(self, source, absolute_identifier):
        return source.replace(self.absolute_identifier_tag, absolute_identifier)

    def _generate_timestamps(self, source):
        return source.replace(self.timestamp_string_tag, datetime.now().ctime())

    def _perform_macro_linkage(self, source, files, dependency_set, additional_data):
        for token, source_node in self.internal_dependencies.items():
                source = source.replace(token, source_node.inner_link(
                    files, dependency_set, self.private_header(token), additional_data))

        return source

    def _resolve_macros(self, source, additional_data):
        macros = self.code_generation_regex.findall(source)
        for macro in macros:
            for rule in self.generation_rules:
                matches = [match.groupdict() for match in rule.get_regex().finditer(macro)]
                if matches:
                    source = rule.resolve(source, matches[0], additional_data)

        return source

    def _try_resolving_macros(self, source, files, dependency_set, node, additional_data):
        source = self._generate_timestamps(source)

        source = self._perform_macro_linkage(source, files, dependency_set, additional_data)

        return self._resolve_macros(source, additional_data)

    def resolve_tokens(self, node, absolute_identifier, files, dependency_set, additional_data):
        old_source = ''
        source = self._apply_absolute_identifier(node.source[:], absolute_identifier)

        while old_source != source:
            old_source = source[:]

            source = self._try_resolving_macros(source, files, dependency_set, node, additional_data)

        return source