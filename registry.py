import collections

_REGISTRY = collections.defaultdict(dict)


def build_register_decorator(_type):
    type_registry = _REGISTRY[_type]

    def register_type(key):
        def inner(fn):
            type_registry[key] = fn
            return fn

        return inner

    return register_type


def build_make_function(_type):
    """
    creates a function that building objects of a specific "type"
    """
    type_registry = _REGISTRY[_type]

    def make_type(definition, **kwargs):
        assert isinstance(definition, dict)
        definition = dict(definition)  # make a copy
        key = definition.pop("key")
        return type_registry[key](**definition, **kwargs)

    return make_type


def build_lookup(_type):
    """
    creates a function that returns the registered functions of a specific "type"
    """
    type_registry = _REGISTRY[_type]

    def lookup(key):
        return type_registry[key]

    return lookup
