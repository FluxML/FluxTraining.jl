_combinename(name, group::String) = _combinename((group, name))
_combinename(name, group::Tuple) = _combinename((group..., name))
_combinename(strings::Tuple) = join(strings, '/')