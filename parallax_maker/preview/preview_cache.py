# (c) 2024 Niels Provos

_preview_cache = {}

def get_preview_cache(key):
    tuple_key = key.to_tuple() if hasattr(key, "to_tuple") else key
    return _preview_cache.get(tuple_key)

def set_preview_cache(key, result):
    tuple_key = key.to_tuple() if hasattr(key, "to_tuple") else key
    _preview_cache[tuple_key] = result

def clear_preview_cache():
    _preview_cache.clear()
