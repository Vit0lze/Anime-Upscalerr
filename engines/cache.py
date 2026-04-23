from collections import OrderedDict
import threading

class EngineCache:
    def __init__(self, max_entries=6):
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._max = max_entries

    def get_or_create(self, key, factory):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if len(self._cache) >= self._max:
                _, old = self._cache.popitem(last=False)
                if hasattr(old, 'release'):
                    try: old.release()
                    except Exception: pass
            engine = factory()
            self._cache[key] = engine
            return engine

    def clear(self):
        with self._lock:
            for eng in self._cache.values():
                if hasattr(eng, 'release'):
                    try: eng.release()
                    except Exception: pass
            self._cache.clear()

_engine_cache = EngineCache(max_entries=6)
