import sys

import pytest


try:
    import treelite_runtime  # noqa: F401
except ModuleNotFoundError:
    TREELITE_ENABLED = False
else:
    TREELITE_ENABLED = True


@pytest.fixture
def treelite_available(available, monkeypatch):
    available = available and TREELITE_ENABLED
    print(f'setting treelite_available = {available}')
    if not available:
        monkeypatch.setitem(sys.modules, "treelite_runtime", None)
    return available
