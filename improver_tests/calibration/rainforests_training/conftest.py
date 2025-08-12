import sys

import pytest


try:
    import tl2cgen  # noqa: F401
except ModuleNotFoundError:
    TREELITE_ENABLED = False
else:
    TREELITE_ENABLED = True


@pytest.fixture
def treelite_available(available, monkeypatch):
    available = available and TREELITE_ENABLED
    print(f'setting treelite_available = {available}')
    if not available:
        monkeypatch.setitem(sys.modules, "tl2cgen", None)
    return available
