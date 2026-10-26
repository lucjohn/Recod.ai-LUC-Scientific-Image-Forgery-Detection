import pytest


@pytest.fixture(scope="session")
def index():
    """The full data index; skips the test when the competition data isn't on this machine."""
    from forgery.data import build_index, data_root

    root = data_root()
    if not root.exists():
        pytest.skip(f"data not found at {root}")
    return build_index(root)
