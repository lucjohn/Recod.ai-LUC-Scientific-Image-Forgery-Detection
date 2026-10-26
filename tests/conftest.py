import pytest


@pytest.fixture(scope="session")
def index():
    """The full data index; skips the test when the competition data isn't on this machine."""
    from forgery.data import build_index, data_root

    try:
        return build_index(data_root())
    except FileNotFoundError as error:
        pytest.skip(f"competition data unavailable: {error}")
