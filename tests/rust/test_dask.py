"""Test for rust functionality when using dask in rust."""

import tiatoolbox.rust.dask as rdask


def test_add() -> None:
    """Test case for add."""
    assert rdask.add(5, 4) == 9
