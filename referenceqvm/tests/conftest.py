import pytest

@pytest.fixture(scope="module")
def qvm():
    from referenceqvm.api import SyncConnection
    return SyncConnection(type_trans='wavefunction')

@pytest.fixture(scope="module")
def qvm_unitary():
    from referenceqvm.api import SyncConnection
    return SyncConnection(type_trans='unitary')
