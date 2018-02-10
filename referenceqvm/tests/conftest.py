import pytest

@pytest.fixture(scope="module")
def qvm():
    from referenceqvm.api import QVMConnection
    return QVMConnection(type_trans='wavefunction')

@pytest.fixture(scope="module")
def qvm_unitary():
    from referenceqvm.api import QVMConnection
    return QVMConnection(type_trans='unitary')
