import pytest
import numpy as np
from referenceqvm.api import SyncConnection
from pyquil.quil import Program
from pyquil.gates import *


def test_identify_bits():
    qvm = SyncConnection()

    p = Program()
    for i in range(5):
        p.inst(X(i))
    for i in range(670):
        p.inst(TRUE(i))
    qvm.wavefunction(p)
    qvm.run(p)
    qvm.run_and_measure(p)
    assert qvm.num_qubits == 5
    assert len(qvm.classical_memory) == 670

    p = Program()
    for i in range(670):
        p.inst(TRUE(i))
    qvm.wavefunction(p)
    qvm.run(p)
    qvm.run_and_measure(p)
    assert qvm.num_qubits == 0
    assert len(qvm.classical_memory) == 670

    p = Program(X(0))
    qvm.wavefunction(p)
    qvm.run(p)
    qvm.run_and_measure(p)
    assert qvm.num_qubits == 1
    assert len(qvm.classical_memory) == 512


def test_empty_program():
    qvm = SyncConnection()

    p = Program()

    with pytest.raises(TypeError):
        qvm.wavefunction(p)
    with pytest.raises(TypeError):
        qvm.run(p)
    with pytest.raises(TypeError):
        qvm.run_and_measure(p)


def test_qubit_limit():
    qvm = SyncConnection()

    p = Program()
    for i in range(24):
        p.inst(X(i))

    with pytest.raises(RuntimeError):
        qvm.wavefunction(p)
    with pytest.raises(RuntimeError):
        qvm.run(p)
    with pytest.raises(RuntimeError):
        qvm.run_and_measure(p)

if __name__ == '__main__':
    test_identify_bits()
    test_empty_program()
    test_qubit_limit()
