import pytest
import numpy as np
from pyquil.quil import Program
from pyquil.gates import *


def test_identify_bits(qvm):
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


# def test_qubit_limit(qvm):
#     p = Program()
#     for i in range(25):
#         p.inst(X(i))

#     with pytest.raises(RuntimeError):
#         qvm.wavefunction(p)
#     with pytest.raises(RuntimeError):
#         qvm.run(p)
#     with pytest.raises(RuntimeError):
#         qvm.run_and_measure(p)
