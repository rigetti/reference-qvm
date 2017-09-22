from pyquil.quil import Program
from pyquil.gates import *
import numpy as np
import pytest
from referenceqvm.gates import gate_matrix


def test_random_gates(qvm_unitary):
    p = Program().inst([H(0), H(1), H(0)])
    test_unitary = qvm_unitary.unitary(p)
    actual_unitary = np.kron(gate_matrix['H'], np.eye(2 ** 1))
    assert np.allclose(test_unitary, actual_unitary)

    p = Program().inst([H(0), X(1), Y(2), Z(3)])
    test_unitary = qvm_unitary.unitary(p)
    actual_unitary = np.kron(gate_matrix['Z'],
                             np.kron(gate_matrix['Y'],
                                     np.kron(gate_matrix['X'],
                                             gate_matrix['H'])))
    assert np.allclose(test_unitary, actual_unitary)

    p = Program().inst([X(2), CNOT(2, 1), CNOT(1, 0)])
    test_unitary = qvm_unitary.unitary(p)
    # gates are multiplied in 'backwards' order
    actual_unitary = np.kron(np.eye(2 ** 1), gate_matrix['CNOT']).dot(
                     np.kron(gate_matrix['CNOT'], np.eye(2 ** 1))).dot(
                     np.kron(gate_matrix['X'], np.eye(2 ** 2)))
    assert np.allclose(test_unitary, actual_unitary)


def test_identity(qvm_unitary):
    p = Program()
    test_unitary = qvm_unitary.unitary(p)
    assert np.allclose(test_unitary, np.eye(2 ** 0))


def test_qaoa_unitary(qvm_unitary):
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    wf_true = np.reshape(np.array(wf_true), (4, 1))
    prog = Program()
    prog.inst([RY(np.pi/2)(0), RX(np.pi)(0),
               RY(np.pi/2)(1), RX(np.pi)(1),
               CNOT(0, 1), RX(-np.pi/2)(1), RY(4.71572463191)(1),
               RX(np.pi/2)(1), CNOT(0, 1),
               RX(-2*2.74973750579)(0), RX(-2*2.74973750579)(1)])

    test_unitary = qvm_unitary.unitary(prog)
    wf_test = np.zeros((4, 1))
    wf_test[0, 0] = 1.0
    wf_test = test_unitary.dot(wf_test)
    assert np.allclose(wf_test, wf_true)


def test_unitary_errors(qvm_unitary):
    # do we properly throw errors when non-gates are thrown into a unitary?

    # try measuring
    prog = Program()
    prog.inst([H(0), H(1)])
    prog.measure(0, [0])
    with pytest.raises(TypeError):
        qvm_unitary.unitary(prog)

    # try an undefined DefGate
    prog = Program()
    prog.defgate("hello", np.array([[0, 1], [1, 0]]))
    prog.inst(("hello2", 0))
    with pytest.raises(TypeError):
        qvm_unitary.unitary(prog)
