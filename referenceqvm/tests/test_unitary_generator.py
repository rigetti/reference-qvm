import pytest
import numpy as np
from referenceqvm.unitary_generator import lifted_two_gate, apply_two_qubit, \
                                           tensor_two_qubit_op, tensor_gates, \
                                           tensor_single_qubit_op
from referenceqvm.gates import gate_matrix
from pyquil.quil import Program
from pyquil.gates import H as Hgate
from pyquil.gates import RX as RXgate
from pyquil.gates import CNOT as CNOTgate


def test_lifted_swap():
    test_matrix = lifted_two_gate(0, gate_matrix["SWAP"], 2)
    result = gate_matrix['SWAP']
    assert np.isclose(test_matrix, result).all()

    test_matrix = lifted_two_gate(0, gate_matrix["SWAP"], 3)
    result = np.kron(np.eye(2**1), gate_matrix['SWAP'])
    assert np.isclose(test_matrix, result).all()

    test_matrix = lifted_two_gate(0, gate_matrix["SWAP"], 4)
    result = np.kron(np.eye(2**2), gate_matrix['SWAP'])
    assert np.isclose(test_matrix, result).all()

    test_matrix = lifted_two_gate(1, gate_matrix["SWAP"], 3)

    result = np.kron(gate_matrix['SWAP'], np.eye(2))
    assert np.isclose(test_matrix, result).all()

    with pytest.raises(ValueError):
        lifted_two_gate(2, gate_matrix['SWAP'], 3)
    with pytest.raises(ValueError):
        lifted_two_gate(3, gate_matrix['SWAP'], 3)
    with pytest.raises(ValueError):
        lifted_two_gate(-1, gate_matrix['SWAP'], 3)
    with pytest.raises(ValueError):
        lifted_two_gate(3, gate_matrix['SWAP'], 4)

    test_matrix = lifted_two_gate(1, gate_matrix['SWAP'], 4)
    result = np.kron(np.eye(2**1), np.kron(gate_matrix['SWAP'], np.eye(2**1)))
    assert np.isclose(test_matrix, result).all()

    test_matrix = lifted_two_gate(2, gate_matrix['SWAP'], 4)
    result = np.kron(np.eye(2**0), np.kron(gate_matrix['SWAP'], np.eye(2**2)))
    assert np.isclose(test_matrix, result).all()

    test_matrix = lifted_two_gate(8, gate_matrix['SWAP'], 10)
    result = np.kron(np.eye(2**0), np.kron(gate_matrix['SWAP'], np.eye(2**8)))
    assert np.isclose(test_matrix, result).all()


def test_two_qubit_gates():
    unitary_test = apply_two_qubit(gate_matrix['CNOT'], [1, 0], 2)
    unitary_true = tensor_two_qubit_op([1, 0], 2)
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['CNOT'], [0, 1], 2)
    unitary_true = tensor_two_qubit_op([0, 1], 2)
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['CNOT'], [2, 1], 3)
    unitary_true = tensor_two_qubit_op([2, 1], 3)
    assert np.isclose(unitary_test, unitary_true).all()

    with pytest.raises(ValueError):
        apply_two_qubit(gate_matrix['CNOT'], [2, 1], 2)

    unitary_test = apply_two_qubit(gate_matrix['UA'], [0, 1], 3)
    unitary_true = np.kron(np.eye(2), gate_matrix['UA'])
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [1, 0], 3)
    unitary_true = np.kron(np.eye(2), gate_matrix['UA'])
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [1, 2], 4)
    unitary_true = np.kron(np.eye(2), np.kron(gate_matrix['UA'], np.eye(2)))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [3, 2], 4)
    unitary_true = np.kron(gate_matrix['UA'], np.eye(4))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [2, 3], 4)
    unitary_true = np.kron(gate_matrix['UA'], np.eye(4))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [0, 3], 4)
    swap_01 = np.kron(np.eye(4), gate_matrix['SWAP'])
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12.dot(swap_01)
    V = np.kron(gate_matrix['UA'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [3, 0], 4)
    swap_01 = np.kron(np.eye(4), gate_matrix['SWAP'])
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12.dot(swap_01)
    V = np.kron(gate_matrix['UA'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [1, 3], 4)
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['UA'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['UA'], [3, 1], 4)
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['UA'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['CNOT'], [3, 1], 4)
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['CNOT'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.isclose(unitary_test, unitary_true).all()

    unitary_test = apply_two_qubit(gate_matrix['SWAP'], [3, 1], 4)
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['SWAP'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.isclose(unitary_test, unitary_true).all()


def test_tensor_single_qubit_op():
    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 0, 4)
    true_unitary = np.kron(np.eye(8), gate_matrix['H'])
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 1, 4)
    true_unitary = np.kron(np.eye(4), np.kron(gate_matrix['H'], np.eye(2)))
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 2, 4)
    true_unitary = np.kron(np.eye(2), np.kron(gate_matrix['H'], np.eye(4)))
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 3, 4)
    true_unitary = np.kron(gate_matrix['H'], np.eye(8))
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 0, 5)
    true_unitary = np.kron(np.eye(2**4), gate_matrix['H'])
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 1, 5)
    true_unitary = np.kron(np.eye(2**3), np.kron(gate_matrix['H'], np.eye(2)))
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 2, 5)
    true_unitary = np.kron(np.eye(2**2), np.kron(gate_matrix['H'], np.eye(2**2)))
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 3, 5)
    true_unitary = np.kron(np.eye(2**1), np.kron(gate_matrix['H'], np.eye(2**3)))
    assert np.isclose(test_unitary, true_unitary).all()

    test_unitary = tensor_single_qubit_op(gate_matrix['H'], 4, 5)
    true_unitary = np.kron(np.eye(2**0), np.kron(gate_matrix['H'], np.eye(2**4)))
    assert np.isclose(test_unitary, true_unitary).all()


def test_tensor_gates_single_qubit():
    prog = Program().inst([Hgate(0)])
    test_unitary = tensor_gates(prog.actions[0][1], 1)
    true_unitary = gate_matrix['H']
    assert np.isclose(test_unitary, true_unitary).all()

    prog = Program().inst([Hgate(0)])
    test_unitary = tensor_gates(prog.actions[0][1], 5)
    true_unitary = np.kron(np.eye(2**4), gate_matrix['H'])
    assert np.isclose(test_unitary, true_unitary).all()

    prog = Program().inst([RXgate(0.2)(3)])
    test_unitary = tensor_gates(prog.actions[0][1], 5)
    true_unitary = np.kron(np.eye(2**1), np.kron(gate_matrix['RX'](0.2),  np.eye(2**3)))
    assert np.isclose(test_unitary, true_unitary).all()

    prog = Program().inst([RXgate(0.5)(4)])
    test_unitary = tensor_gates(prog.actions[0][1], 5)
    true_unitary = np.kron(np.eye(2**0), np.kron(gate_matrix['RX'](0.5),  np.eye(2**4)))
    assert np.isclose(test_unitary, true_unitary).all()


def test_tensor_gates_two_qubit():
    prog = Program().inst([CNOTgate(0, 1)])
    test_unitary = tensor_gates(prog.actions[0][1], 4)
    true_unitary = tensor_two_qubit_op([0, 1], 4)
    assert np.isclose(test_unitary, true_unitary).all()

    prog = Program().inst([CNOTgate(1, 0)])
    test_unitary = tensor_gates(prog.actions[0][1], 4)
    true_unitary = tensor_two_qubit_op([1, 0], 4)
    assert np.isclose(test_unitary, true_unitary).all()

    prog = Program().inst([CNOTgate(1, 3)])
    test_unitary = tensor_gates(prog.actions[0][1], 4)
    true_unitary = tensor_two_qubit_op([1, 3], 4)
    assert np.isclose(test_unitary, true_unitary).all()


if __name__ == "__main__":
    test_lifted_swap()
    test_two_qubit_gates()
    test_tensor_single_qubit_op()
    test_tensor_gates_single_qubit()
    test_tensor_gates_two_qubit()
