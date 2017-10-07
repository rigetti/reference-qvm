import pytest
import warnings
import numpy as np
from referenceqvm.unitary_generator import (lifted_gate, apply_gate,
                                            tensor_gates, tensor_up)
from referenceqvm.gates import gate_matrix, utility_gates
from pyquil.quil import Program
from pyquil.gates import H as Hgate
from pyquil.gates import RX as RXgate
from pyquil.gates import CNOT as CNOTgate
from pyquil.paulis import PauliTerm, PauliSum


def test_lifted_swap():
    # SWAP indexed at 0
    test_matrix = lifted_gate(0, gate_matrix['SWAP'], 2).toarray()
    result = gate_matrix['SWAP']
    assert np.allclose(test_matrix, result)

    test_matrix = lifted_gate(0, gate_matrix["SWAP"], 3).toarray()
    result = np.kron(np.eye(2**1), gate_matrix['SWAP'])
    assert np.allclose(test_matrix, result)

    test_matrix = lifted_gate(0, gate_matrix["SWAP"], 4).toarray()
    result = np.kron(np.eye(2**2), gate_matrix['SWAP'])
    assert np.allclose(test_matrix, result)

    # SWAP indexed at max num_qubits
    test_matrix = lifted_gate(1, gate_matrix["SWAP"], 3).toarray()
    result = np.kron(gate_matrix['SWAP'], np.eye(2))
    assert np.allclose(test_matrix, result)

    # SWAP indexed outside of the range throws error
    with pytest.raises(ValueError):
        lifted_gate(2, gate_matrix['SWAP'], 3)
    with pytest.raises(ValueError):
        lifted_gate(3, gate_matrix['SWAP'], 3)
    with pytest.raises(ValueError):
        lifted_gate(-1, gate_matrix['SWAP'], 3)
    with pytest.raises(ValueError):
        lifted_gate(3, gate_matrix['SWAP'], 4)

    test_matrix = lifted_gate(1, gate_matrix['SWAP'], 4).toarray()
    result = np.kron(np.eye(2**1), np.kron(gate_matrix['SWAP'], np.eye(2**1)))
    assert np.allclose(test_matrix, result)

    test_matrix = lifted_gate(2, gate_matrix['SWAP'], 4).toarray()
    result = np.kron(np.eye(2**0), np.kron(gate_matrix['SWAP'], np.eye(2**2)))
    assert np.allclose(test_matrix, result)

    test_matrix = lifted_gate(8, gate_matrix['SWAP'], 10).toarray()
    result = np.kron(np.eye(2**0), np.kron(gate_matrix['SWAP'], np.eye(2**8)))
    assert np.allclose(test_matrix, result)


def test_two_qubit_gates():
    unitary_test = apply_gate(gate_matrix['CNOT'], [1, 0], 2).toarray()
    unitary_true = np.kron(utility_gates['P0'], np.eye(2)) + \
                   np.kron(utility_gates['P1'], gate_matrix['X'])
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['CNOT'], [0, 1], 2).toarray()
    unitary_true = np.kron(np.eye(2), utility_gates['P0']) + \
                   np.kron(gate_matrix['X'], utility_gates['P1'])
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['CNOT'], [2, 1], 3).toarray()
    unitary_true = np.kron(gate_matrix['CNOT'], np.eye(2 ** 1))
    assert np.allclose(unitary_test, unitary_true)

    with pytest.raises(ValueError):
        apply_gate(gate_matrix['CNOT'], [2, 1], 2)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [0, 1], 3).toarray()
    unitary_true = np.kron(np.eye(2), gate_matrix['ISWAP'])
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [1, 0], 3).toarray()
    unitary_true = np.kron(np.eye(2), gate_matrix['ISWAP'])
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [1, 2], 4).toarray()
    unitary_true = np.kron(np.eye(2), np.kron(gate_matrix['ISWAP'], np.eye(2)))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [3, 2], 4).toarray()
    unitary_true = np.kron(gate_matrix['ISWAP'], np.eye(4))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [2, 3], 4).toarray()
    unitary_true = np.kron(gate_matrix['ISWAP'], np.eye(4))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [0, 3], 4).toarray()
    swap_01 = np.kron(np.eye(4), gate_matrix['SWAP'])
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12.dot(swap_01)
    V = np.kron(gate_matrix['ISWAP'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [3, 0], 4).toarray()
    swap_01 = np.kron(np.eye(4), gate_matrix['SWAP'])
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12.dot(swap_01)
    V = np.kron(gate_matrix['ISWAP'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [1, 3], 4).toarray()
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['ISWAP'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['ISWAP'], [3, 1], 4).toarray()
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['ISWAP'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['CNOT'], [3, 1], 4).toarray()
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['CNOT'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)

    unitary_test = apply_gate(gate_matrix['SWAP'], [3, 1], 4).toarray()
    swap_12 = np.kron(np.eye(2), np.kron(gate_matrix['SWAP'], np.eye(2)))
    swapper = swap_12
    V = np.kron(gate_matrix['SWAP'], np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_single_qubit_gates():
    test_unitary = apply_gate(gate_matrix['H'], 0, 4).toarray()
    true_unitary = np.kron(np.eye(8), gate_matrix['H'])
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 1, 4).toarray()
    true_unitary = np.kron(np.eye(4), np.kron(gate_matrix['H'], np.eye(2)))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 2, 4).toarray()
    true_unitary = np.kron(np.eye(2), np.kron(gate_matrix['H'], np.eye(4)))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 3, 4).toarray()
    true_unitary = np.kron(gate_matrix['H'], np.eye(8))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 0, 5).toarray()
    true_unitary = np.kron(np.eye(2**4), gate_matrix['H'])
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 1, 5).toarray()
    true_unitary = np.kron(np.eye(2**3), np.kron(gate_matrix['H'], np.eye(2)))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 2, 5).toarray()
    true_unitary = np.kron(np.eye(2**2), np.kron(gate_matrix['H'], np.eye(2**2)))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 3, 5).toarray()
    true_unitary = np.kron(np.eye(2**1), np.kron(gate_matrix['H'], np.eye(2**3)))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = apply_gate(gate_matrix['H'], 4, 5).toarray()
    true_unitary = np.kron(np.eye(2**0), np.kron(gate_matrix['H'], np.eye(2**4)))
    assert np.allclose(test_unitary, true_unitary)


def test_tensor_gates_single_qubit():
    prog = Program().inst([Hgate(0)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 1).toarray()
    true_unitary = gate_matrix['H']
    assert np.allclose(test_unitary, true_unitary)

    prog = Program().inst([Hgate(0)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 5).toarray()
    true_unitary = np.kron(np.eye(2**4), gate_matrix['H'])
    assert np.allclose(test_unitary, true_unitary)

    prog = Program().inst([RXgate(0.2)(3)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 5).toarray()
    true_unitary = np.kron(np.eye(2**1), np.kron(gate_matrix['RX'](0.2),  np.eye(2**3)))
    assert np.allclose(test_unitary, true_unitary)

    prog = Program().inst([RXgate(0.5)(4)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 5).toarray()
    true_unitary = np.kron(np.eye(2**0), np.kron(gate_matrix['RX'](0.5),  np.eye(2**4)))
    assert np.allclose(test_unitary, true_unitary)


def test_tensor_gates_two_qubit():
    prog = Program().inst([CNOTgate(0, 1)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 4).toarray()
    true_unitary = apply_gate(gate_matrix['CNOT'], [0, 1], 4).toarray()
    assert np.allclose(test_unitary, true_unitary)

    prog = Program().inst([CNOTgate(1, 0)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 4).toarray()
    true_unitary = apply_gate(gate_matrix['CNOT'], [1, 0], 4).toarray()
    assert np.allclose(test_unitary, true_unitary)

    prog = Program().inst([CNOTgate(1, 3)])
    test_unitary = tensor_gates(gate_matrix, {}, prog.actions[0][1], 4).toarray()
    true_unitary = apply_gate(gate_matrix['CNOT'], [1, 3], 4).toarray()
    assert np.allclose(test_unitary, true_unitary)


def test_tensor_up_error_catch():
    """Testing tensor up type checking"""
    x_term = PauliTerm("X", 5)

    # testing type rejection
    with pytest.raises(TypeError):
        tensor_up(x_term, 5)

    # testing index rejection
    with pytest.raises(IndexError):
        tensor_up(PauliSum([x_term]), 3)


def test_tensor_up_correctness():
    """Check the correctness of the tensor up routine"""
    xy_term = PauliSum([PauliTerm("X", 0)*PauliTerm("Y", 1)])

    # test correctness
    trial_matrix = tensor_up(xy_term, 2)
    true_matrix = np.kron(gate_matrix['Y'], gate_matrix['X'])
    np.testing.assert_allclose(trial_matrix, true_matrix)

    x1_term = PauliSum([PauliTerm("X", 1)])
    trial_matrix = tensor_up(x1_term, 2)
    true_matrix = np.kron(gate_matrix['X'], gate_matrix['I'])
    np.testing.assert_allclose(trial_matrix, true_matrix)

    zpz_term = PauliTerm("Z", 0) + PauliTerm("Z", 1)
    trial_matrix = tensor_up(zpz_term, 2)
    true_matrix = np.zeros((4, 4))
    true_matrix[0, 0] = 2
    true_matrix[-1, -1] = -2
    np.testing.assert_allclose(trial_matrix, true_matrix)
