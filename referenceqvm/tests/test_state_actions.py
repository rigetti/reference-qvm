"""
Test the infrastructure for building a state by projection onto the +1
eigenspace of a set of generators or stabilizers
"""
import numpy as np
from referenceqvm.state_actions import compute_action, project_stabilized_state
from referenceqvm.tests.test_stabilzier_qvm import five_qubit_code_generators
from pyquil.paulis import sX, sZ, sY, sI, PauliSum
import pytest


def test_compute_action_type_checks():
    """
    Make sure type checks are consistent and working
    """
    with pytest.raises(TypeError):
        compute_action([0, 0, 0, 0, 0], PauliSum([sX(0)]), 5)

    with pytest.raises(TypeError):
        compute_action([0, 0, 0, 0, 0], sX(0), 4)

    with pytest.raises(TypeError):
        compute_action(3, 'a', 4)

    with pytest.raises(TypeError):
        compute_action(-3, sX(0), 4)

    with pytest.raises(TypeError):
        compute_action('0001', sX(0), 4)


def test_compute_action_identity():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(4):
        pauli_term = sI(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        assert new_basis_state == comp_basis_state
        assert np.isclose(coeff, 1)


def test_compute_action_X():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(4):
        pauli_term = sX(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(4):
        pauli_term = sX(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii] ^ 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)


def test_compute_action_XX():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(3):
        pauli_term = sX(ii) * sX(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii + 1] = 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(3):
        pauli_term = sX(ii) * sX(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii] ^ 1
        true_basis_state[ii + 1] = true_basis_state[ii + 1] ^ 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)


def test_compute_action_Y():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(4):
        pauli_term = sY(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii] ^ 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1j)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(4):
        pauli_term = sY(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii] ^ 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, -1j)


def test_compute_action_YY():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(3):
        pauli_term = sY(ii) * sY(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii + 1] = 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, -1)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(3):
        pauli_term = sY(ii) * sY(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii] ^ 1
        true_basis_state[ii + 1] = true_basis_state[ii + 1] ^ 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, -1)


def test_compute_action_Z():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(4):
        pauli_term = sZ(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii]
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(4):
        pauli_term = sZ(ii)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] = true_basis_state[ii]
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, -1)


def test_compute_action_ZZ():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(3):
        pauli_term = sZ(ii) * sZ(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(3):
        pauli_term = sZ(ii) * sZ(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1)


def test_compute_action_XY():
    """
    Action of Pauli operators on state
    """
    comp_basis_state = [0, 0, 0, 0]
    for ii in range(3):
        pauli_term = sX(ii) * sY(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] ^= 1
        true_basis_state[ii + 1] ^= 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, 1j)

    comp_basis_state = [1, 1, 1, 1]
    for ii in range(3):
        pauli_term = sX(ii) * sY(ii + 1)
        new_basis_state, coeff = compute_action(comp_basis_state, pauli_term,
                                            len(comp_basis_state))
        # abuse of comparisons in python
        true_basis_state = comp_basis_state.copy()
        true_basis_state[ii] ^= 1
        true_basis_state[ii + 1] ^= 1
        assert new_basis_state == true_basis_state
        assert np.isclose(coeff, -1j)


def test_stabilizer_projection_Z():
    """
    test if we project out the correct state
    """
    stabilizer_state = project_stabilized_state([sZ(0)])
    true_state = np.zeros((2, 1))
    true_state[0, 0] = 1
    assert np.allclose(true_state, stabilizer_state.todense())


def test_stabilizer_projection_ZZ():
    """
    test if we project out the correct state
    """
    stabilizer_state = project_stabilized_state([sZ(0) * sZ(1), sX(0) * sX(1)])
    true_state = np.zeros((4, 1))
    true_state[0, 0] = true_state[3, 0] = 1
    true_state /= np.sqrt(2)
    assert np.allclose(true_state, stabilizer_state.todense())


def test_stabilizer_projection_ZZZ():
    """
    test if we project out the correct state
    """
    stabilizer_state = project_stabilized_state([sZ(0) * sZ(1), sZ(1) * sZ(2),
                                                     sX(0) * sX(1) * sX(2)])
    true_state = np.zeros((8, 1))
    true_state[0, 0] = true_state[7, 0] = 1
    true_state /= np.sqrt(2)
    assert np.allclose(true_state, np.array(stabilizer_state.todense()))

