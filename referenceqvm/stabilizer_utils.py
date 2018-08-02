"""
State actions on classical states

Commonly, in order to recover a state you need to compute the action
of Pauli operators on classical basis states.

In this module we provide infrastructure to do this for Pauli Operators from
pyquil.

Given
"""
import sys
from functools import reduce
from scipy.sparse import csc_matrix
import numpy as np
from pyquil.paulis import PauliTerm, sX, sZ, sY


def compute_action(classical_state, pauli_operator, num_qubits):
    """
    Compute action of Pauli opertors on a classical state

    The classical state is enumerated as the left most bit is the least-significant
    bit.  This is how one usually reads off classical data from the QVM.  Not
    how the QVM stores computational basis states.

    :param classical_state: binary repr of a state or an integer.  Should be
                            left most bit (0th position) is the most significant bit
    :param num_qubits:
    :return: new classical state and the coefficient it picked up.
    """
    if not isinstance(pauli_operator, PauliTerm):
        raise TypeError("pauli_operator must be a PauliTerm")

    if not isinstance(classical_state, (list, int)):
        raise TypeError("classical state must be a list or an integer")

    if isinstance(classical_state, int):
        if classical_state < 0:
            raise TypeError("classical_state must be a positive integer")

        classical_state = list(map(int, np.binary_repr(classical_state,
                                                       width=num_qubits)))
    if len(classical_state) != num_qubits:
        raise TypeError("classical state not long enough")

    # iterate through tensor elements of pauli_operator
    new_classical_state = classical_state.copy()
    coefficient = 1
    for qidx, telem in pauli_operator:
        if telem == 'X':
            new_classical_state[qidx] = new_classical_state[qidx] ^ 1
        elif telem == 'Y':
            new_classical_state[qidx] = new_classical_state[qidx] ^ 1
            # set coeff
            if new_classical_state[qidx] == 0:
                coefficient *= -1j
            else:
                coefficient *= 1j

        elif telem == 'Z':
            # set coeff
            if new_classical_state[qidx] == 1:
                coefficient *= -1

    return new_classical_state, coefficient


def state_family_generator(state, pauli_operator):
    """
    Generate a new state by applying the pauli_operator to each computational bit-string

    This is accomplished in a sparse format where a sparse vector is returned
    after the action is accumulate in a new list of data and indices

    :param state:
    :param pauli_operator:
    :return:
    """
    if not isinstance(state, csc_matrix):
        raise TypeError("we only take csc_matrix")

    num_qubits = int(np.log2(state.shape[0]))
    new_coeffs = []
    new_indices = []

    # iterate through non-zero
    rindices, cindices = state.nonzero()
    for ridx, cidx in zip(rindices, cindices):
        # this is so gross looking
        bitstring = list(map(int, np.binary_repr(ridx, width=num_qubits)))[::-1]
        new_ket, new_coefficient = compute_action(bitstring, pauli_operator, num_qubits)
        new_indices.append(int("".join([str(x) for x in new_ket[::-1]]), 2))
        new_coeffs.append(state[ridx, cidx] * new_coefficient * pauli_operator.coefficient)

    return csc_matrix((new_coeffs, (new_indices, [0] * len(new_indices))),
                      shape=(2 ** num_qubits, 1), dtype=complex)


def project_stabilized_state(stabilizer_list, num_qubits=None,
                             classical_state=None):
    """
    Project out the state stabilized by the stabilizer matrix

    |psi> = (1/2^{n}) * Product_{i=0}{n-1}[ 1 + G_{i}] |vac>

    :param stabilizer_list:
    :param num_qubits: integer number of qubits
    :param classical_state: Default None.  Defaults to |+>^{otimes n}

    :return: state projected by stabilizers
    """
    if num_qubits is None:
        num_qubits = len(stabilizer_list)

    if classical_state is None:
        indptr = np.array([0] * (2 ** num_qubits))
        indices = np.arange(2 ** num_qubits)
        data = np.ones((2 ** num_qubits)) / np.sqrt((2 ** num_qubits))
    else:
        if not isinstance(classical_state, list):
            raise TypeError("I only accept lists as the classical state")
        if len(classical_state) != num_qubits:
            raise TypeError("Classical state does not match the number of qubits")

        # convert into an integer
        ket_idx = int("".join([str(x) for x in classical_state[::-1]]), 2)
        indptr = np.array([0])
        indices = np.array([ket_idx])
        data = np.array([1.])

    state = csc_matrix((data, (indices, indptr)), shape=(2 ** num_qubits, 1),
                       dtype=complex)

    for generator in stabilizer_list:
        # (I + G(i)) / 2
        state += state_family_generator(state, generator)
        state /= 2

    normalization = (state.conj().T.dot(state)).todense()
    state /= np.sqrt(float(normalization.real))  # this is needed or it will cast as a np.matrix

    return state


def pauli_stabilizer_to_binary_stabilizer(stabilizer_list):
    """
    Convert a list of stabilizers represented as PauliTerms to a binary tableau form

    :param List stabilizer_list: list of stabilizers where each element is a PauliTerm
    :return: return an integer matrix representing the stabilizers where each row is a
             stabilizer.  The size of the matrix is n x (2 * n) where n is the maximum
             qubit index.
    """
    if not all([isinstance(x, PauliTerm) for x in stabilizer_list]):
        raise TypeError("At least one element of stabilizer_list is not a PauliTerm")

    max_qubit = max([max(x.get_qubits()) for x in stabilizer_list]) + 1
    stabilizer_tableau = np.zeros((len(stabilizer_list), 2 * max_qubit + 1), dtype=int)
    for row_idx, term in enumerate(stabilizer_list):
        for i, pterm in term:  # iterator for each tensor-product element of the Pauli operator
            if pterm == 'X':
                stabilizer_tableau[row_idx, i] = 1
            elif pterm == 'Z':
                stabilizer_tableau[row_idx, i + max_qubit] = 1
            elif pterm == 'Y':
                stabilizer_tableau[row_idx, i] = 1
                stabilizer_tableau[row_idx, i + max_qubit] = 1
            else:
                # term is identity
                pass

        if not (np.isclose(term.coefficient, -1) or np.isclose(term.coefficient, 1)):
            raise ValueError("stabilizers must have a +/- coefficient")

        if int(np.sign(term.coefficient.real)) == 1:
            stabilizer_tableau[row_idx, -1] = 0
        elif int(np.sign(term.coefficient.real)) == -1:
            stabilizer_tableau[row_idx, -1] = 1
        else:
            raise TypeError('unrecognized on pauli term of stabilizer')

    return stabilizer_tableau


def binary_stabilizer_to_pauli_stabilizer(stabilizer_tableau):
    """
    Convert a stabilizer tableau to a list of PauliTerms

    :param stabilizer_tableau:  Stabilizer tableau to turn into pauli terms
    :return: a list of PauliTerms representing the tableau
    :rytpe: List of PauliTerms
    """
    stabilizer_list = []
    num_qubits = (stabilizer_tableau.shape[1] - 1) // 2
    for nn in range(stabilizer_tableau.shape[0]):  # iterate through the rows
        stabilizer_element = []
        for ii in range(num_qubits):
            if stabilizer_tableau[nn, ii] == 1 and stabilizer_tableau[nn, ii + num_qubits] == 0:
                stabilizer_element.append(sX(ii))
            elif stabilizer_tableau[nn, ii] == 0 and stabilizer_tableau[nn, ii + num_qubits] == 1:
                stabilizer_element.append(sZ(ii))
            elif stabilizer_tableau[nn, ii] == 1 and stabilizer_tableau[nn, ii + num_qubits] == 1:
                stabilizer_element.append(sY(ii))

        stabilizer_term = reduce(lambda x, y: x * y, stabilizer_element) * ((-1) ** stabilizer_tableau[nn, -1])
        stabilizer_list.append(stabilizer_term)
    return stabilizer_list


def symplectic_inner_product(vector1, vector2):
    """
    Operators commute if the symplectic inner product of their binary form is zero

    Operators anticommute if symplectic inner product of their binary form is one

    :param vector1: binary form of operator with no sign info
    :param vector2: binary form of a pauli operator with no sign info
    :return: 0, 1
    """
    if vector1.shape != vector2.shape:
        raise ValueError("vectors must be the same size.")

    # TODO: add a check for binary or integer linear arrays

    hadamard_product = np.multiply(vector1, vector2)
    return reduce(lambda x, y: x ^ y, hadamard_product)
