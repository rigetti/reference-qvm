import numpy as np
from gates import gate_matrix
from pyquil.paulis import PauliSum
from pyquil.quilbase import Slot


def lifted_two_gate(i, matrix, num_qubits):
    """
    lift SWAP on i, i+1 to Hilbert space of num_qubits
    """
    if i < 0 or i >= num_qubits - 1:
        raise ValueError("SWAP index out of range")
    if matrix.shape != (4, 4):
        raise TypeError("two-qubit operator incorrect shape.  Expected (4, 4) got {}".format(matrix.shape))
    top_qubits = num_qubits - i - 2  # 1 from count from 0, 1 from swap takes 2
    top_matrix = np.eye(2**top_qubits)
    bottom_matrix = np.eye(2**i)  # there are (i) qubits below the target
    return np.kron(top_matrix, np.kron(matrix, bottom_matrix))


def permutation_swaps(j, k, num_qubits):
    """
    Generate the permutation matrix that permutes two single particle Hilbert
    spaces into adjacent positions
    """
    if j >= num_qubits or k >= num_qubits or j < 0 or k < 0:
        raise ValueError("Permutation SWAP index not valid")

    swap_matrix = np.eye(2**num_qubits)
    if j == k:
        return swap_matrix
    elif j > k:
        for i in xrange(k, j - 1):
            swap_matrix = lifted_two_gate(i, gate_matrix['SWAP'], num_qubits).dot(swap_matrix)
    elif j < k:
        for i in xrange(j, k - 1):
            swap_matrix = lifted_two_gate(i, gate_matrix['SWAP'], num_qubits).dot(swap_matrix)
        swap_matrix = lifted_two_gate(k - 1, gate_matrix['SWAP'], num_qubits).dot(swap_matrix)

    return swap_matrix


def apply_two_qubit(matrix, args, num_qubits):
    """
    Apply the two-qubit gate between qubits args[0] = i and args[1] = j
    """
    i, j = args[0], args[1]
    pi_permutation_matrix = permutation_swaps(i, j, num_qubits)
    # lifted_two_gate always applys to i, i + 1.
    v_matrix = lifted_two_gate(max(i, j) - 1, matrix, num_qubits)
    return np.dot(np.conj(pi_permutation_matrix.T),
                  np.dot(v_matrix, pi_permutation_matrix))


def tensor_single_qubit_op(operator, index, n_qubits):
    """tensors up single qubit operator at the `index` position"""
    top_qubits = np.eye(2**(n_qubits - 1 - index))
    bottom_qubits = np.eye(2**index)
    return np.kron(top_qubits, np.kron(operator, bottom_qubits))


def tensor_two_qubit_op(args, n_qubits):
    """tensors CNOT operator at the control in args[0] and target args[1] """

    # cnot is I_{t} \otimes P0_{c} + X_{t} \otimes P1_{c}
    # I_{t} \otimes P0_{c}
    first_term = np.array([1])
    for i in range(n_qubits):

        # control
        if i == args[0]:
            first_term = np.kron(gate_matrix['P0'], first_term)
        else:
            first_term = np.kron(gate_matrix['I'], first_term)

    # X_{t} \otimes P1_{c}
    second_term = np.array([1])
    for i in range(n_qubits):

        if i == args[0]:
            second_term = np.kron(gate_matrix['P1'], second_term)
        elif i == args[1]:
            second_term = np.kron(gate_matrix['X'], second_term)
        else:
            second_term = np.kron(gate_matrix['I'], second_term)

    return first_term + second_term


def tensor_gates(quil_gate, n_qubits):
    """
    Take a pyquil_gate instruction and return a unitary
    """
    def value(param_obj):
        if isinstance(param_obj, (float, int, long)):
            return param_obj
        elif isinstance(param_obj, Slot):
            return param_obj.value()

    if quil_gate.operator_name not in gate_matrix.keys():
        raise ValueError("pyQuil gate specified is not in the gate_matrix dictionary")

    if len(quil_gate.arguments) == 1:
        if len(quil_gate.parameters) != 0:
            gate = tensor_single_qubit_op(gate_matrix[quil_gate.operator_name](quil_gate.parameters[0]),
                                          quil_gate.arguments[0]._index, n_qubits)
        else:
            gate = tensor_single_qubit_op(gate_matrix[quil_gate.operator_name],
                                          quil_gate.arguments[0]._index, n_qubits)

    elif len(quil_gate.arguments) == 2:
        if len(quil_gate.parameters) != 0:
            gate = apply_two_qubit(gate_matrix[quil_gate.operator_name](quil_gate.parameters[0]),
                                   map(lambda x: x._index, quil_gate.arguments), n_qubits)
        else:
            gate = apply_two_qubit(gate_matrix[quil_gate.operator_name],
                                   map(lambda x: x._index, quil_gate.arguments), n_qubits)
    else:
        raise TypeError("We only take one- and two-qubit gates right now")

    return gate


def tensor_up(pauli_terms, num_qubits):
    """
    Takes a PauliSum object along with a total number of
    qubits and returns a matrix corresponding the tensor representation of the
    object.

    Useful for generating the full Hamiltonian after a particular fermion to
    pauli transformation.

    :param pauli_terms: (PauliSum) object of PauliTerm
    :param num_qubits: (int) number of qubits in the system
    :returns: (numpy array) representation of the paui_terms operator
    """
    assert isinstance(pauli_terms, PauliSum), "can only tensor PauliSum"

    if __debug__:
        for term in pauli_terms.terms:
            if len(term._ops.keys()) > 0:
                assert max(term._ops.keys()) < num_qubits, "pauli_terms has " \
                                                           "higher index than qubits"

    big_hilbert = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    # left kronecker product corresponds to the correct basis ordering
    for term in pauli_terms.terms:
        tmp_big_hilbert = np.array([1])

        for index in xrange(num_qubits):
            pauli_mat = gate_matrix[term[index]]

            tmp_big_hilbert = np.kron(pauli_mat, tmp_big_hilbert)

        tmp_big_hilbert = tmp_big_hilbert * term.coefficient

        big_hilbert = big_hilbert + tmp_big_hilbert

    return big_hilbert
