"""
State actions on classical states

Commonly, in order to recover a state you need to compute the action
of Pauli operators on classical basis states.

In this module we provide infrastructure to do this for Pauli Operators from
pyquil.

Given
"""
import numpy as np
from pyquil.paulis import PauliTerm


def compute_action(classical_state, pauli_operator, num_qubits):
    """
    Compute action of Pauli opertors on a classical state

    :param classical_state: binary repr of a state or an integer.  Should be
                            left most bit (0th position) is the most significant bit
    :param num_qubits:
    :return:
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

