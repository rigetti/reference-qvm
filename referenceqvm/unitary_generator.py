#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""
Utility functions for generating gates for evolving states on the full Hilbert
space for qubits.

Note: uses SciPy sparse diagonal (DIA) representation to increase space and
timeefficiency.
"""
from collections import Sequence
from numbers import Integral
import warnings

import scipy.sparse as sps
from pyquil.quilbase import *
from pyquil.paulis import PauliSum

from referenceqvm.gates import gate_matrix

"""
If True, only physically-implementable operations allowed!
i.e. local SWAPS only (topology of QPU is periodic with nearest-neighbor gate
operations allowed, and a qubit architecture may be input as needed)

For now, implicitly assumes a linear chain of qubit connectivity, for ease &
guaranteed termination in swap algorithm. Arbitrary SWAP operations to be
implemented in a future release.
"""
topological_QPU = False


def lifted_gate(i, matrix, num_qubits):
    """
    Lifts input k-qubit gate on adjacent qubits starting from qubit i
    to complete Hilbert space of dimension 2 ** num_qubits.

    Ex: 1-qubit gate, lifts from qubit i
    Ex: 2-qubit gate, lifts from qubits (i+1, i)
    Ex: 3-qubit gate, lifts from qubits (i+2, i+1, i), operating in that order

    In general, this takes a k-qubit gate (2D matrix 2^k x 2^k) and lifts
    it to the complete Hilbert space of dim 2^num_qubits, as defined by
    the rightward tensor product (1) in arXiv:1608.03355.

    Note that while the qubits are addressed in decreasing order,
    starting with num_qubit - 1 on the left and ending with qubit 0 on the
    right (in a little-endian fashion), gates are still lifted to apply
    on qubits in increasing index (right-to-left) order.

    :param int i: starting qubit to lift matrix from (incr. index order)
    :param np.array matrix: the matrix to be lifted
    :param int num_qubits: number of overall qubits present in space

    :return: matrix representation of operator acting on the
        complete Hilbert space of all num_qubits.
    :rtype: sparse_array
    """
    # input is checked in parent function apply_gate()
    # Find gate size (number of qubits operated on)
    if (matrix.shape[0] & matrix.shape[0] - 1) != 0:
        raise TypeError("Invalid gate size. Must be power of 2! "
                        "Received {} size".format(matrix.shape))
    else:
        gate_size = np.log2(matrix.shape[0])
    # Is starting gate index out of range?
    if not (0 <= i < num_qubits + 1 - gate_size):
        raise ValueError("Gate index out of range!")

    # Outer-product to lift gate to complete Hilbert space
    # bottom: i qubits below target
    bottom_matrix = sps.eye(2 ** i).astype(np.complex128)
    # top: Nq - i (bottom) - gate_size (gate) qubits above target
    top_qubits = num_qubits - i - gate_size
    top_matrix = sps.eye(2 ** top_qubits).astype(np.complex128)
    return sps.kron(top_matrix, sps.kron(matrix, bottom_matrix))


def swap_inds_helper(i, j, arr):
    """
    Swaps indices in array, in-place.

    :param int i: index 1
    :param int j: index 2
    :param array-like arr: {list, np.array} array to be modified in-place
    """
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


def two_swap_helper(j, k, num_qubits, qubit_map):
    """
    Generate the permutation matrix that permutes two single-particle Hilbert
    spaces into adjacent positions.

    ALWAYS swaps j TO k. Recall that Hilbert spaces are ordered in decreasing
    qubit index order. Hence, j > k implies that j is to the left of k.

    End results:
        j == k: nothing happens
        j > k: Swap j right to k, until j at ind (k) and k at ind (k+1).
        j < k: Swap j left to k, until j at ind (k) and k at ind (k-1).

    Done in preparation for arbitrary 2-qubit gate application on ADJACENT
    qubits.

    :param int j: starting qubit index
    :param int k: ending qubit index
    :param int num_qubits: number of qubits in Hilbert space
    :param np.array qubit_map: current index mapping of qubits

    :return: tuple of swap matrix for the specified permutation,
             and the new qubit_map, after permutation is made
    :rtype: tuple (np.array, np.array)
    """
    if not (0 <= j < num_qubits and 0 <= k < num_qubits):
        raise ValueError("Permutation SWAP index not valid")

    perm = sps.eye(2 ** num_qubits).astype(np.complex128)
    new_qubit_map = np.copy(qubit_map)

    if j == k:
        # nothing happens
        return perm, new_qubit_map
    elif j > k:
        # swap j right to k, until j at ind (k) and k at ind (k+1)
        for i in range(j, k, -1):
            perm = lifted_gate(i - 1, gate_matrix['SWAP'], num_qubits)\
                          .dot(perm)
            swap_inds_helper(i - 1, i, new_qubit_map)
    elif j < k:
        # swap j left to k, until j at ind (k) and k at ind (k-1)
        for i in range(j, k, 1):
            perm = lifted_gate(i, gate_matrix['SWAP'], num_qubits).dot(perm)
            swap_inds_helper(i, i + 1, new_qubit_map)

    return perm, new_qubit_map


def permutation_arbitrary(args, num_qubits):
    """
    Generate the permutation matrix that permutes an arbitrary number of
    single-particle Hilbert spaces into adjacent positions.

    Transposes the qubit indices in the order they are passed to a
    contiguous region in the complete Hilbert space, in increasing
    qubit index order (preserving the order they are passed in).

    Gates are usually defined as `GATE 0 1 2`, with such an argument ordering
    dictating the layout of the matrix corresponding to GATE. If such an
    instruction is given, actual qubits (0, 1, 2) need to be swapped into the
    positions (2, 1, 0), because the lifting operation taking the 8 x 8 matrix
    of GATE is done in the little-endian (reverse) addressed qubit space.

    For example, suppose I have a Quil command CCNOT 20 15 10.
    The median of the qubit indices is 15 - hence, we permute qubits
    [20, 15, 10] into the final map [16, 15, 14] to minimize the number of
    swaps needed, and so we can directly operate with the final CCNOT, when
    lifted from indices [16, 15, 14] to the complete Hilbert space.

    Notes: assumes qubit indices are unique (assured in parent call).

    See documentation for further details and explanation.

    Done in preparation for arbitrary gate application on
    adjacent qubits.

    :param Sequence args: (int) Qubit indices in the order the gate is
        applied to.
    :param int num_qubits: Number of qubits in system

    :return:
        perm - permutation matrix providing the desired qubit reordering
        qubit_arr - new indexing of qubits presented in left to right
            decreasing index order. Should be identical to passed 'args'.
        start_i - starting index to lift gate from
    :rtype:  tuple (sparse_array, np.array, int)
    """
    # Don't permit NoneType or empty sequences, but allow 0
    if isinstance(args, Sequence):
        if not args:
            raise ValueError("Need at least one qubit index to perform"
                             "permutation")
    else:
        args = [args]

    inds = np.array([value_get(x) for x in args])
    for ind in inds:
        if not (0 <= ind < num_qubits):
            raise ValueError("Permutation SWAP index not valid")

    # Begin construction of permutation
    perm = sps.eye(2 ** num_qubits).astype(np.complex128)

    # First, sort the list and find the median.
    sort_i = np.argsort(inds)
    sorted_inds = inds[sort_i]
    med_i = int(len(sort_i) / 2)
    med = sorted_inds[med_i]

    # The starting position of all specified Hilbert spaces begins at
    # the qubit at (median - med_i)
    start = med - med_i
    # Array of final indices the arguments are mapped to, from
    # high index to low index, left to right ordering
    final_map = np.arange(start, start + len(inds))[::-1]
    start_i = final_map[-1]

    # Note that the lifting operation takes a k-qubit gate operating
    # on the qubits i+k-1, i+k-2, ... i (left to right).
    # two_swap_helper can be used to build the
    # permutation matrix by filling out the final map by sweeping over
    # the args from left to right and back again, swapping qubits into
    # position. we loop over the args until the final mapping matches
    # the argument.
    qubit_arr = np.arange(num_qubits)  # current qubit indexing

    made_it = False
    right = True
    while not made_it:
        array = range(len(inds)) if right else range(len(inds))[::-1]
        for i in array:
            pmod, qubit_arr = two_swap_helper(np.where(qubit_arr == inds[i])[0][0],
                                              final_map[i], num_qubits,
                                              qubit_arr)

            # update permutation matrix
            perm = pmod.dot(perm)
            if np.allclose(qubit_arr[final_map[-1]:final_map[0] + 1][::-1], inds):
                made_it = True
                break

        # for next iteration, go in opposite direction
        right = not right

    assert np.allclose(qubit_arr[final_map[-1]:final_map[0] + 1][::-1], inds)

    return perm, qubit_arr[::-1], start_i


def permutation_arbitrary_swap(args, num_qubits):
    """
    Not yet implemented.
    """
    raise NotImplementedError("Arbitrary topological QPU not yet implemented")


def apply_gate(matrix, args, num_qubits):
    """
    Apply k-qubit gate of size (2**k, 2**k) on the qubits in the order passed
    in args. e.g. GATE(arg[0], arg[1], ... arg[k-1]).

    If topological_QPU is True, we use local SWAP gates only as allowed by the
    qubit architecture --- as detailed in
    permutation_arbitrary() --- to permute the gate arguments to be adjacent to
    each other, and then lift the gate to the complete Hilbert space and
    perform the multiplication.

    :param np.array matrix: matrix specification of GATE
    :param tuple args: (int) qubit indices to operate gate on
    :param int num_qubits: number of qubits overall

    :return: transformed gate that acts on the specified qubits
    :rtype: np.array
    """
    if not isinstance(num_qubits, Integral) or num_qubits < 1:
        raise ValueError("Improper number of qubits passed.")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise TypeError("Gate array must be two-dimensional and "
                        "square matrix.")

    # Find gate size (number of qubits operated on)
    if (matrix.shape[0] & matrix.shape[0] - 1) != 0:
        raise TypeError("Invalid gate size. Must be power of 2! "
                        "Received {} size".format(matrix.shape))
    else:
        gate_size = int(np.log2(matrix.shape[0]))

    # Is gate size proper?
    if not (1 <= gate_size <= num_qubits):
        raise TypeError("Invalid gate size. k-qubit gates supported, for "
                        "k in [1, num_qubits]")

    if not topological_QPU:
        # use local SWAPs
        pi_permutation_matrix, final_map, start_i = permutation_arbitrary(args, num_qubits)
    else:
        # assume fully-connected, arbitrary SWAPs allowed
        raise NotImplementedError("Arbitrary SWAPs not yet implemented")

    # Transform qubit indices into ints
    if isinstance(args, Sequence):
        args = tuple(value_get(x) for x in args)
    else:
        args = value_get(args)

    if start_i:
        assert np.allclose(final_map[- gate_size - start_i: - start_i],
                           np.array(args))
    else:
        assert np.allclose(final_map[- gate_size - start_i:], np.array(args))

    v_matrix = lifted_gate(start_i, matrix, num_qubits)
    return np.dot(np.conj(pi_permutation_matrix.T),
                  np.dot(v_matrix, pi_permutation_matrix))


def tensor_gates(gate_set, defgate_set, pyquil_gate, num_qubits):
    """
    Take a pyQuil_gate instruction (assumed in the Quil Standard Gate Set
    or in defined_gates dictionary), returns the unitary over the complete
    Hilbert space corresponding to the instruction.

    :param dict gate_set: gate dictionary (name, matrix) pairs
    :param dict defgate_set: defined gate dictionary (name, matrix) pairs
    :param Instr pyquil_gate: Instruction object for pyQuil gate
    :param int num_qubits: number of qubits in Hilbert space

    :return: input gate lifted to full Hilbert space and applied
    :rtype: np.array
    """
    if pyquil_gate.operator_name in gate_set:
        # Input gate set. Assumed to be standard gate set.
        dict_check = gate_set
    elif pyquil_gate.operator_name in defgate_set:
        # defined_gates
        dict_check = defgate_set
    else:
        raise ValueError("Instruction (presumed a Gate or DefGate) is not "
                         "found in standard gate set or defined "
                         "gate set of program!")

    args = tuple(value_get(x) for x in pyquil_gate.arguments) \
            if dict_check == gate_matrix else tuple(pyquil_gate.arguments)

    if pyquil_gate.parameters:
        gate = apply_gate(dict_check[pyquil_gate.operator_name]
                          (*[value_get(p) for p in pyquil_gate.parameters]),
                          args,
                          num_qubits)
    else:
        gate = apply_gate(dict_check[pyquil_gate.operator_name],
                          args,
                          num_qubits)

    return gate


def tensor_up(pauli_terms, num_qubits):
    """
    Takes a PauliSum object along with a total number of
    qubits and returns a matrix corresponding the tensor representation of the
    object.

    Useful for generating the full Hamiltonian after a particular fermion to
    pauli transformation. For example:

    Converting a PauliSum X0Y1 + Y1X0 into the matrix

    .. code-block:: python

       [[ 0.+0.j,  0.+0.j,  0.+0.j,  0.-2.j],
       [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [ 0.+2.j,  0.+0.j,  0.+0.j,  0.+0.j]]


    :param pauli_terms: (PauliSum) object of PauliTerm
    :param num_qubits: (int) number of qubits in the system
    :returns: (numpy array) representation of the paui_terms operator
    """
    if not isinstance(pauli_terms, PauliSum):
        raise TypeError("can only tensor PauliSum")

    # check if operator is valid w.r.t the input number of qubits
    for term in pauli_terms.terms:
        if term._ops.keys():
            if max(term._ops.keys()) >= num_qubits:
                raise IndexError("pauli_terms has higher index than qubits")

    big_hilbert = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    # left kronecker product corresponds to the correct basis ordering
    for term in pauli_terms.terms:

        tmp_big_hilbert = np.array([1])
        for index in range(num_qubits):
            tmp_big_hilbert = np.kron(gate_matrix[term[index]], tmp_big_hilbert)

        big_hilbert += tmp_big_hilbert * term.coefficient

    return big_hilbert


def value_get(param_obj):
    """
    Function that returns the raw number / string stored in certain pyQuil
    objects.
    """
    if isinstance(param_obj, (float, int)):
        return param_obj
    elif isinstance(param_obj, AbstractQubit):
        return param_obj.index()
    elif isinstance(param_obj, Addr):
        return param_obj.address
    elif isinstance(param_obj, Slot):
        return param_obj.value()
    elif isinstance(param_obj, Label):
        return param_obj.name
