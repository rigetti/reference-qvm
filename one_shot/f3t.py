"""
An implementation of the Fermionic Fast Fourier Transform (F3T)

References:

"""
from math import log, pi
import numpy as np
import sys
import copy

from pyquil.gates import SWAP, CPHASE, X, I
from pyquil.quil import Program
from pyquil.paulis import PauliTerm, PauliSum, exponentiate, sX, sY, sZ

from grove.fermion_transforms.jwtransform import JWTransform


def fswap_generator(a, b, transform=JWTransform(), together=False):
    """
    Return a program that implements the fermionc swap between two qubits

    General form using fermionic creation/annihilation operators.
    Any transform can be specified to generate the unitary fswap.
    The order that the generators are mapped to circuits very much matters.
    First, number operators should be mapped.  Second, the cross terms should
    be mapped.

    :param Int a: fist qubit to swap
    :param Int b: second qubit to swap
    :param transform: Transform object from grove. Default is Jordan-Wigner
    :param Bool together: combine the number operator and cross terms into a single
                          Pauli term.
    """
    num_op_a = transform.create(a)*transform.kill(a)
    num_op_b = transform.create(b)*transform.kill(b)
    cross_op_ab = transform.create(a)*transform.kill(b)
    cross_op_ba = transform.create(b)*transform.kill(a)
    identity = PauliTerm("I", a)

    fswap_nums = -1*num_op_a + -1*num_op_b + identity
    fswap_cross =  cross_op_ab + cross_op_ba

    if together:
        return fswap_nums + fswap_cross

    return [fswap_nums, fswap_cross]

def fswap_unitary(a, b, transform=JWTransform()):
    """
    Give the fswap unitary for an arbitray pair of qubits (a, b)
    """
    fswap_nums, fswap_cross = fswap_generator(a, b, transform=transform, together=False)
    prog = Program()
    for term in fswap_nums.terms:
        prog += exponentiate((np.pi/2) * term)
    for term in fswap_cross.terms:
        prog += exponentiate((np.pi/2) * term)

    # phase fixup by -1j
    prog += exponentiate(PauliTerm("I", 0, -np.pi/2))

    return prog

def fswap_native(a, b):
    """
    Return a program that implements the fermionc swap between two qubits

    This function returns fswap in the most natural gates for
    most superconducting qubit architectures.

    :param Int a: fist qubit to swap
    :param Int b: second qubit to swap
    """
    prog = Program()
    prog.inst([CPHASE(np.pi)(a, b), SWAP(a, b)])

    return prog


def permutations(n_qubits, compilation=True):
    """
    Swaps defined by bitreversal operations.  Aggregates swaps from recursive FFT algorithm

    :param Int n_qubits: Number of qubits to perform F3T on
    :param Bool compilation: If true, use the native fswap definition.  If false, generate fswap from generators
    """
    swap_program = Program()

    int_justify = int(log(n_qubits, 2))
    for ii in xrange(n_qubits/2):
        binary = bin(ii)[2:].rjust(int_justify,'0')
        if ii != int(binary[::-1], 2):
            if compilation:
                swap_program += fswap_native(ii, int(binary[::-1], 2))
            else:
                swap_program += fswap_unitary(ii, int(binary[::-1], 2))

    return swap_program


def partition_sets(qubits):
    """
    partition a list of qubits into an even and odd set by index
    """
    even = qubits[::2]
    odd = qubits[1::2]
    return even, odd


def butterfly_sets(qubits):
    """
    partition a list into two sets to zip together via butterfly
    """
    dim = len(qubits)
    even = qubits[:dim/2]
    odd = qubits[dim/2:]
    return even, odd



def even_odd_combine(a, b, transform=JWTransform()):
    """
    Combine even and odd subpieces of the FFT with the F0d gate

    Note: Remember that the general exponentiation scheme in pyquil premultiplies pauli terms by
    -1j.  So we'll have a spurious negative sign if we don't account for this
    """
    fswap_num, fswap_cross = fswap_generator(a, b, transform=transform, together=False)
    number_op_a = transform.create(a)*transform.kill(a)
    number_op_b = transform.create(b)*transform.kill(b)
    F0d_prog = Program()
    for term in number_op_b.terms:
        F0d_prog += exponentiate((np.pi/2)*term)

    for term in fswap_num.terms:
        F0d_prog += exponentiate((-np.pi/4)*term)
    for term in fswap_cross.terms:
        F0d_prog += exponentiate((-np.pi/4)*term)

    for term in number_op_a.terms:
        F0d_prog += exponentiate((-np.pi/4)*term)
    for term in number_op_b.terms:
        F0d_prog += exponentiate((np.pi/4)*term)

    return F0d_prog



def twiddle_factor(k, m, index, transform=JWTransform()):
    """
    Return circuit for twiddle factor on index

    :param k: numerator of fraction of 2 pi
    :param m: denominator of fraction of 2 pi
    :param index: index of qubit to twiddle
    :param transform: (Optional) Default JWTransform.  Transformation
                      of Fermion operator to qubits.
    """
    number_op_b = (2 * np.pi * float(k) / float(m)) * transform.create(index)*transform.kill(index)
    twiddle = Program()
    for term in number_op_b.terms:
        twiddle += exponentiate(term)

    return twiddle


def butterfly(a, b, k, m, transform=JWTransform()):
    """
    butterfly two qubits together
:param Int a: qubit index a
    :param Int b: qubit index b that twiddle is applied to
    :param transform: Transform object from grove. Default is Jordan-Wigner
    """
    prog = even_odd_combine(a, b)
    twiddle_prog = twiddle_factor(k, m, b)
    return prog + twiddle_prog


def recursive_fft(qubits):
    """
    Compute the fermionic fast Fourier transform

    :param qubits: list of qubits to perform the FFFT on.
    :returns: program representing F3T
    """
    # check for non empty list and length is a power of 2
    if len(qubits) == 0 or len(qubits) & (len(qubits) - 1) != 0:
        raise ValueError("Must specify the number of qubits in powers of 2")

    qubit_map = dict(zip(qubits, range(len(qubits))))
    prog = recursive_butterfly(range(len(qubits)))
    prog_perms = permutations(len(qubits))

    unmapped_program = prog + prog_perms
    return map_qubits(qubit_map, unmapped_program)


def map_qubits(qubit_map, program_to_map):
    """
    Translate a program to a different set of qubits
    """
    qubit_map_inv = dict(zip(qubit_map.values(), qubit_map.keys()))
    mapped_program = Program()
    for gate in program_to_map:
        new_gate = copy.deepcopy(gate[1])
        for arg in new_gate.arguments:
            arg._index = qubit_map_inv[arg._index]

        mapped_program.inst(new_gate)

    return mapped_program

def recursive_butterfly(qubits):
    """
    Recursive FFT algorithm

    Butterfly -> FT(N/2) odds FT(N/2) evens -> permutations.
    Qubit ordering is [0, k] evens and [0, k] odds.
    """
    FFT_prog = Program()

    if len(qubits) == 1:
        return FFT_prog

    else:
        even_set, odd_set = butterfly_sets(qubits)
        for index, (e_qubit, o_qubit) in enumerate(zip(even_set, odd_set)):
            FFT_prog += butterfly(e_qubit, o_qubit, index, len(even_set)*2)

        FFT_prog += recursive_butterfly(even_set)
        FFT_prog += recursive_butterfly(odd_set)
        return FFT_prog

if __name__ == "__main__":
    # imagine two 4-color complete graphs [0, 1, 2, 3] and [4, 5, 6, 7]
    prog_1 = recursive_fft(range(4, 4 + 4))
    prog_2 = recursive_fft(range(4))
    prog = prog_1 + prog_2
    start_prog = Program()
    start_prog.inst([X(0), X(4)])

    delocalized_fermions = start_prog + prog

    # the resulting state should have delocalized fermions
    # psi = \sum_{i \in [1, 4], j \in [5, 8]}a_{i}^{\dagger}a_{j}^{\dagger}|0\rangle

    # example execution with referenceqvm
    from referenceqvm.reference import Connection
    from pyquil.wavefunction import Wavefunction
    rqvm = Connection(type='wavefunction')
    wf, _ = rqvm.wavefunction(delocalized_fermions)
    wf = Wavefunction(wf)
    print wf
