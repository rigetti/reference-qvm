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
Pure QVM that only executes pyQuil programs containing Clifford group generators,
and return the wavefunction or stabilizer
"""
import sys
from functools import reduce
from pyquil.quil import Program, get_classical_addresses_from_program
from pyquil.quilbase import *
from pyquil.paulis import PauliTerm, sI, sZ, sX, sY

from referenceqvm.qam import QAM
from referenceqvm.gates import stabilizer_gate_matrix
from referenceqvm.unitary_generator import value_get


class QVM_Stabilizer(QAM):
    """
    A  P Y T H O N

    Q U A N T U M

    V I R T U A L

    M A C H I N E

    S I M U L A T I N G

    S T A B I L I Z E R

    S T A T E S
    """
    def __init__(self, num_qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=stabilizer_gate_matrix,
                 defgate_set=None):
        """
        Subclassed from QAM this is a pure QVM.
        """
        super(QVM_Stabilizer, self).__init__(num_qubits=num_qubits, program=program,
                                             program_counter=program_counter,
                                             classical_memory=classical_memory,
                                             gate_set=gate_set,
                                             defgate_set=defgate_set)
        # this sets that arbitrary instructions are not allowed
        # this can probably be factored out eventually.  It is a relic of the
        # the old days of generating unitaries of programs
        self.all_inst = False
        if num_qubits is None:
            self.tableau = None
        else:
            self.tableau = self._n_qubit_tableau(num_qubits)

    def load_program(self, pyquil_program):
        """
        Loads a pyQuil program into the QAM memory.
            Synthesizes pyQuil program into instruction list.
            Initializes program object and program counter.

        This overwrites the parent class load program.  Required because
        unitary check QVM checks makes it annoying to allow measurements.
        TODO: Change load_program for each qvm-subclass

        :param Program pyquil_program: program to be ran
        """
        # typecheck
        if not isinstance(pyquil_program, Program):
            raise TypeError("I can only generate from pyQuil programs")

        if self.all_inst is None:
            raise NotImplementedError("QAM needs to be subclassed in order to "
                                      "load program")

        # create defgate dictionary
        defined_gates = {}
        for dg in pyquil_program.defined_gates:
            defined_gates[dg.name] = dg.matrix
        self.defgate_set = defined_gates

        # if QVM_Unitary, check if all instructions are valid.
        invalid = False
        for instr in pyquil_program:
            if isinstance(instr, Gate):
                if not (instr.name in self.gate_set.keys() or instr.name in self.defgate_set.keys()):
                    invalid = True
                    break
            elif isinstance(instr, Measurement):
                pass
            else:
                invalid = True
                break

        # NOTE: all_inst is set by the subclass
        if invalid is True and self.all_inst is False:
            raise TypeError("Some gates used are not allowed in this QAM")

        # set internal program and counter to their appropriate values
        self.program = pyquil_program
        self.program_counter = 0

        # setup quantum and classical memory
        q_max, c_max = self.identify_bits()
        if c_max <= 512:  # allocate at least 512 cbits (as floor)
            c_max = 512
        self.num_qubits = q_max
        self.classical_memory = np.zeros(c_max).astype(bool)

    def _n_qubit_tableau(self, num_qubits):
        """
        Construct an empty tableau for a n-qubit system

        :param num_qubits:
        :return:
        """
        tableau = np.zeros((2 * num_qubits, (2 * num_qubits) + 1),
                                dtype=int)

        # set up initial state |0>^{otimes n}
        for ii in range(2 * self.num_qubits):
            tableau[ii, ii] = 1
        return tableau

    def transition(self, instruction):
        """
        Implements a full transition, including pre/post noise hooks.

        :param QuilAction instruction: instruction to be executed

        :return: if program is halted TRUE, else FALSE
        :rtype: bool int
        """
        self.pre()
        self._transition(instruction)
        self.post()

        # return HALTED (i.e. program_counter is end of program)
        return self.program_counter == len(self.program)

    def pre(self):
        """
        Pre-transition hook - use for noisy evolution models. Unimplemented for now.
        """
        pass

    def post(self):
        """
        Post-transition hook - use for noisy evolution models. Unimplemented for now.
        """
        pass

    def _transition(self, instruction):
        """
        Implements a transition on the generator matrix representing the stabilizers

        :param Gate instruction: QuilAction gate to be implemented
        """
        if isinstance(instruction, Measurement):
            # mutates classical memory!  I should change this...
            self._apply_measurement(instruction)
            self.program_counter += 1

        elif isinstance(instruction, Gate):
            # apply Gate or DefGate
            if instruction.name == 'H':
                self._apply_hadamard(instruction)
            elif instruction.name == 'S':
                self._apply_phase(instruction)
            elif instruction.name == 'CNOT':
                self._apply_cnot(instruction)
            elif instruction.name == 'I':
                pass
            else:
                raise TypeError("We checked for correct gate types previously" +
                                " so the impossible has happened!")

            self.program_counter += 1

        elif isinstance(instruction, Jump):
            # unconditional Jump; go directly to Label
            self.program_counter = self.find_label(instruction.target)

        elif isinstance(instruction, JumpTarget):
            # Label; pass straight over
            self.program_counter += 1

    def run(self, pyquil_program, classical_addresses=None, trials=1):
        """
        Run program.

        Loads and checks program if all gates are within the stabilizer set.
        Then executes program

        :param porgram:
        :param classical_addresses:
        :param trials:
        :return:
        """
        self.load_program(pyquil_program)

        if classical_addresses is None:
            classical_addresses = get_classical_addresses_from_program(pyquil_program)

        results = []
        for _ in range(trials):
            # set up stabilizers
            self.tableau = self._n_qubit_tableau(self.num_qubits)
            self.kernel()
            results.append(list(map(int, self.classical_memory[classical_addresses])))

            # reset qvm
            self.memory_reset()
            self.program_counter = 0

        return results

    def stabilizer_tableau(self):
        """
        return the stabilizer part of the tableau

        :return: stabilizer matrix
        """
        return self.tableau[self.num_qubits:, :]

    def destabilizer_tableau(self):
        """
        Return the destabilizer part of the tableau

        :return: destabilizer matrix
        """
        return self.tableau[:self.num_qubits, :]

    def _rowsum(self, h, i):
        """
        Implementation of Aaronson-Gottesman rowsum algorithm

        :param Int h: row index 'h'
        :param Int i: row index 'i'
        :return:
        """
        # NOTE: this is left multiplication of P(i).  P(i) * P(h)
        phase_accumulator = self._rowsum_phase_accumulator(h, i)
        # now set the r_{h}
        if phase_accumulator % 4 == 0:
            self.tableau[h, -1] = 0
        elif phase_accumulator % 4 == 2:
            self.tableau[h, -1] = 1
        else:
            raise ValueError("An impossible value for the phase_accumulator has occurred")

        # now update the rows of the tableau
        for j in range(self.num_qubits):
            self.tableau[h, j] = self.tableau[i, j] ^ self.tableau[h, j]
            self.tableau[h, j + self.num_qubits] = self.tableau[i, j + self.num_qubits] ^ \
                                                   self.tableau[h, j + self.num_qubits]

    def _rowsum_phase_accumulator(self, h, i):
        """
        phase accumulator sub algorithm for the rowsum routine

        Note: this accumulates the $i$ phase for row_i * row_h  NOT row_h, row_i
        :param Int h: row index 'h'
        :param Int i: row index 'i'
        :return: phase mod 4
        """
        phase_accumulator = 0
        for j in range(self.num_qubits):
            # self._g_update(x_{hj}, z_{hj}, x_{ij}, z_{ij})
            phase_accumulator += self._g_update(self.tableau[i, j],
                                                self.tableau[i, self.num_qubits + j],
                                                self.tableau[h, j],
                                                self.tableau[h, self.num_qubits + j])
        phase_accumulator += 2 * self.tableau[h, -1]
        phase_accumulator += 2 * self.tableau[i, -1]
        return phase_accumulator % 4

    def _g_update(self, x1, z1, x2, z2):
        """
        function that takes 4 bits and returns the power of $i$ {0, 1, -1}

        when the pauli matrices represented by x1z1 and x2z2 are multiplied together

        :param x1:
        :param z1:
        :param x2:
        :param z2:
        :return:
        """
        # if the first term is identity
        if x1 == z1 == 0:
            return 0

        # if the first term is Y
        # Y * Z = (1 - 0) = 1j ^ { 1} = 1j
        # Y * I = (0 - 0) = 1j ^ { 0} = 1
        # Y * X = (0 - 1) = 1j ^ {-1} = -1j
        # Y * Y = (1 - 1) = 1j ^ { 0} = 1
        if x1 == z1 == 1:
            return z2 - x2

        # if the first term is X return z2 * (2 * x2 - 1)
        # X * I = (0 * (2 * 0 - 1) = 0  -> 1j^{ 0} =  1
        # X * X = (0 * (2 * 1 - 1) = 0  -> 1j^{ 0} =  1
        # X * Y = (1 * (2 * 1 - 1) = 1  -> 1j^{ 1} =  1j
        # X * Z = (1 * (2 * 0 - 1) = -1 -> 1j^{-1} = -1j
        if x1 == 1 and z1 == 0:
            return z2 * (2 * x2 - 1)

        # if the first term is Z return x2 * (1 - 2 * z2)
        # Z * I = (0 * (1 - 2 * 0)) = 0  -> 1j^{ 0} = 1
        # Z * X = (1 * (1 - 2 * 0)) = 1  -> 1j^{ 1} = 1j
        # Z * Y = (1 * (1 - 2 * 1)) = -1 -> 1j^{-1} = -1j
        # Z * Z = (0 * (1 - 2 * 1)) = 0  -> 1j^{0} = 1
        if x1 == 0 and z1 == 1:
            return x2 * (1 - 2 * z2)

        raise ValueError("we were unable to multiply the pauli operators together!")

    def _apply_cnot(self, instruction):
        """
        Apply a CNOT to the tableau

        :param instruction: pyquil abstract instruction.  Must have
        """
        a, b = list(instruction.get_qubits())  # control (a) and target (b)
        for i in range(2 * self.num_qubits):
            self.tableau[i, -1] = self._cnot_phase_update(i, a, b)
            self.tableau[i, b] = self.tableau[i, b] ^ self.tableau[i, a]
            self.tableau[i, a + self.num_qubits] = self.tableau[i, a + self.num_qubits] ^ self.tableau[i, b + self.num_qubits]

    def _cnot_phase_update(self, i, c, t):
        """
        update r_{i}

        :param i: tableau row index
        :param c: control qubit index
        :param t: target qubit index
        :return: 0/1 phase update for r_{i}
        """
        return self.tableau[i, -1] ^ (self.tableau[i, c] * self.tableau[i, t + self.num_qubits]) * (
                                  self.tableau[i, t] ^ self.tableau[i, c + self.num_qubits] ^ 1)

    def _apply_hadamard(self, instruction):
        """
        Apply a hadamard gate on qubit defined in instruction

        :param instruction:
        :return:
        """
        qubit_label = list(instruction.get_qubits())[0]
        for i in range(2 * self.num_qubits):
            self.tableau[i, -1] = self.tableau[i, -1] ^ (self.tableau[i, qubit_label] * self.tableau[i, qubit_label + self.num_qubits])
            self.tableau[i, [qubit_label, qubit_label + self.num_qubits]] = self.tableau[i, [qubit_label + self.num_qubits, qubit_label]]

    def _apply_phase(self, instruction):
        """
        Apply the phase gate instruction ot the tableau

        :param instruction:
        :return:
        """
        qubit_label = list(instruction.get_qubits())[0]
        for i in range(2 * self.num_qubits):
            self.tableau[i, -1] = self.tableau[i, -1] ^ (self.tableau[i, qubit_label] * self.tableau[i, qubit_label + self.num_qubits])
            self.tableau[i, qubit_label + self.num_qubits] = self.tableau[i, qubit_label + self.num_qubits] ^ self.tableau[i, qubit_label]

    def _apply_measurement(self, instruction):
        t_qbit = value_get(instruction.qubit)
        t_cbit = value_get(instruction.classical_reg)

        # check if the output of the measurement is random
        # this is analogous to the case when the measurement operator does not
        # commute with at least one stabilizer
        if any(self.tableau[self.num_qubits:, t_qbit] == 1):
            # find the first one.
            xpa_idx = np.where(self.tableau[self.num_qubits:, t_qbit] == 1)[0][0]  # take the first index
            xpa_idx += self.num_qubits  # adjust so we can index into the tableau
            for ii in range(2 * self.num_qubits):  # loop over each row and call rowsum(ii, xpa_idx)
                if ii != xpa_idx and self.tableau[ii, t_qbit] == 1:
                    self._rowsum(ii, xpa_idx)

            # moving the operator into the destabilizer and then replacing with
            # the measurement operator
            self.tableau[xpa_idx - self.num_qubits, :] = self.tableau[xpa_idx, :]

            # this is like replacing the non-commuting element with the measurement operator
            self.tableau[xpa_idx, :] = np.zeros((1, 2 * self.num_qubits + 1))
            self.tableau[xpa_idx, t_qbit + self.num_qubits] = 1

            # perform the measurement
            self.tableau[xpa_idx, -1] = 1 if np.random.random() > 0.5 else 0

            # set classical results to return
            self.classical_memory[t_cbit] = self.tableau[xpa_idx, -1]

        # outcome of measurement is deterministic...need to determine sign
        else:
            # augment tableaue with a scratch space
            self.tableau = np.vstack((self.tableau, np.zeros((1, 2 * self.num_qubits + 1), dtype=int)))
            for ii in range(self.num_qubits):
                # We need to check if R(i) anticommutes with Za...which it does if x_{ia} = 1
                if self.tableau[ii, t_qbit] == 1:  # refrencing the destabilizers

                    # check something silly.  Does the destabilizer anticommute with the observable?  It SHOULD!
                    tmp_vector_representing_z_qubit = np.zeros((2 * self.num_qubits), dtype=int)
                    tmp_vector_representing_z_qubit[t_qbit] = 1
                    assert symplectic_inner_product(tmp_vector_representing_z_qubit, self.tableau[ii, :-1]) == 1

                    # row sum on the stabilizers (summing up operators such that we get operator Z_{a})
                    self._rowsum(2 * self.num_qubits, ii + self.num_qubits)  # note: A-G says 2 n + 1...this is correct...but they start counting at 1 not zero

            # set the classical bit to be the last element of the scratch row
            self.classical_memory[t_cbit] = self.tableau[-1, -1]

            # remove the scratch space
            self.tableau = self.tableau[:2 * self.num_qubits, :]


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

    :param stabilizer_tableau:
    :return:
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


def binary_rref(code_matrix):
    """
    Convert the binary code_matrix into rref

    :param code_matrix:
    :return:
    """
    code_matrix = code_matrix.astype(int)
    rdim, cdim = code_matrix.shape
    for ridx in range(rdim):
        # print("start [{},{}] = {}".format(ridx, ridx,
        #                                   code_matrix[ridx, ridx]))
        # set first element
        if not np.isclose(code_matrix[ridx, ridx], 1):
            for ii in range(ridx + 1, rdim):
                if np.isclose(code_matrix[ii, ridx], 1.0):
                    # switch rows
                    # code_matrix[ridx, :], code_matrix[ii, :] = code_matrix[ii, :], code_matrix[ridx, :]
                    code_matrix[[ridx, ii]] = code_matrix[[ii, ridx]]
                    print("r_{} <-> r_{}".format(ii + 1, ridx + 1))

        # print(code_matrix)
        # start elimination of all other rows
        # print("pivot [{}, {}] = {}".format(ridx, ridx,
        #                                    code_matrix[ridx, ridx]))

        rows_to_eliminate = set(range(rdim))
        rows_to_eliminate.remove(ridx)
        # print(rows_to_eliminate)
        for elim_ridx in list(rows_to_eliminate):
            # eliminate by Hadamard product on the rows (XOR element wise)
            # if other row element is 1 then eliminate by subtracting ridx row

            # if np.isclose(code_matrix[elim_ridx, ridx], 1.0):
            #     code_matrix[elim_ridx, :] = code_matrix[elim_ridx, :] - \
            #                                 code_matrix[ridx, :]
            if np.isclose(code_matrix[elim_ridx, ridx], 1.0):
                code_matrix[elim_ridx, :] = code_matrix[elim_ridx, :] ^ code_matrix[ridx, :]
                # print("elim [{}, :]".format(elim_ridx))

        # print("semi rref at row {}".format(ridx))
        # print(code_matrix)
    return code_matrix


def symplectic_inner_product(vector1, vector2):
    """
    Operators commute if their binary form symplectic inner product is zero

    Operators anticommute if their binary form symplectic inner product is one

    :param vector1: binary form of operator with no sign info
    :param vector2: binary form of a pauli operator with no sign info
    :return: 0, 1
    """
    if vector1.shape != vector2.shape:
        raise ValueError("vectors must be the same size.")

    # TODO: add a check for binary or integer linear arrays

    hadamard_product = np.multiply(vector1, vector2)
    return reduce(lambda x, y: x ^ y, hadamard_product)


if __name__ == "__main__":
    # practice my clifford transformations
    # <Z1,Z2,...Zn> = |0000...0N>
    # N |psi> = N g |psi> = NgNd N|psi> = g2 |psi2> = |psi2>

    from referenceqvm.gates import X, Y, Z, H, CNOT, S, I
    from sympy import Matrix

    # HZH = X
    test_X = H.dot(Z).dot(H)
    assert np.allclose(test_X, X)

    # HXH = Z
    test_Z = H.dot(X).dot(H)
    assert np.allclose(test_Z, Z)

    # HYH = iHZHHXH = iXZ = -Y
    test_nY = H.dot(Y).dot(H)
    assert np.allclose(test_nY, -Y)

    # XYX = -Y
    assert np.allclose(X.dot(Y).dot(X), -Y)

    # XZX = -Z
    assert np.allclose(X.dot(Z).dot(X), -Z)

    # SXS = Y
    assert np.allclose(S.dot(X).dot(np.conj(S).T), Y)

    # CNOT(X otimes X)CNOT = X otimes I
    # (C, T) for CNOT as written in reference-qvm
    assert np.allclose(CNOT.dot(np.kron(I, X)).dot(CNOT), np.kron(I, X))
    assert np.allclose(CNOT.dot(np.kron(X, I)).dot(CNOT), np.kron(X, X))
    assert np.allclose(CNOT.dot(np.kron(X, X)).dot(CNOT), np.kron(X, I))

    # CNOT(Z otimes Z)CNOT = Z otimes I
    assert np.allclose(CNOT.dot(np.kron(Z, Z)).dot(CNOT), np.kron(I, Z))
    assert np.allclose(CNOT.dot(np.kron(Z, I)).dot(CNOT), np.kron(Z, I))
    assert np.allclose(CNOT.dot(np.kron(I, Z)).dot(CNOT), np.kron(Z, Z))

    Z1 = np.kron(Z, I)
    Z2 = np.kron(I, Z)
    ZZ = np.kron(Z, Z)
    X1 = np.kron(X, I)
    X2 = np.kron(I, X)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)

    generator_matrix = np.zeros((3, 4))
    generator_matrix[0, 2] = generator_matrix[0, 3] = 1  # ZZ
    generator_matrix[1, 0] = generator_matrix[1, 1] = 1  # XX
    generator_matrix[2, 0] = generator_matrix[2, 2] = 1  # Y
    generator_matrix[2, 1] = generator_matrix[2, 3] = 1  # Y
    parity_vector = np.zeros((3, 1))
    parity_vector[-1] = 1

    # define the amplitudes from the generator set by diagonalizing
    # the tableau matrix?
    generator_matrix = np.hstack((generator_matrix, parity_vector))
    print(generator_matrix)

    generator_matrix_2 = binary_rref(generator_matrix)
    print('\n')
    print(generator_matrix_2)

    mat = Matrix(generator_matrix)
    mat, pivots = mat.rref()
    mat = np.asarray(mat).astype(int)
    print(mat)

