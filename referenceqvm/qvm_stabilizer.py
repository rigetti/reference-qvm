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
Pure QVM that only executes pyQuil programs containing Clifford group generators.
This is based off of the paper by Aaronson and Gottesman Phys. Rev. A 70, 052328
"""
from functools import reduce
from pyquil.quil import Program, get_classical_addresses_from_program
from pyquil.quilbase import *
from pyquil.paulis import PauliTerm, sI, sZ, sX, sY
from pyquil.wavefunction import Wavefunction

from referenceqvm.qam import QAM
from referenceqvm.gates import stabilizer_gate_matrix
from referenceqvm.unitary_generator import value_get, tensor_up
from referenceqvm.stabilizer_utils import (project_stabilized_state,
                                           symplectic_inner_product,
                                           binary_stabilizer_to_pauli_stabilizer)


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

        :param Program pyquil_program: program to be run
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

        :param Int num_qubits: number of qubits represented by the tableau
        :return: a numpy integer array representing the tableau stabilizing
                 the |000..000> state
        :rtype: np.ndarray
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
        Run pyquil program and return the results

        Loads and checks program if all gates are within the stabilizer set.
        Then executes program.  If measurements are requested then the measured
        results are returned

        :param Program pyquil_program: a pyquil Program containing only
                                       CNOT-H-S-MEASUREMENT operations
        :param classical_addresses: classical addresses to return
        :param trials: number of times to repeat the execution of the program
        :return: list of lists of classical memory after each run
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

    def density(self, pyquil_program):
        """
        Run program and compute the density matrix

        Loads and checks program if all gates are within the stabilizer set.
        Then executes program and returns the final density matrix for the
        stabilizer

        :param Program pyquil_program: a pyquil Program containing only
                                       CNOT-H-S-MEASUREMENT operations
        :return:
        """
        self.load_program(pyquil_program)
        self.tableau = self._n_qubit_tableau(self.num_qubits)
        self.kernel()
        stabilizers = binary_stabilizer_to_pauli_stabilizer(self.stabilizer_tableau())
        pauli_ops = list(map(lambda x: 0.5 * (sI(0) + x), stabilizers))
        pauli_ops = reduce(lambda x, y: x * y, pauli_ops)
        return tensor_up(pauli_ops, self.num_qubits)

    def wavefunction(self, program):
        """
        Simulate the program and then project out the final state.

        Return the final state as a wavefunction pyquil object

        :param program: pyQuil program composed of stabilizer operations only
        :return: pyquil.Wavefunction.wavefunction object.
        """
        self.load_program(program)
        self.tableau = self._n_qubit_tableau(self.num_qubits)
        self.kernel()
        stabilizers = binary_stabilizer_to_pauli_stabilizer(self.stabilizer_tableau())
        stabilizer_state = project_stabilized_state(stabilizers)
        stabilizer_state = np.array(stabilizer_state.todense())  # todense() returns a matrixlib type
        return Wavefunction(stabilizer_state.flatten())

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
        :return: None but mutates the tableau
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

        :param x1: binary variable for the first pauli operator representing X
        :param z1: binary variable for th first pauli operator representing Z
        :param x2: binary variable for the second pauli operator representing X
        :param z2: binary variable for the second pauli operator representing Z
        :return: power that 1j is raised to when multiplying paulis together
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

        raise ValueError("We were unable to multiply the pauli operators" +
                         " together!")

    def _apply_cnot(self, instruction):
        """
        Apply a CNOT to the tableau

        :param instruction: pyquil abstract instruction.
        """
        a, b = [value_get(x) for x in instruction.qubits]
        for i in range(2 * self.num_qubits):
            self.tableau[i, -1] = self._cnot_phase_update(i, a, b)
            self.tableau[i, b] = self.tableau[i, b] ^ self.tableau[i, a]
            self.tableau[i, a + self.num_qubits] = self.tableau[i, a + self.num_qubits] ^ self.tableau[i, b + self.num_qubits]

    def _cnot_phase_update(self, i, c, t):
        """
        update r_{i} as a submethod to applying CNOT to the tableau

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

        :param instruction: pyquil abstract instruction.
        """
        qubit_label = [value_get(x) for x in instruction.qubits][0]
        for i in range(2 * self.num_qubits):
            self.tableau[i, -1] = self.tableau[i, -1] ^ (self.tableau[i, qubit_label] * self.tableau[i, qubit_label + self.num_qubits])
            self.tableau[i, [qubit_label, qubit_label + self.num_qubits]] = self.tableau[i, [qubit_label + self.num_qubits, qubit_label]]

    def _apply_phase(self, instruction):
        """
        Apply the phase gate instruction ot the tableau

        :param instruction: pyquil abstract instruction.
        """
        qubit_label = [value_get(x) for x in instruction.qubits][0]
        for i in range(2 * self.num_qubits):
            self.tableau[i, -1] = self.tableau[i, -1] ^ (self.tableau[i, qubit_label] * self.tableau[i, qubit_label + self.num_qubits])
            self.tableau[i, qubit_label + self.num_qubits] = self.tableau[i, qubit_label + self.num_qubits] ^ self.tableau[i, qubit_label]

    def _apply_measurement(self, instruction):
        """
        Apply a measurement

        :param instruction: pyquil abstract instruction.
        """
        t_qbit = value_get(instruction.qubit)
        t_cbit = value_get(instruction.classical_reg)

        # check if the output of the measurement is random
        # this is analogous to the case when the measurement operator does not
        # commute with at least one stabilizer
        if any(self.tableau[self.num_qubits:, t_qbit] == 1):
            # find the first `1'.
            xpa_idx = np.where(self.tableau[self.num_qubits:, t_qbit] == 1)[0][0]  # take the first index
            xpa_idx += self.num_qubits   # adjust so we can index into the tableau
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
            # augment tableau with a scratch space
            self.tableau = np.vstack((self.tableau, np.zeros((1, 2 * self.num_qubits + 1), dtype=int)))
            for ii in range(self.num_qubits):
                # We need to check if R(i) anticommutes with Za...which it does if x_{ia} = 1
                if self.tableau[ii, t_qbit] == 1:  # referencing the destabilizers

                    # TODO: Remove this for performance?
                    # check something silly.  Does the destabilizer anticommute with the observable?  It SHOULD!
                    tmp_vector_representing_z_qubit = np.zeros((2 * self.num_qubits), dtype=int)
                    tmp_vector_representing_z_qubit[t_qbit] = 1
                    assert symplectic_inner_product(tmp_vector_representing_z_qubit, self.tableau[ii, :-1]) == 1

                    # row sum on the stabilizers (summing up operators such that we get operator Z_{a})
                    # note: A-G says 2 n + 1...this is correct...but they start counting at 1 not zero
                    self._rowsum(2 * self.num_qubits, ii + self.num_qubits)

            # set the classical bit to be the last element of the scratch row
            self.classical_memory[t_cbit] = self.tableau[-1, -1]

            # remove the scratch space
            self.tableau = self.tableau[:2 * self.num_qubits, :]
