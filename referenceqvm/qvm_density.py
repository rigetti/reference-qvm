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
Pure QVM that only executes pyQuil programs containing Gates, and returns the
unitary resulting from the program evolution.
"""
import sys
import scipy.sparse as sps

from pyquil.quil import Program
from pyquil.quilbase import *

from referenceqvm.qam import QAM
from referenceqvm.unitary_generator import lifted_gate, tensor_gates, value_get
from referenceqvm.gates import utility_gates


def sparse_trace(sparse_matrix):
    dim = sparse_matrix.shape[0]
    diagonal = [sparse_matrix[i, i] for i in range(dim)]
    return np.sum(diagonal)


class QVM_Density(QAM):
    """
    A  P Y T H O N

    Q U A N T U M

    V I R T U A L

    M A C H I N E

    Only pyQuil programs containing pure Gates or DefGate objects are accepted.
    The QVM_Unitary kernel applies all the gates, and returns the unitary
    corresponding to the input program.

    Note: no classical control flow, measurements allowed.
    """
    def __init__(self, num_qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=None, defgate_set=None,
                 density=None):
        """
        Subclassed from QAM this is a pure QVM.
        """
        super(QVM_Density, self).__init__(num_qubits=num_qubits, program=program,
                                          program_counter=program_counter,
                                          classical_memory=classical_memory,
                                          gate_set=gate_set,
                                          defgate_set=defgate_set)
        self._density = density
        self.all_inst = True

    def measurement(self, qubit_index):
        """
        Perform measurement

        :param qubit_index: qubit to measure
        :return:
        """
        measure_0 = lifted_gate(qubit_index, utility_gates['P0'],
                                self.num_qubits)
        prob_zero = sparse_trace(measure_0.dot(self._density))
        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            # decohere state using the measure_0 operator
            unitary = measure_0.dot(
                sps.eye(2 ** self.num_qubits) / np.sqrt(prob_zero))
            measured_val = 0
        else:  # measure one
            measure_1 = lifted_gate(qubit_index, utility_gates['P1'],
                                    self.num_qubits)
            unitary = measure_1.dot(
                sps.eye(2 ** self.num_qubits) / np.sqrt(1 - prob_zero))
            measured_val = 1

        return measured_val, unitary

    def _transition(self, instruction):
        """
        Implements a transition on the density-qvm.

        :param Gate instruction: QuilAction gate to be implemented
        """
        if isinstance(instruction, Measurement):
            # perform measurement and modify wf in-place
            t_qbit = value_get(instruction.qubit)
            t_cbit = value_get(instruction.classical_reg)
            measured_val, unitary = self.measurement(t_qbit)
            self._density = unitary.dot(self._density).dot(np.conj(unitary.T))

            # load measured value into classical bit destination
            self.classical_memory[t_cbit] = measured_val
            self.program_counter += 1

        elif isinstance(instruction, Gate):
            # apply Gate or DefGate
            unitary = tensor_gates(self.gate_set, self.defgate_set, instruction, self.num_qubits)
            self._density = unitary.dot(self._density).dot(np.conj(unitary).T)
            self.program_counter += 1

        elif isinstance(instruction, Jump):
            # unconditional Jump; go directly to Label
            self.program_counter = self.find_label(instruction.target)

        elif isinstance(instruction, JumpTarget):
            # Label; pass straight over
            self.program_counter += 1

        elif isinstance(instruction, JumpConditional):
            # JumpConditional; check classical reg
            cond = self.classical_memory[value_get(instruction.condition)]
            dest_index = self.find_label(instruction.target)
            if isinstance(instruction, JumpWhen):
                jump_if_cond = True
            elif isinstance(instruction, JumpUnless):
                jump_if_cond = False
            else:
                raise TypeError("Invalid JumpConditional")

            if not (cond ^ jump_if_cond):
                # jumping: set prog counter to JumpTarget
                self.program_counter = dest_index
            else:
                # not jumping: hop over this JumpConditional
                self.program_counter += 1

        elif isinstance(instruction, UnaryClassicalInstruction):
            # UnaryClassicalInstruction; set classical reg
            target_ind = value_get(instruction.target)
            old = self.classical_memory[target_ind]
            if isinstance(instruction, ClassicalTrue):
                new = True
            elif isinstance(instruction, ClassicalFalse):
                new = False
            elif isinstance(instruction, ClassicalNot):
                new = not old
            else:
                raise TypeError("Invalid UnaryClassicalInstruction")

            self.classical_memory[target_ind] = new
            self.program_counter += 1

        elif isinstance(instruction, BinaryClassicalInstruction):
            # BinaryClassicalInstruction; compute and set classical reg
            left_ind = value_get(instruction.left)
            left_val = self.classical_memory[left_ind]
            right_ind = value_get(instruction.right)
            right_val = self.classical_memory[right_ind]
            if isinstance(instruction, ClassicalAnd):
                # compute LEFT AND RIGHT, set RIGHT to the result
                self.classical_memory[right_ind] = left_val & right_val
            elif isinstance(instruction, ClassicalOr):
                # compute LEFT OR RIGHT, set RIGHT to the result
                self.classical_memory[right_ind] = left_val | right_val
            elif isinstance(instruction, ClassicalMove):
                # move LEFT to RIGHT
                self.classical_memory[right_ind] = left_val
            elif isinstance(instruction, ClassicalExchange):
                # exchange LEFT and RIGHT
                self.classical_memory[left_ind] = right_val
                self.classical_memory[right_ind] = left_val
            else:
                raise TypeError("Invalid BinaryClassicalInstruction")

            self.program_counter += 1
        elif isinstance(instruction, Halt):
            # do nothing; set program_counter to end of program
            self.program_counter = len(self.program)
        else:
            raise TypeError("Invalid / unsupported instruction type: {}\n"
                            "Currently supported: unary/binary classical "
                            "instructions, control flow (if/while/jumps), "
                            "measurements, and gates/defgates.".format(type(instruction)))

    def _pre(self, instruction):
        """
        Pre-hook for state-machine execution

        Useful for applying error prior to measurement

        :param instruction: Instruction to apply to the state of the
                            state-machine
        :return: None
        """
        pass

    def _post(self, instruction):
        """
        Post-hook for state-machine execution

        useful for applying error after the gate

        :param instruction: Instruction to apply to the state of the
                            state-machine
        :return: None
        """
        pass

    def transition(self, instruction):
        """
        Implements a transition on the density-qvm with pre and post hooks

        :param Gate instruction: QuilAction gate to be implemented
        """
        self._pre(instruction)
        self._transition(instruction)
        self._post(instruction)

    def density(self, pyquil_program):
        """
        Return the density matrix of a pyquil program

        This method initializes a qvm with a gate_set, protoquil program (expressed
        as a pyquil program), and then executes the QVM statemachine.

        :param pyquil_program: (pyquil.Program) object containing only protoQuil
                                instructions.

        :return: a density matrix corresponding to the output of the program.
        :rtype: np.array
        """

        # load program
        self.load_program(pyquil_program)

        # setup density
        N = 2 ** self.num_qubits
        self._density = sps.csc_matrix(([1.0], ([0], [0])), shape=(N, N))
        # evolve unitary
        self.kernel()

        return self._density

    def expectation(self, pyquil_program, operator_programs=[Program()]):
        """
        Calculate the expectation value given a state prepared.

        :param pyquil_program: (pyquil.Program) object containing only protoQuil
                               instructions.
        :param operator_programs: (optional, list) of PauliTerms. Default is
                                  Identity operator.

        :return: expectation value of the operators.
        :rtype: float
        """
        raise NotImplementedError()

    def run(self, pyquil_program, classical_addresses=None, trials=1):
        """
        Run a pyQuil program multiple times, accumulating the values deposited
        in a list of classical addresses.

        This uses numpy's inverse sampling method to calculate bit string
        outcomes

        :param Program pyquil_program: A pyQuil program.
        :param list classical_addresses: A list of classical addresses.
                                         This is ignored but kept to have
                                         similar input as Forest QVM.
        :param int trials: Number of shots to collect.

        :return: A list of lists of bits. Each sublist corresponds to the
                 values in `classical_addresses`.
        :rtype: list
        """
        results = []
        for trial in range(trials):
            _ = self.density(pyquil_program)
            results.append(self.classical_memory[classical_addresses])
        return list(results)

    def run_and_measure(self, pyquil_program, trials=1):
        """
        Run and measure. converts to run

        :param pyquil_program:
        :param trials:
        :return:
        """
        density = self.density(pyquil_program)
        probabilities = [density[i, i].real for i in range(2 ** self.num_qubits)]

        results_as_integers = np.random.choice(2 ** self.num_qubits,
                                               size=trials,
                                               p=probabilities)

        results = list(map(lambda x:
                           list(map(int, np.binary_repr(x, width=self.num_qubits))),
                           results_as_integers))
        return results


