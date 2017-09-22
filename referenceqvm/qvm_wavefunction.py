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
An pure implementation of a QVM that executes nearly all of Quil, via program
objects created in pyQuil and then submitted to the QVM.

The goal of this reference-QVM is to produce an accurate QAM according to the
specs of arXiV:1608.03355, that is both pedagogical and useful. In particular,
the QVM can run pyQuil programs operating on a reasonable number of qubits
(for most standard computers, programs requiring less than ~20 qubits should
run in less than a few seconds).

This is to allow fast prototyping and development of quantum programs.
Any programs requiring larger numbers of qubits should be sent to the Rigetti
QVM, currently running on a cloud server.

Nearly all abstractions allowed by pyQuil are valid programs that can be run,
excepting the following unsupported Quil instructions:
- DEFCIRCUIT
- WAIT
- NOP
- parametrized DefGates (also unsupported by pyQuil)

See documentation for further details, e.g. DEFCIRCUIT, WAIT not supported.
"""
# TODO - implement random number seeding for deterministic experiments
import numpy as np
import scipy.sparse as sps

from pyquil.quil import Program
from pyquil.quilbase import (Instr,
                             Measurement,
                             UnaryClassicalInstruction,
                             BinaryClassicalInstruction,
                             ClassicalTrue,
                             ClassicalFalse,
                             ClassicalOr,
                             ClassicalNot,
                             ClassicalAnd,
                             ClassicalExchange,
                             ClassicalMove,
                             Label,
                             Jump,
                             JumpTarget,
                             JumpConditional,
                             JumpWhen,
                             JumpUnless,
                             Halt)
from pyquil.wavefunction import Wavefunction
from referenceqvm.unitary_generator import lifted_gate, tensor_gates, value_get
from referenceqvm.gates import utility_gates
from referenceqvm.qam import QAM


class QVM_Wavefunction(QAM):
    """
    A  P Y T H O N

    Q U A N T U M

    V I R T U A L

    M A C H I N E

    Supports run(), run_and_measure(), and wavefunction() methods.

    Subclass QAM to QVM_Unitary to obtain unitaries from pyQuil program.
    """
    def __init__(self, num_qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=None, defgate_set=None):
        """
        Subclassed from QAM this is a pure QVM.
        """
        super(QVM_Wavefunction, self).__init__(num_qubits=num_qubits,
                                               program=program,
                                               program_counter=program_counter,
                                               classical_memory=classical_memory,
                                               gate_set=gate_set,
                                               defgate_set=defgate_set)
        self.wf = None  # no wavefunction upon init
        self.all_inst = True  # allow all instructions

    def measurement(self, qubit_index, psi=None):
        """
        Given the wavefunction 'psi' and 'qubit_index' to measure over, returns
        the measurement unitary, measurement outcome, and resulting wavefunction.

        Provides the measurement outcome, measurement unitary, and resultant
        wavefunction after measurement.

        :param int qubit_index: index of qubit that I am measuring
        :param np.array psi: wavefunction vector to be measured (and collapsed in-place)

        :return: measurement_value, `unitary` for measurement
        :rtype: tuple (int, sparse_array)
        """
        # lift projective measurement operator to Hilbert space
        # prob(0) = <psi P0 | P0 psi> = psi* . P0* . P0 . psi
        measure_0 = lifted_gate(qubit_index, utility_gates['P0'], self.num_qubits)
        if type(psi) is type(None):
            proj_psi = measure_0.dot(self.wf)
        else:
            proj_psi = measure_0.dot(psi)
        prob_zero = np.dot(np.conj(proj_psi).T, proj_psi)[0, 0]

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            # decohere state using the measure_0 operator
            unitary = measure_0.dot(sps.eye(2 ** self.num_qubits) / np.sqrt(prob_zero))
            measured_val = 0
        else:  # measure one
            measure_1 = lifted_gate(qubit_index, utility_gates['P1'], self.num_qubits)
            unitary = measure_1.dot(sps.eye(2 ** self.num_qubits) / np.sqrt(1 - prob_zero))
            measured_val = 1

        return measured_val, unitary

    def find_label(self, label):
        """
        Helper function that iterates over the program and looks for a
        JumpTarget that has a Label matching the input label.

        :param Label label: Label object to search for in program

        :return: program index where Label is found
        :rtype: int
        """
        assert isinstance(label, Label)
        for index, action in enumerate(self.program):
            if isinstance(action, JumpTarget):
                if label == action.label:
                    return index

        # Label was not found in program.
        raise RuntimeError("Improper program - Jump Target not found in the "
                           "input program!")

    def _transition(self, instruction):
        """
        Implements a transition on the wf-qvm.
        Assumes entire Program() is already loaded into self.program as
        the synthesized list of Quilbase action objects.

        Possible types of instructions:
            Measurement
            gate in self.gate_set or self.defgates_set
            Jump, JumpTarget, JumpConditional
            Unary and Binary ClassicalInstruction

        :param action-like instruction: {Measurement, Instr, Jump, JumpTarget,
            JumpTarget, JumpConditional, UnaryClassicalInstruction,
            BinaryClassicalInstruction, Halt} instruction to execute.
        """
        if isinstance(instruction, Measurement):
            # perform measurement and modify wf in-place
            t_qbit = value_get(instruction.arguments[0])
            t_cbit = value_get(instruction.arguments[1])
            measured_val, unitary = self.measurement(t_qbit, psi=None)
            self.wf = unitary.dot(self.wf)

            # load measured value into classical bit destination
            self.classical_memory[t_cbit] = measured_val
            self.program_counter += 1

        elif isinstance(instruction, Instr):
            # apply Gate or DefGate
            unitary = tensor_gates(self.gate_set, self.defgate_set, instruction, self.num_qubits)
            self.wf = unitary.dot(self.wf)
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

    def run(self, pyquil_program, classical_addresses=None, trials=1):
        """
        Run a pyQuil program multiple times, accumulating the values deposited
        in a list of classical addresses.

        :param Program pyquil_program: A pyQuil program.
        :param list classical_addresses: A list of classical addresses.
        :param int trials: Number of shots to collect.

        :return: A list of lists of bits. Each sublist corresponds to the
                 values in `classical_addresses`.
        :rtype: list
        """
        results = []
        for trial in range(trials):
            _, classical_vals = self.wavefunction(pyquil_program,
                                                  classical_addresses=classical_addresses)
            results.append(classical_vals)

        return results

    def run_and_measure(self, pyquil_program, qubits=None, trials=1):
        """
        Run a pyQuil program once to determine the final wavefunction, and
        measure multiple times.

        If unused qubits are measured, they will always return zero.

        :param Program pyquil_program: A pyQuil program.
        :param list qubits: A list of qubits to be measured after each trial.
        :param int trials: Number of shots to collect.

        :return: A list of a list of bits.
        :rtype: list
        """
        if type(qubits) is type(None):
            qubits = []
        elif type(qubits) is not list:
            raise TypeError("Must pass in qubit indices as list")

        results = []
        for trial in range(trials):
            wf, _ = self.wavefunction(pyquil_program,
                                      classical_addresses=None)
            trial_results = []
            wf = wf.amplitudes
            for qubit_index in qubits:
                if qubit_index < self.num_qubits:
                    measured_val, unitary = self.measurement(qubit_index, psi=wf)
                    wf = unitary.dot(wf)
                else:
                    measured_val = 0  # unallocated qubits are zero.
                trial_results.append(measured_val)
            results.append(trial_results)

        return results

    def wavefunction(self, pyquil_program, classical_addresses=None):
        """
        Simulate a pyQuil program and get the wavefunction back.

        :param Program pyquil_program: A pyQuil program.
        :param list classical_addresses: An optional list of classical
                 addresses.

        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits
                 corresponding to the classical addresses.
        :rtype: tuple
        """
        # load program
        self.load_program(pyquil_program)

        # check valid classical memory access
        if not isinstance(classical_addresses, type(None)):
            # check that no classical addresses are repeated
            assert len(set(classical_addresses)) == len(classical_addresses)
            # check that all classical addresses are valid
            if np.min(classical_addresses) < 0 or \
               np.max(classical_addresses) >= len(self.classical_memory):
                raise RuntimeError("Improper classical memory access outside "
                                   "allocated classical memory range.")
            # set classical bitmask
            mask = np.array(classical_addresses)
        else:
            mask = None

        # setup wavefunction
        self.wf = np.zeros((2 ** self.num_qubits, 1), dtype=np.complex128)
        self.wf[0, 0] = 1.0

        # evolve wf with program, via kernel
        self.kernel()

        # convert bools (False, True) to ints (0, 1)
        if isinstance(mask, type(None)):
            # return all
            results = [int(x) for x in self.classical_memory]
        else:
            results = [int(x) for x in self.classical_memory[mask]]

        return Wavefunction(self.wf), results
