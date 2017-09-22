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
QAM superclass. Implements state machine model, program loading and processing,
kernel - leaving details of evolution up to subclasses.
"""
import numpy as np

from referenceqvm.unitary_generator import value_get

from pyquil.quil import Program
from pyquil.quilbase import (Instr,
                             Measurement,
                             UnaryClassicalInstruction,
                             BinaryClassicalInstruction)


class QAM(object):
    """
    STATE MACHINE MODEL OF THE QVM:
        classical memory (classical bits, True or False)
        quantum memory (qubits)
        program array (list of pyQuil actions)
        program counter - index into array, pointing to next instruction
        valid gate set & defined gate set

    :param int num_qubits: number of qubits
    :param list program: synthesized pyQuil program list
    :param int program_counter: index into program list
    :param array-like classical_memory: {list, np.array} list of classical bits
    :param dict gate_set: dictionary of (gate_name, array) pairs
    :param dict defgate_set: dictionary of (defgate_name, array) pairs
    """
    def __init__(self, num_qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=None, defgate_set=None):
        self.num_qubits = num_qubits
        self.classical_memory = classical_memory
        self.program = program
        self.program_counter = program_counter
        self.gate_set = gate_set
        self.defgate_set = defgate_set

        self.all_inst = None

    def load_program(self, pyquil_program):
        """
        Loads a pyQuil program into the QAM memory.
            Synthesizes pyQuil program into instruction list.
            Initializes program object and program counter.

        :param Program pyquil_program: program to be ran
        """
        # typecheck
        if not isinstance(pyquil_program, Program):
            raise TypeError("I can only generate from pyQuil programs")

        if self.all_inst is None:
            raise NotImplementedError("QAM needs to be subclassed in order to "
                                      "load program")

        # synthesize program into instruction list
        synthesized_prog = pyquil_program.synthesize()

        # create defgate dictionary
        defined_gates = {}
        for dg in pyquil_program.defined_gates:
            defined_gates[dg.name] = dg.matrix
        self.defgate_set = defined_gates

        # if QVM_Unitary, check if all instructions are valid.
        invalid = False
        for index, action in enumerate(synthesized_prog):
            if isinstance(action, Instr):
                if (action.operator_name not in self.gate_set.keys()
                   and action.operator_name not in self.defgate_set.keys()):
                    invalid = True
                    break
            else:
                invalid = True
                break

        if invalid is True and self.all_inst is False:
            raise TypeError("In QVM_Unitary, only Gates and DefGates are "
                            "supported")

        # set internal program and counter to their appropriate values
        self.program = synthesized_prog
        self.program_counter = 0

        # setup quantum and classical memory
        q_max, c_max = self.identify_bits()
        if c_max <= 512:  # allocate at least 512 cbits (as floor)
            c_max = 512
        self.num_qubits = q_max
        self.classical_memory = np.zeros(c_max).astype(bool)

    def identify_bits(self):
        """
        Iterates through QAM program and finds number of qubits and cbits
        needed to run program.

        :return: number of qubits, number of classical bits used by
                 program
        :rtype: tuple
        """
        q_max, c_max = (-1, -1)
        for index, inst in enumerate(self.program):
            if isinstance(inst, Measurement):
                # instruction is measurement, acts on qbits and cbits
                if value_get(inst.arguments[0]) > q_max:
                    q_max = value_get(inst.arguments[0])
                elif value_get(inst.arguments[1]) > c_max:
                    c_max = value_get(inst.arguments[1])
            elif isinstance(inst, UnaryClassicalInstruction):
                # instruction acts on cbits
                if value_get(inst.target) > c_max:
                    c_max = value_get(inst.target)
            elif isinstance(inst, BinaryClassicalInstruction):
                # instruction acts on cbits
                if value_get(inst.left) > c_max:
                    c_max = value_get(inst.left)
                elif value_get(inst.right) > c_max:
                    c_max = value_get(inst.right)
            elif isinstance(inst, Instr):
                # instruction is Gate or DefGate, acts on qbits
                if max(map(lambda x: value_get(x), inst.arguments)) > q_max:
                    q_max = max(map(lambda x: value_get(x), inst.arguments))
        q_max += 1  # 0-indexed
        c_max += 1  # 0-indexed
        q_limit = 51
        if q_max > q_limit:
            # hardcoded qubit maximum
            raise RuntimeError("Too many qubits. Maximum qubit number "
                               "supported: {}".format(q_limit))
        return (q_max, c_max)

    def current_instruction(self):
        """
        Returns what should be run by the QVM next.

        :return: next instruction
        :rtype: pyQuil Action
        """
        return self.program[self.program_counter]

    def kernel(self):
        """
        Run the QVM!

        While program_counter is less than the program length, evaluate the
        current instruction pointed to by program_counter.

        transition(instruction) increments program_counter and returns
        whether the next instruction is HALT. If so, then break.
        """
        while self.program_counter < len(self.program):
            halted = self.transition(self.current_instruction())
            if halted:
                break

    def transition(self, instruction):
        """
        Abstract class for the transition type
        """
        raise NotImplementedError("transition is an abstract class of QAM. "
                                  "Implement in subclass")

    def wavefunction(self, pyquil_program, classical_addresses=None):
        """
        Abstract class for the transition type
        """
        raise NotImplementedError("wavefunction is an abstract class of QAM. "
                                  "Implement in subclass")

    def unitary(self, pyquil_program):
        """
        Abstract class for the transition type
        """
        raise NotImplementedError("unitary is an abstract class of QAM. "
                                  "Implement in subclass")
