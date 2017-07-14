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
import numpy as np
import scipy.sparse as sps

from pyquil.quil import Program
from pyquil.quilbase import *
from pyquil.wavefunction import Wavefunction

from .unitary_generator import lifted_gate, tensor_gates, value_get
from .gates import gate_matrix, utility_gates
from .qam import QAM


class QVM_Unitary(QAM):
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
    def __init__(self, qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=None, defgate_set=None,
                 unitary=None):
        """
        Subclassed from QAM this is a pure QVM.
        """
        super(QVM_Unitary, self).__init__(qubits=qubits, program=program,
                                          program_counter=program_counter,
                                          classical_memory=classical_memory,
                                          gate_set=gate_set,
                                          defgate_set=defgate_set)
        self.umat = unitary
        self.all_inst = False

    def transition(self, instruction):
        """
        Implements a transition on the unitary-qvm.

        :param instruction: QuilAction gate to be implemented
        """
        if instruction.operator_name in self.gate_set or \
            instruction.operator_name in self.defgate_set:
            # get the unitary and evolve the state
            unitary = tensor_gates(self.gate_set, self.defgate_set, \
                                   instruction, self.num_qubits)
            self.umat = unitary.dot(self.umat)
            self.program_counter += 1
        else:
            raise TypeError("Gate {} is not in the "
                            "gate set".format(instruction.operator_name))

    def unitary(self, pyquil_program):
        """
        Return the unitary of a pyquil program

        This method initializes a qvm with a gate_set, protoquil program (expressed
        as a pyquil program), and then executes the QVM statemachine.

        :param pyquil_program: (pyquil.Program) object containing only protoQuil
                                instructions.

        :return: a unitary corresponding to the output of the program.
        :rtype: np.array
        """

        # load program
        self.load_program(pyquil_program)
        
        # setup unitary
        self.umat = np.eye(2 ** self.num_qubits)

        # evolve unitary
        self.kernel()

        return self.umat

    def expectation(self, pyquil_program, operator_programs=[Program()]):
        """
        Calculate the expectation value given a state prepared.

        Currently unimplemented.

        :param pyquil_program: (pyquil.Program) object containing only protoQuil
                               instructions.
        :param operator_programs: (optional, list) of PauliTerms. Default is
                                  Identity operator.

        :return: expectation value of the operators.
        :rtype: float
        """
        # TODO
        pass
        # num_qubits, num_cbits = self.identify_bits(pyquil_program)
        # self.num_qubits = num_qubits
        # self.wf = np.zeros((2**num_qubits, 1))
        # self.wf[0, 0] = 1.0
        # self.program_counter = 0
        # self.elapsed_time = 0
        # self.program = program_gen(pyquil_program)
        # self.kernel()
        # rho = self.wf.dot(np.conj(self.wf).T)
        # qvm_unitary = QVM_Unitary(gate_set=gate_matrix.keys())
        # hamiltonian_operators = map(lambda x: np.trace(
        #                             qvm_unitary.unitary(x, max_index=max_index).dot(rho)),
        #                             operator_programs)

        # return hamiltonian_operators