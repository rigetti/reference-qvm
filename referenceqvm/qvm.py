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
An pure implementation of a QAM that executes nearly all of Quil, via program
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

TODO - implement random number seeding, so if you run the same program you
       always get the same result
"""
import numpy as np
import scipy.sparse as sps

from .unitary_generator import lifted_gate, tensor_gates, value_get
from .gates import gate_matrix, utility_gates

from pyquil.quil import Program
from pyquil.quilbase import *
from pyquil.wavefunction import Wavefunction


class QAM(object):
    """
    Subclass to make other QVMs.
    """
    def __init__(self, qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=None, defgate_set=None):
        """
        STATE MACHINE MODEL OF THE QVM
        """
        self.num_qubits = qubits
        self.classical_memory = classical_memory
        self.program = program
        self.program_counter = program_counter
        self.gate_set = gate_set
        self.defgate_set = defgate_set

    def load_program(self, pyquil_program):
        """
        Loads a pyQuil program into the QAM memory.

        Synthesizes pyQuil program into instruction list.
        Initializes program object and program counter.

        :param pyquil_program: (pyQuil program data object) program to be ran

        :void return:
        """
        # typecheck
        if not isinstance(pyquil_program, Program):
            raise TypeError("I can only generate from pyQuil programs")
        if len(pyquil_program.actions) == 0:
            raise TypeError("Invalid program - zero actions.")

        if isinstance(self, QVM):
            # if QVM, allow all instruction functionality
            self.all_inst = True
        elif isinstance(self, QVM_Unitary):
            # if QVM_Unitary, only allowed to have Gates and DefGates.
            self.all_inst = False

        # synthesize program into instruction list
        p = pyquil_program.synthesize()

        # create defgate dictionary
        defined_gates = {}
        for dg in pyquil_program.defined_gates:
            defined_gates[dg.name] = dg.matrix
        self.defgate_set = defined_gates

        # if QVM_Unitary, check if all instructions are valid.
        invalid = False
        for index, action in enumerate(p):
            if isinstance(action, Instr):
                if not action.operator_name in self.gate_set.keys()\
                    and not action.operator_name in self.defgate_set.keys():
                    invalid = True
                    break
            else:
                invalid = True
                break
        if invalid == True and self.all_inst == False:
            raise TypeError("In QVM_Unitary, only Gates and DefGates are "
                            "supported")

        # set internal program and counter to their appropriate values
        self.program = p
        self.program_counter = 0

        # setup quantum and classical memory
        q_max, c_max = self.identify_bits()
        if c_max < 512:
            # floor for number of cbits = 512!
            c_max = 512
        self.num_qubits = q_max
        self.classical_memory = np.zeros(c_max).astype(bool)

    def identify_bits(self):
        """
        Iterates through QAM program and finds number of qubits and cbits 
        needed to run program.
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
        q_limit = 23
        if q_max >= q_limit:
            # hardcoded qubit maximum (so you don't kill the QVM)
            raise RuntimeError("Too many qubits. Maximum qubit number "
                               "supported: {}".format(q_limit + 1))
        return (q_max + 1, c_max + 1)

    def current_instruction(self):
        """
        Returns what should be run by the QVM next.
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

    def transition(self):
        """
        Abstract class for the transition type
        """
        raise NotImplementedError("transition is an abstract class of QAM. "
                                  "Implement in subclass")

    def wavefunction(self):
        """
        Abstract class for the transition type
        """
        raise NotImplementedError("wavefunction is an abstract class of QAM. "
                                  "Implement in subclass")

    def unitary(self):
        """
        Abstract class for the transition type
        """
        raise NotImplementedError("unitary is an abstract class of QAM. "
                                  "Implement in subclass")


class QVM(QAM):
    """
    A  P Y T H O N
    Q U A N T U M
    V I R T U A L
    M A C H I N E

    Supports run(), run_and_measure(), and wavefunction() methods.

    Subclass QAM to QVM_Unitary to obtain unitaries from pyQuil program.
    """
    def __init__(self, qubits=None, program=None, program_counter=None,
                 classical_memory=None, gate_set=None, defgate_set=None):
        """
        Subclassed from QAM this is a pure QVM.
        """
        super(QVM, self).__init__(qubits=qubits, program=program,
                                  program_counter=program_counter,
                                  classical_memory=classical_memory,
                                  gate_set=gate_set, defgate_set=defgate_set)
        self.wf = None

    def measurement(self, qubit_index, psi=None):
        """
        Given the wavefunction 'psi' and 'qubit_index' to measure over, returns
        the measurement unitary, measurement outcome, and resulting wavefunction.

        Provides the measurement outcome, measurement unitary, and resultant
        wavefunction after measurement.

        :param psi: wavefunction vector to be measured (and collapsed in-place)
        :param qubit_index: is the qubit that I am measuring
        :returns: measurement_value, `unitary` for measurement, resulting wavefunct
        """
        # lift projective measurement operator to Hilbert space
        # prob(0) = <psi P0 | P0 psi> = psi* . P0* . P0 . psi
        measure_0 = lifted_gate(qubit_index, utility_gates['P0'], \
                                self.num_qubits)
        if type(psi) is type(None):
            proj_psi = measure_0.dot(self.wf)
        else:
            proj_psi = measure_0.dot(psi)
        prob_zero = np.dot(np.conj(proj_psi).T,\
                           proj_psi)[0, 0]

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:  
            # decohere state using the measure_0 operator
            unitary = measure_0.dot(sps.eye(2 ** self.num_qubits) / \
                                    np.sqrt(prob_zero))
            measured_val = 0
        else:  # measure one
            measure_1 = lifted_gate(qubit_index, utility_gates['P1'], \
                                    self.num_qubits)
            unitary = measure_1.dot(sps.eye(2 ** self.num_qubits) / \
                                    np.sqrt(1 - prob_zero))
            measured_val = 1

        return measured_val, unitary

    def find_label(self, label):
        """
        Helper function that iterates over the program and looks for a 
        JumpTarget that has a Label matching the input label.
        """
        assert isinstance(label, Label)
        for index, action in enumerate(self.program):
            if isinstance(action, JumpTarget):
                if label == action.label:
                    return index

        # if we reach this point, Label was not found in program.
        raise RuntimeError("Improper program - Jump Target not found in the "
                           "input program!")

    def _transition(self, instruction):
        """
        Implements a transition on the wf-qvm.
        Assumes entire Program() is already loaded into self.program as 
        the synthesized list of Quilbase action objects.

        :param QuilAction instruction: instruction to execute.

        Possible types of instructions:
            gate in self.gate_set or self.defgates_set
            Measurement

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
            unitary = tensor_gates(self.gate_set, self.defgate_set, \
                                   instruction, self.num_qubits)
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
                            "measurements, and gates/defgates."\
                            .format(type(instruction)))

    def transition(self, instruction):
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

        :param Program pyquil_program: A pyQuil program. :param list classical_addresses: A list of classical addresses.
        :param int trials: Number of shots to collect.
        :return: A list of lists of bits. Each sublist corresponds to the
                 values in `classical_addresses`.
        :rtype: list
        """
        results = []
        for trial in range(trials):
            _, classical_vals = self.wavefunction(pyquil_program, \
                                    classical_addresses=classical_addresses)
            results.append(classical_vals)

        return results

    def run_and_measure(self, pyquil_program, qubits=None, trials=1):
        """
        Run a pyQuil program once to determine the final wavefunction, and
        measure multiple times.
        
        :param Program pyquil_program: A pyQuil program.
        :param list qubits: A list of qubits to be measured after each trial.
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        if type(qubits) is type(None):
            qubits = []

        results = []
        for trial in range(trials):
            wf, _ = self.wavefunction(pyquil_program, \
                                    classical_addresses=None)
            trial_results = []
            wf = wf.amplitudes
            for qubit_index in qubits:
                if qubit_index < self.num_qubits:
                    measured_val, unitary = self.measurement(qubit_index,\
                                                             psi=wf)
                    wf = unitary.dot(wf)
                else:
                    measured_val = 0  # unallocated qubits are zero.
                trial_results.append(measured_val)
            results.append(trial_results)

        return results

    def wavefunction(self, pyquil_program, classical_addresses=None):
        """
        Simulate a pyQuil program and get the wavefunction back.
        
        :param Program quil_program: A pyQuil program.
        :param list classical_addresses: An optional list of classical
                 addresses.
        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits
                 corresponding to the classical addresses.
        :rtype: tuple
        """
        if type(classical_addresses) is not type(None):
            # check that no classical addresses are repeated
            assert len(set(classical_addresses)) == len(classical_addresses)
            # set classical bitmask
            mask = np.array(classical_addresses)
        else:
            mask = None

        # load program
        self.load_program(pyquil_program)

        # setup wavefunction
        self.wf = np.zeros((2 ** self.num_qubits, 1)).astype(np.complex128)
        self.wf[0, 0] = 1.0

        # evolve wf with program, via kernel
        self.kernel()

        return Wavefunction(self.wf), list(self.classical_memory[mask])


class QVM_Unitary(QAM):
    """
    A Python QVM (Quantum Virtual Machine).
    
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

    def transition(self, instruction):
        """
        Implements a transition on the unitary-qvm.
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
        :returns: a unitary corresponding to the output of the program.
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
        Calculate the expectation value given a state prepared

        :param pyquil_program: (pyquil.Program) object containing only protoQuil
                               instructions.
        :param operator_programs: (optional, list) of PauliTerms. Default is
                                  Identiy operator.
        :returns: (float) expectation value of the operators.
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
