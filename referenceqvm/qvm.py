"""
An pure implementation of a QAM that can only exectute protoQuil
"""
from referenceqvm.unitary_generator import tensor_gates
from referenceqvm.gates import gate_matrix
from referenceqvm.utilities import clense_program
import numpy as np


class QVM(object):
    """
    A  P Y T H O N

    Q U A N T U M
    V I R T U A L
    M A C H I N E
    """
    def __init__(self, qubits=None, program=None, program_counter=None,
                 wf=None, rho=None, classical_memory=None, gate_set=None,
                 unitary=None):
        """
        STATE MACHINE MODEL OF THE QVM
        """
        self.num_qubits = qubits
        self.wf = wf
        self.rho = rho
        self.unitary = unitary
        self.classical_memory = classical_memory
        self.program = map(lambda x: x[1], program.actions)  # get rid of instruction index
        self.program_counter = program_counter
        self.gate_set = gate_set

    def current_instruction(self):
        """
        returns what should be run by the QVM next
        """
        return self.program[self.program_counter]


def run_wf(qvm):
    """
    Run the QVM in wf mode
    """
    while qvm.program_counter < len(qvm.program):
        wf_transition(qvm, qvm.current_instruction())
    return qvm


def run_rho(qvm):
    """
    Run the QVM in rho mode
    """
    while qvm.program_counter < len(qvm.program):
        rho_transition(qvm, qvm.current_instruction())
    return qvm


def run_unitary(qvm):
    """
    Run the QVM in generate unitary mode
    """
    while qvm.program_counter < len(qvm.program):
        unitary_transition(qvm, qvm.current_instruction())
    return qvm


def wf_transition(qvm, instruction):
    """
    Transition the vqam with the instruction

    instruction is a string instruction.
    """
    if instruction.operator_name in qvm.gate_set:
        # get the unitary and evolve the state
        unitary = tensor_gates(instruction, qvm.num_qubits)
        qvm.wf = np.dot(unitary, qvm.wf)
        qvm.program_counter += 1


def rho_transition(qvm, instruction):
    """
    Transition the qvm with the instruction

    instruction is a string instruction.
    """
    if instruction.operator_name in qvm.gate_set:
        # get the unitary and evolve the state
        unitary = tensor_gates(instruction, qvm.num_qubits)
        qvm.rho = np.dot(unitary, np.dot(qvm.rho, np.conj(unitary).T))
        qvm.program_counter += 1


def unitary_transition(qvm, instruction):
    """
    Transition the qvm with the instruction

    instruction is a string instruction.
    """
    if instruction.operator_name in qvm.gate_set:
        # get the unitary and evolve the state
        unitary = tensor_gates(instruction, qvm.num_qubits)
        qvm.unitary = np.dot(unitary, qvm.unitary)
        qvm.program_counter += 1


def identify_qubits(program):
    """
    Find qubits used in the program
    """
    # find highest qubit index
    max_index = 0
    for instruction_index, gate in program.actions:
        if max(map(lambda x: x._index, gate.arguments)) > max_index:
            max_index = max(map(lambda x: x._index, gate.arguments))
    return max_index + 1


def wavefunction(pyquil_program):
    """
    Return the wavefunction after running a pyquil Program.

    This method initializes a qvm with a gate_set, protoquil program (expressed
    as a pyquil program), and then executes the QVM statemachine.

    :params pyquil_program: (pyquil.Program) object containing only protoQuil
                            instructions.
    :returns: a wavefunction corresponding to the output of the program.
    """
    max_index = identify_qubits(pyquil_program)
    wf = np.zeros((2**max_index, 1))
    wf[0, 0] = 1.0
    qvm = QVM(qubits=max_index, program=pyquil_program, program_counter=0,
              gate_set=gate_matrix.keys(), wf=wf)
    qvm = run_wf(qvm)
    return qvm.wf


def density(pyquil_program):
    """
    Return the density matrix after running a pyquil Program.

    This method initializes a qvm with a gate_set, protoquil program (expressed
    as a pyquil program), and then executes the QVM statemachine.

    :params pyquil_program: (pyquil.Program) object containing only protoQuil
                            instructions.
    :returns: a denisty matrix corresponding to the output of the program.
    """
    max_index = identify_qubits(pyquil_program)
    rho = np.zeros((2**max_index, 1))
    rho[0, 0] = 1.0
    rho = np.dot(rho, rho.T)
    qvm = QVM(qubits=max_index, program=pyquil_program, program_counter=0,
              gate_set=gate_matrix.keys(), rho=rho)
    qvm = run_rho(qvm)
    return qvm.rho


def unitary(pyquil_program):
    """
    Return the unitary of a pyquil program

    This method initializes a qvm with a gate_set, protoquil program (expressed
    as a pyquil program), and then executes the QVM statemachine.

    :params pyquil_program: (pyquil.Program) object containing only protoQuil
                            instructions.
    :returns: a unitary corresponding to the output of the program.
    """
    pyquil_program = clense_program(pyquil_program)
    max_index = identify_qubits(pyquil_program)
    unitary = np.eye(2**max_index)
    qvm = QVM(qubits=max_index, program=pyquil_program, program_counter=0,
              gate_set=gate_matrix.keys(), unitary=unitary)
    qvm = run_unitary(qvm)
    return qvm.unitary
