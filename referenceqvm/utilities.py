"""
Utilities for transforming programs
"""
from pyquil.quil import Program
from pyquil.quilbase import RawInstr, Instr


def clense_program(pyquil_program):
    """
    Removes Raw Instructions from  pyquil program and puts them into the Gate format

    :param pyquil_program: (pyquil.Program) object with a set of instructions
    """
    output_program = Program()
    for instr_index, gate in pyquil_program.actions:
        if isinstance(gate, RawInstr):
            # transform into gate object
            gate_object = parse_raw_instruction(gate)
            output_program.actions.append((instr_index, gate_object))
        else:
            output_program.actions.append((instr_index, gate))

    return output_program


def parse_raw_instruction(raw):
    """
    Parse a pyquil raw instruction

    :return: Gate Instruction
    """
    raw_split = raw.instr.split()  # split on spaces
    operator_name = raw_split[0]
    arguments = raw_split[1:]
