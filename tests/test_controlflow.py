import pytest
import numpy as np
import api
from pyquil.quil import Program
from pyquil.gates import *


def test_if_then():
    qvm = api.Connection()

    main = Program().inst(X(0))
    branch_a = Program().inst(X(0))
    branch_b = Program().inst()
    creg = 0
    main.if_then(creg, branch_a, branch_b)

    # if TRUE creg, then measure 0 should give 0
    prep = Program().inst(TRUE(0))
    prog = prep + main
    assert qvm.run_and_measure(prog, [0])[0][0] == 0
    # if FALSE creg, then measure 0 should give 1
    prep = Program().inst(FALSE(0))
    prog = prep + main
    assert qvm.run_and_measure(prog, [0])[0][0] == 1


def test_while():
    qvm = api.Connection()

    # Name our classical registers:
    classical_flag_register = 2

    # Write out the loop initialization and body programs:
    init_register = Program(TRUE([classical_flag_register]))
    loop_body = Program(X(0), H(0)).measure(0, classical_flag_register)

    # Put it all together in a loop program:
    loop_prog = init_register.while_do(classical_flag_register, loop_body)

    _, cregs = qvm.wavefunction(loop_prog, [2])
    assert cregs[0] == False


def test_halt():
    qvm = api.Connection()

    prog = Program(X(0))
    prog.inst(HALT)
    prog.inst(X(0))
    cregs = qvm.run_and_measure(prog, [0])
    # HALT should stop execution; measure should give 1
    assert cregs[0][0] == 1

    prog = Program(X(0)).inst(X(0))
    cregs = qvm.run_and_measure(prog, [0])
    assert cregs[0][0] == 0


def test_errors():
    qvm = api.Connection()

    # NOP unsupported
    prog = Program(NOP)
    with pytest.raises(TypeError):
        qvm.wavefunction(prog)
    with pytest.raises(TypeError):
        qvm.run(prog)
    with pytest.raises(TypeError):
        qvm.run_and_measure(prog)

    # WAIT unsupported
    prog = Program(WAIT)
    with pytest.raises(TypeError):
        qvm.wavefunction(prog)
    with pytest.raises(TypeError):
        qvm.run(prog)
    with pytest.raises(TypeError):
        qvm.run_and_measure(prog)


if __name__ == "__main__":
    test_if_then()
    test_while()
    test_halt()
    test_errors()
