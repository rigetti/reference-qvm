"""
testing the correctness of wavefunction() and density()
"""
from pyquil.quil import Program
from pyquil.gates import H as Hgate
from pyquil.gates import CNOT as CNOTgate
from pyquil.gates import Y as Ygate
from pyquil.gates import X as Xgate
from pyquil.gates import Z as Zgate
from pyquil.gates import I as Igate
from pyquil.gates import RX as RXgate
from pyquil.gates import RY as RYgate
from pyquil.gates import RZ as RZgate
from pyquil.gates import PHASE as PHASEgate
from pyquil.gates import _make_gate
from pyquil.gates import STANDARD_GATES
from pyquil.paulis import PauliTerm, PauliSum, exponentiate
import numpy as np
import pytest
from referenceqvm.api import Connection
from referenceqvm.qvm import QVM, QVM_Unitary
from referenceqvm.gates import gate_matrix


def test_initialize():
    """
    can we initialize a qvm object
    """
    qvm = Connection(type_trans='wavefunction')
    assert isinstance(qvm, QVM)
    qvm = Connection(type_trans='unitary')
    assert isinstance(qvm, QVM_Unitary)


def test_belltest():
    """
    Generate a bell state with fake qvm and qvm and compare
    """
    prog = Program().inst([Hgate(0), CNOTgate(0, 1)])
    qvm = Connection(type_trans='wavefunction')
    bellout, _ = qvm.wavefunction(prog)
    bell = np.zeros((4, 1))
    bell[0, 0] = bell[-1, 0] = 1.0 / np.sqrt(2)
    assert np.allclose(bellout.amplitudes, bell)


def test_occupation_basis():
    prog = Program().inst([Xgate(0), Xgate(1), Igate(2), Igate(3)])
    state = np.zeros((2 ** 4, 1))
    state[3, 0] = 1.0
    qvm = Connection(type_trans='wavefunction')
    meanfield_state, _ = qvm.wavefunction(prog)
    assert np.allclose(meanfield_state.amplitudes, state)


def test_exp_circuit():
    true_wf = np.array([[ 0.54030231-0.84147098j], 
                        [ 0.00000000+0.j        ],
                        [ 0.00000000+0.j        ],
                        [ 0.00000000+0.j        ],
                        [ 0.00000000+0.j        ],
                        [ 0.00000000+0.j        ],
                        [ 0.00000000+0.j        ],
                        [ 0.00000000+0.j        ]])

    create2kill1 = PauliTerm("X", 1, -0.25)*PauliTerm("Y", 2)
    create2kill1 += PauliTerm("Y", 1, 0.25)*PauliTerm("Y", 2)
    create2kill1 += PauliTerm("Y", 1, 0.25)*PauliTerm("X", 2)
    create2kill1 += PauliTerm("X", 1, 0.25)*PauliTerm("X", 2)
    create2kill1 += PauliTerm("I", 0, 1.0)
    prog = Program()
    for term in create2kill1.terms:
        single_exp_prog = exponentiate(term)
        prog += single_exp_prog

    qvm = Connection(type_trans='wavefunction')
    wf, _ = qvm.wavefunction(prog)
    wf = np.reshape(wf.amplitudes, (-1, 1))
    assert np.allclose(wf, true_wf)


def test_qaoa_circuit():
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    wf_true = np.reshape(np.array(wf_true), (4, 1))
    prog = Program()
    prog.inst([RYgate(np.pi/2)(0), RXgate(np.pi)(0),
               RYgate(np.pi/2)(1), RXgate(np.pi)(1),
               CNOTgate(0, 1), RXgate(-np.pi/2)(1), RYgate(4.71572463191)(1),
               RXgate(np.pi/2)(1), CNOTgate(0, 1),
               RXgate(-2*2.74973750579)(0), RXgate(-2*2.74973750579)(1)])
    qvm = Connection(type_trans='wavefunction')
    wf_test, _ = qvm.wavefunction(prog)
    assert np.allclose(wf_test.amplitudes, wf_true)


def test_larger_qaoa_circuit():
    square_qaoa_circuit = [Hgate(0), Hgate(1), Hgate(2), Hgate(3),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           CNOTgate(0, 1),
                           RZgate(0.78564882604980579)(1),
                           CNOTgate(0, 1),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           CNOTgate(0, 3),
                           RZgate(0.78564882604980579)(3),
                           CNOTgate(0, 3),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           CNOTgate(1, 2),
                           RZgate(0.78564882604980579)(2),
                           CNOTgate(1, 2),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029)(0),
                           CNOTgate(2, 3),
                           RZgate(0.78564882604980579)(3),
                           CNOTgate(2, 3),
                           Hgate(0),
                           RZgate(-0.77868204192240842)(0),
                           Hgate(0),
                           Hgate(1),
                           RZgate(-0.77868204192240842)(1),
                           Hgate(1),
                           Hgate(2),
                           RZgate(-0.77868204192240842)(2),
                           Hgate(2),
                           Hgate(3),
                           RZgate(-0.77868204192240842)(3),
                           Hgate(3)]

    prog = Program(square_qaoa_circuit)
    qvm = Connection(type_trans='wavefunction')
    wf_test, _ = qvm.wavefunction(prog)

    wf_true = np.array([8.43771693e-05-0.1233845*1j, -1.24927731e-01+0.00329533*1j,
                        -1.24927731e-01+0.00329533*1j,
                        -2.50040954e-01+0.12661547*1j,
                        -1.24927731e-01+0.00329533*1j,  -4.99915497e-01-0.12363516*1j,
                        -2.50040954e-01+0.12661547*1j,  -1.24927731e-01+0.00329533*1j,
                        -1.24927731e-01+0.00329533*1j,  -2.50040954e-01+0.12661547*1j,
                        -4.99915497e-01-0.12363516*1j,  -1.24927731e-01+0.00329533*1j,
                        -2.50040954e-01+0.12661547*1j,  -1.24927731e-01+0.00329533*1j,
                        -1.24927731e-01+0.00329533*1j,
                        8.43771693e-05-0.1233845*1j])

    wf_true = np.reshape(wf_true, (2 ** 4, 1))
    assert np.allclose(wf_test.amplitudes, wf_true)


def test_qaoa_unitary():
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    wf_true = np.reshape(np.array(wf_true), (4, 1))
    prog = Program()
    prog.inst([RYgate(np.pi/2)(0), RXgate(np.pi)(0),
               RYgate(np.pi/2)(1), RXgate(np.pi)(1),
               CNOTgate(0, 1), RXgate(-np.pi/2)(1), RYgate(4.71572463191)(1),
               RXgate(np.pi/2)(1), CNOTgate(0, 1),
               RXgate(-2*2.74973750579)(0), RXgate(-2*2.74973750579)(1)])

    qvm = Connection(type_trans='unitary')
    test_unitary = qvm.unitary(prog)
    wf_test = np.zeros((4, 1))
    wf_test[0, 0] = 1.0
    wf_test = test_unitary.dot(wf_test)
    assert np.allclose(wf_test, wf_true)


def test_unitary_errors():
    # do we properly throw errors when non-gates are thrown into a unitary?

    # try measuring
    prog = Program()
    prog.inst([Hgate(0), Hgate(1)])
    prog.measure(0, [0])
    qvm = Connection(type_trans='unitary')
    with pytest.raises(TypeError):
        qvm.unitary(prog)

    # try an undefined DefGate
    prog = Program()
    prog.defgate("hello", np.array([[0, 1], [1, 0]]))
    prog.inst(("hello2", 0))
    with pytest.raises(TypeError):
        qvm.unitary(prog)


if __name__ == "__main__":
    test_initialize()
    test_belltest()
    test_occupation_basis()
    test_exp_circuit()
    test_qaoa_circuit()
    test_larger_qaoa_circuit()
    test_qaoa_unitary()
    test_unitary_errors()
    print("Completed!")
