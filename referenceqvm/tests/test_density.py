import numpy as np
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
from referenceqvm.api import QVMConnection
from referenceqvm.qvm_density import QVM_Density


def test_initialize():
    qvm = QVMConnection(type_trans='density')
    assert isinstance(qvm, QVM_Density)


def test_qaoa_density():
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    wf_true = np.reshape(np.array(wf_true), (4, 1))
    rho_true = np.dot(wf_true, np.conj(wf_true).T)
    prog = Program()
    prog.inst([RYgate(np.pi/2)(0), RXgate(np.pi)(0),
               RYgate(np.pi/2)(1), RXgate(np.pi)(1),
               CNOTgate(0, 1), RXgate(-np.pi/2)(1), RYgate(4.71572463191)(1),
               RXgate(np.pi/2)(1), CNOTgate(0, 1),
               RXgate(-2*2.74973750579)(0), RXgate(-2*2.74973750579)(1)])

    qvm = QVMConnection(type_trans='density')
    wf_rho = qvm.density(prog)
    assert np.isclose(rho_true, wf_rho.todense()).all()


def test_larger_qaoa_density():
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
    qvm = QVMConnection(type_trans='density')
    rho_test = qvm.density(prog)
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

    wf_true = np.reshape(wf_true, (2**4, 1))
    rho_true = np.dot(wf_true, np.conj(wf_true).T)
    assert np.isclose(rho_test.todense(), rho_true).all()
