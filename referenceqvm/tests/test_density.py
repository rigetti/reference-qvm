import numpy as np
import scipy.sparse as sps
from math import pi
from pyquil.quil import Program
from pyquil.gates import H as Hgate
from pyquil.gates import CNOT as CNOTgate
from pyquil.gates import CZ as CZgate
from pyquil.gates import X as Xgate
from pyquil.gates import RX as RXgate
from pyquil.gates import RY as RYgate
from pyquil.gates import RZ as RZgate
from pyquil.gates import I as Igate
from pyquil.gates import PHASE as PHASEgate
from referenceqvm.api import QVMConnection
from referenceqvm.qvm_density import QVM_Density, NoiseModel, INFINITY, lifted_gate
from referenceqvm.gates import noise_gates, X, Y, Z, I, P0, P1, RY


def random_1q_density():
    state = np.random.random(2) + 1j*np.random.random()
    normalization = np.conj(state).T.dot(state)
    state /= np.sqrt(normalization)
    state = state.reshape((-1, 1))

    rho = state.dot(np.conj(state).T)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, np.conj(rho).T)
    return rho


def test_initialize():
    qvm = QVMConnection(type_trans='density')
    assert isinstance(qvm, QVM_Density)


def test_qaoa_density():
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    wf_true = np.reshape(np.array(wf_true), (4, 1))
    rho_true = np.dot(wf_true, np.conj(wf_true).T)
    prog = Program()
    prog.inst([RYgate(np.pi/2, 0), RXgate(np.pi, 0),
               RYgate(np.pi/2, 1), RXgate(np.pi, 1),
               CNOTgate(0, 1), RXgate(-np.pi/2, 1), RYgate(4.71572463191, 1),
               RXgate(np.pi/2, 1), CNOTgate(0, 1),
               RXgate(-2*2.74973750579, 0), RXgate(-2*2.74973750579, 1)])

    qvm = QVMConnection(type_trans='density')
    wf_rho = qvm.density(prog)
    assert np.isclose(rho_true, wf_rho.todense()).all()


def test_larger_qaoa_density():
    square_qaoa_circuit = [Hgate(0), Hgate(1), Hgate(2), Hgate(3),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           CNOTgate(0, 1),
                           RZgate(0.78564882604980579, 1),
                           CNOTgate(0, 1),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           CNOTgate(0, 3),
                           RZgate(0.78564882604980579, 3),
                           CNOTgate(0, 3),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           CNOTgate(1, 2),
                           RZgate(0.78564882604980579, 2),
                           CNOTgate(1, 2),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           Xgate(0),
                           PHASEgate(0.3928244130249029, 0),
                           CNOTgate(2, 3),
                           RZgate(0.78564882604980579, 3),
                           CNOTgate(2, 3),
                           Hgate(0),
                           RZgate(-0.77868204192240842, 0),
                           Hgate(0),
                           Hgate(1),
                           RZgate(-0.77868204192240842, 1),
                           Hgate(1),
                           Hgate(2),
                           RZgate(-0.77868204192240842, 2),
                           Hgate(2),
                           Hgate(3),
                           RZgate(-0.77868204192240842, 3),
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


def get_compiled_prog(theta):
    return Program([
        RZgate(-pi/2, 0),
        RXgate(-pi/2, 0),
        RZgate(-pi/2, 1),
        RXgate( pi/2, 1),
        CZgate(1, 0),
        RZgate(-pi/2, 1),
        RXgate(-pi/2, 1),
        RZgate(theta, 1),
        RXgate( pi/2, 1),
        CZgate(1, 0),
        RXgate( pi/2, 0),
        RZgate( pi/2, 0),
        RZgate(-pi/2, 1),
        RXgate( pi/2, 1),
        RZgate(-pi/2, 1),
    ])


def test_kraus_t1_normalization():
    kraus_ops = noise_gates['relaxation'](0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_t2_normalization():
    kraus_ops = noise_gates['dephasing'](0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_depolarizing_normalization():
    kraus_ops = noise_gates['depolarizing'](0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_bitflip_normalization():
    kraus_ops = noise_gates['bit_flip'](0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_phaseflip_normalization():
    kraus_ops = noise_gates['phase_flip'](0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_bitphaseflip_normalization():
    kraus_ops = noise_gates['bitphase_flip'](0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_application_bitflip():
    qvm = QVMConnection(type_trans='density')
    initial_density = random_1q_density()
    qvm._density = sps.csc_matrix(initial_density)
    qvm.num_qubits = 1
    p = 0.372
    kraus_ops = noise_gates['bit_flip'](p)
    final_density = (1 - p) * initial_density + p * X.dot(initial_density).dot(X)
    test_density = qvm.apply_kraus(kraus_ops, 0)
    assert np.allclose(test_density.todense(), final_density)


def test_kraus_application_phaseflip():
    qvm = QVMConnection(type_trans='density')
    initial_density = random_1q_density()
    qvm._density = sps.csc_matrix(initial_density)
    qvm.num_qubits = 1
    p = 0.372
    kraus_ops = noise_gates['phase_flip'](p)
    final_density = (1 - p) * initial_density + p * Z.dot(initial_density).dot(Z)
    test_density = qvm.apply_kraus(kraus_ops, 0)
    assert np.allclose(test_density.todense(), final_density)


def test_kraus_application_bitphaseflip():
    qvm = QVMConnection(type_trans='density')
    initial_density = random_1q_density()
    qvm._density = sps.csc_matrix(initial_density)
    qvm.num_qubits = 1
    p = 0.372
    kraus_ops = noise_gates['bitphase_flip'](p)
    final_density = (1 - p) * initial_density + p * Y.dot(initial_density).dot(Y)
    test_density = qvm.apply_kraus(kraus_ops, 0)
    assert np.allclose(test_density.todense(), final_density)


def test_kraus_application_relaxation():
    qvm = QVMConnection(type_trans='density')
    rho = random_1q_density()
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    p = 0.372
    kraus_ops = noise_gates['relaxation'](p)
    final_density = np.array([[rho[0, 0] + rho[1, 1] * p, np.sqrt(1 - p) * rho[0, 1]],
                              [np.sqrt(1 - p) * rho[1, 0], (1 - p) * rho[1, 1]]])
    test_density = qvm.apply_kraus(kraus_ops, 0)
    assert np.allclose(test_density.todense(), final_density)


def test_kraus_application_dephasing():
    qvm = QVMConnection(type_trans='density')
    rho = random_1q_density()
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    p = 0.372
    kraus_ops = noise_gates['dephasing'](p)
    final_density = np.array([[rho[0, 0], (1 - p) * rho[0, 1]],
                              [(1 - p) * rho[1, 0], rho[1, 1]]])
    test_density = qvm.apply_kraus(kraus_ops, 0)
    assert np.allclose(test_density.todense(), final_density)


def test_kraus_application_depolarizing():
    qvm = QVMConnection(type_trans='density')
    rho = random_1q_density()
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    p = 0.372
    kraus_ops = noise_gates['depolarizing'](p)
    final_density = (1 - p) * rho + (p / 3) * (X.dot(rho).dot(X) +
                                               Y.dot(rho).dot(Y) +
                                               Z.dot(rho).dot(Z))
    test_density = qvm.apply_kraus(kraus_ops, 0)
    assert np.allclose(test_density.todense(), final_density)


def test_kraus_compound_T1T2_application():
    qvm = QVMConnection(type_trans='density')
    rho = random_1q_density()
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    p1 = 0.372
    p2 = 0.45
    kraus_ops_1 = noise_gates['relaxation'](p1)
    kraus_ops_2 = noise_gates['dephasing'](p2)
    final_density = np.array([[rho[0, 0] + rho[1, 1] * p1, (1 - p2 ) * np.sqrt(1 - p1) * rho[0, 1]],
                              [(1 - p2) * np.sqrt(1 - p1) * rho[1, 0], (1 - p1) * rho[1, 1]]])
    qvm._density = qvm.apply_kraus(kraus_ops_1, 0)
    qvm._density = qvm.apply_kraus(kraus_ops_2, 0)
    assert np.allclose(qvm._density.todense(), final_density)


def test_kraus_through_qvm_t1():
    noise = NoiseModel(T2=INFINITY, ro_fidelity=1.0)
    qvm = QVMConnection(type_trans='density', noise_model=noise)
    rho = random_1q_density()
    identity_program = Program().inst(Igate(0))
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    qvm.load_program(identity_program)
    qvm.kernel()
    p = 1 - np.exp(-noise.gate_time_1q/noise.T1)
    final_density = np.array([[rho[0, 0] + rho[1, 1] * p, np.sqrt(1 - p) * rho[0, 1]],
                              [np.sqrt(1 - p) * rho[1, 0], (1 - p) * rho[1, 1]]])
    assert np.allclose(final_density, qvm._density.todense())


def test_kraus_through_qvm_t2():
    noise = NoiseModel(T1=INFINITY, ro_fidelity=1.0)
    qvm = QVMConnection(type_trans='density', noise_model=noise)
    rho = random_1q_density()
    identity_program = Program().inst(Igate(0))
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    qvm.load_program(identity_program)
    qvm.kernel()
    p = 1 - np.exp(-noise.gate_time_1q/noise.T2)
    final_density = np.array([[rho[0, 0], (1 - p) * rho[0, 1]],
                              [(1 - p) * rho[1, 0],  rho[1, 1]]])
    assert np.allclose(final_density, qvm._density.todense())


def test_kraus_through_qvm_t1t2():
    noise = NoiseModel(ro_fidelity=1.0)
    qvm = QVMConnection(type_trans='density', noise_model=noise)
    rho = random_1q_density()
    identity_program = Program().inst(Igate(0))
    qvm._density = sps.csc_matrix(rho)
    qvm.num_qubits = 1
    qvm.load_program(identity_program)
    qvm.kernel()
    p = 1 - np.exp(-noise.gate_time_1q/noise.T2)
    p1 = 1 - np.exp(-noise.gate_time_1q/noise.T1)
    final_density = np.array([[rho[0, 0] + p1 * rho[1, 1], np.sqrt(1 - p1) * (1 - p) * rho[0, 1]],
                              [np.sqrt(1 - p1) * (1 - p) * rho[1, 0],  (1 - p1) * rho[1, 1]]])
    assert np.allclose(final_density, qvm._density.todense())


def multi_qubit_decay_bellstate():
    """
    Test multiqubit decay
    """
    program = Program().inst([RYgate(np.pi/3)(0), CNOTgate(0, 1)])
    noise = NoiseModel(ro_fidelity=1.0)
    qvm = QVMConnection(type_trans='density', noise_model=noise)

    initial_density = np.zeros((4, 4), dtype=complex)
    initial_density[0, 0] = 1.0
    cnot_01 = np.kron(I, P0) + np.kron(X, P1)

    p1 = 1 - np.exp(-noise.gate_time_1q/noise.T1)
    p2 = 1 - np.exp(-noise.gate_time_1q/noise.T2)

    kraus_ops_1 = noise_gates['relaxation'](p1)
    kraus_ops_2 = noise_gates['dephasing'](p2)

    gate_1 = np.kron(np.eye(2), RY(np.pi/3))

    state = gate_1.dot(initial_density).dot(np.conj(gate_1).T)
    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in kraus_ops_1:
            operator = lifted_gate(ii, kop, 2).todense()
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in kraus_ops_2:
            operator = lifted_gate(ii, kop, 2).todense()
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    state = cnot_01.dot(state).dot(cnot_01.T)
    p1 = 1 - np.exp(-noise.gate_time_2q/noise.T1)
    p2 = 1 - np.exp(-noise.gate_time_2q/noise.T2)
    kraus_ops_1 = noise_gates['relaxation'](p1)
    kraus_ops_2 = noise_gates['dephasing'](p2)

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in kraus_ops_1:
            operator = lifted_gate(ii, kop, 2).todense()
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in kraus_ops_2:
            operator = lifted_gate(ii, kop, 2).todense()
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    density = qvm.density(program)
    assert np.allclose(density.todense(), state)
