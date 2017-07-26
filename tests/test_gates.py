from referenceqvm.gates import gate_matrix
import numpy as np

def test_gate_matrix():
    I = gate_matrix['I']
    assert np.isclose(I, np.eye(2)).all()
    X = gate_matrix['X']
    assert np.isclose(X, np.array([[0, 1], [1, 0]])).all()
    Y = gate_matrix['Y']
    assert np.isclose(Y, np.array([[0, -1j], [1j, 0]])).all()
    Z = gate_matrix['Z']
    assert np.isclose(Z, np.array([[1, 0], [0, -1]])).all()
    
    H = gate_matrix['H']
    assert np.isclose(H, (1.0/np.sqrt(2))*np.array([[1, 1], [1, -1]])).all()
    S = gate_matrix['S']
    assert np.isclose(S, np.array([[1.0, 0], [0, 1j]])).all()
    T = gate_matrix['T']
    assert np.isclose(T, np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]])).all()

    phi_range = np.linspace(0, 2 * np.pi, 120)
    for phi in phi_range:
        assert np.isclose(gate_matrix['PHASE'](phi), np.array([[1.0, 0.0], [0.0, np.exp(1j*phi)]])).all()
        assert np.isclose(gate_matrix['RX'](phi), np.array([[np.cos(phi/2.0), -1j*np.sin(phi/2.0)],
                                                              [-1j*np.sin(phi/2.0), np.cos(phi/2.0)]])).all()
        assert np.isclose(gate_matrix['RY'](phi), np.array([[np.cos(phi/2.0), -np.sin(phi/2.0)],
                                                              [np.sin(phi/2.0), np.cos(phi/2.0)]])).all()
        assert np.isclose(gate_matrix['RZ'](phi), np.array([[np.cos(phi/2.0) - 1j*np.sin(phi/2.0), 0],
                                                              [0, np.cos(phi/2.0) + 1j*np.sin(phi/2.0)]])).all()

    assert np.isclose(gate_matrix['CNOT'], np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 1],
                                            [0, 0, 1, 0]])).all()
    assert np.isclose(gate_matrix['CCNOT'], np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 1, 0]])).all()

    # insert more tests here
    # TODO
